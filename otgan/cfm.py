"""Optimal-transport conditional flow matching (OT-CFM) — 2018 vs 2024.

Flow matching (Lipman et al., ICLR 2023) trains a vector field ``v(x, t)`` by
plain regression: draw noise ``x0``, data ``x1`` and ``t ~ U(0, 1)``, then
regress ``v((1-t) x0 + t x1, t)`` onto the straight-line velocity ``x1 - x0``.
With independent ``(x0, x1)`` pairs (I-CFM) the conditional paths cross and the
marginal flow is curved; Tong et al. (TMLR 2024, "Improving and generalizing
flow-based generative models with minibatch optimal transport") pair ``x0``
with ``x1`` through a minibatch OT plan instead, which straightens the learned
flow and cuts the integration error at sampling time.

The point of this module: the repo's own ``otgan.sinkhorn.sinkhorn`` powers
both the 2018 GAN and the 2024 flow matcher. In OT-GAN (Salimans et al. 2018)
that solver prices minibatches inside an adversarial energy distance; here the
very same log-domain loop pairs noise with data for a single MSE objective —
no critic, no minimax, no torchcfm dependency. ``CFMTrainer`` plugs into the
same ``BaseTrainer`` harness (fixed-noise grids, FID/KID/IS, checkpoints) as
the GAN trainers, so the two eras are compared like-for-like.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from otgan.ema import EMAGenerator
from otgan.sinkhorn import sinkhorn
from otgan.trainer import BaseTrainer


# ---------------------------------------------------------------------------
# Vector-field network
# ---------------------------------------------------------------------------
def sinusoidal_embedding(t: torch.Tensor, dim: int = 128) -> torch.Tensor:
    """Transformer-style sinusoidal features of ``t``, shape ``(B,) -> (B, dim)``.

    ``t`` lives in [0, 1]; it is scaled by 1000 so the geometric frequency
    ladder (1 .. 1/10000) resolves it the way diffusion timesteps are resolved.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / (half - 1)
    )
    args = 1000.0 * t.to(torch.float32)[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class _TimeBlock(nn.Module):
    """Two (GroupNorm(8) + SiLU + 3x3 conv) layers with the time embedding
    added between them through a small linear projection (FiLM-lite: add)."""

    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        x = self.conv1(F.silu(self.norm1(x)))
        x = x + self.t_proj(temb)[:, :, None, None]
        return self.conv2(F.silu(self.norm2(x)))


class SmallUNet(nn.Module):
    """A deliberately small U-Net vector field ``v(x, t)`` (~0.7M params at base=32).

    Encoder blocks at widths ``base, 2*base, 4*base`` with stride-2 conv
    downsamples between, a bottleneck block, and a mirrored decoder using
    nearest-neighbor upsampling + conv with skip connections. The sinusoidal
    time embedding goes through an MLP and is added per block.
    """

    def __init__(self, channels: int = 1, base: int = 32, t_dim: int = 128):
        super().__init__()
        self.t_dim = t_dim
        self.t_mlp = nn.Sequential(nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        self.stem = nn.Conv2d(channels, base, 3, padding=1)
        self.enc1 = _TimeBlock(base, base, t_dim)
        self.down1 = nn.Conv2d(base, 2 * base, 3, stride=2, padding=1)
        self.enc2 = _TimeBlock(2 * base, 2 * base, t_dim)
        self.down2 = nn.Conv2d(2 * base, 4 * base, 3, stride=2, padding=1)
        self.mid = _TimeBlock(4 * base, 4 * base, t_dim)
        self.up1 = nn.Conv2d(4 * base, 2 * base, 3, padding=1)
        self.dec2 = _TimeBlock(4 * base, 2 * base, t_dim)  # cat with enc2 skip
        self.up2 = nn.Conv2d(2 * base, base, 3, padding=1)
        self.dec1 = _TimeBlock(2 * base, base, t_dim)  # cat with enc1 skip
        self.out_norm = nn.GroupNorm(8, base)
        self.out_conv = nn.Conv2d(base, channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """``x``: images ``(B, C, H, W)``; ``t``: times in [0, 1], shape ``(B,)``."""
        temb = self.t_mlp(sinusoidal_embedding(t, self.t_dim))
        h1 = self.enc1(self.stem(x), temb)  # B x base   x 32 x 32
        h2 = self.enc2(self.down1(h1), temb)  # B x 2*base x 16 x 16
        h = self.mid(self.down2(h2), temb)  # B x 4*base x  8 x  8
        h = self.up1(F.interpolate(h, scale_factor=2.0, mode="nearest"))
        h = self.dec2(torch.cat([h, h2], dim=1), temb)  # B x 2*base x 16 x 16
        h = self.up2(F.interpolate(h, scale_factor=2.0, mode="nearest"))
        h = self.dec1(torch.cat([h, h1], dim=1), temb)  # B x base   x 32 x 32
        return self.out_conv(F.silu(self.out_norm(h)))


# ---------------------------------------------------------------------------
# Minibatch OT coupling
# ---------------------------------------------------------------------------
def minibatch_coupling(
    x0: torch.Tensor,
    x1: torch.Tensor,
    mode: str = "sinkhorn",
    eps: float = 0.05,
    iters: int = 50,
) -> torch.Tensor:
    """Reorder ``x1`` so that row ``i`` is the OT partner of ``x0[i]``.

    The cost is the squared euclidean distance between flattened samples,
    divided by the feature dimension (a per-coordinate MSE) and then by its
    own mean. The double normalization makes ``eps`` dimensionless — the same
    value means the same plan sharpness regardless of image size or pixel
    scale, so a single default transfers across datasets.

    Modes:
      - ``'none'``: identity pairing (independent CFM, the Lipman et al. baseline);
      - ``'sinkhorn'``: entropic plan from ``otgan.sinkhorn.sinkhorn`` (the same
        solver the 2018 GAN uses), partners drawn per row with ``torch.multinomial``
        so the pairing is a sample from the plan, as in Tong et al. (2024);
      - ``'exact'``: unregularized plan via POT's network simplex (``ot.emd``),
        partners by row argmax (the plan is a scaled permutation matrix).
    """
    if mode == "none":
        return x1
    f0 = x0.flatten(1)
    f1 = x1.flatten(1)
    C = torch.cdist(f0, f1).pow(2) / f0.shape[1]
    C = C / C.mean().clamp_min(1e-12)
    n0, n1 = C.shape
    a = torch.full((n0,), 1.0 / n0, device=C.device, dtype=C.dtype)
    b = torch.full((n1,), 1.0 / n1, device=C.device, dtype=C.dtype)
    if mode == "sinkhorn":
        plan = sinkhorn(a, b, C, eps, iters)
        # multinomial renormalizes each row; clamp guards fully-underflowed rows.
        idx = torch.multinomial(plan.clamp_min(1e-30), 1).squeeze(1)
    elif mode == "exact":
        try:
            import ot
        except ImportError as exc:  # pragma: no cover - exercised only without POT
            raise ImportError(
                "cfm_coupling='exact' requires the POT package (pip install pot); "
                "use cfm_coupling='sinkhorn' for the dependency-free entropic solver"
            ) from exc
        a_np, b_np = a.double().cpu().numpy(), b.double().cpu().numpy()
        plan_np = ot.emd(a_np, b_np, C.double().cpu().numpy())
        idx = torch.from_numpy(plan_np.argmax(axis=1)).to(x1.device)
    else:
        raise ValueError(f"mode must be 'sinkhorn', 'exact' or 'none', got {mode!r}")
    return x1[idx]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class CFMTrainer(BaseTrainer):
    """Trains the OT-CFM vector field; sampling Euler-integrates it from noise."""

    ckpt_name = "ot_cfm.pt"

    def _build(self) -> None:
        cfg = self.cfg
        self.model = SmallUNet(channels=cfg.channels).to(self.device)
        # BaseTrainer expects ``generator``/``ema``; for a flow the "generator"
        # is the vector field — sample() below integrates it instead of calling
        # it once, so the shared grid/FID harness works unchanged.
        self.generator = self.model
        self.ema = EMAGenerator(self.model, cfg.ema_decay).to(self.device)
        self.opt = optim.Adam(
            self.model.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2)
        )

    def _modules(self) -> list:
        return [self.model]

    # ---- latent / sampling ---------------------------------------------
    def sample_z(self, n: int) -> torch.Tensor:
        """Image-shaped noise: the flow starts at ``x(0) ~ N(0, I)`` in pixel space."""
        cfg = self.cfg
        return torch.randn(n, cfg.channels, cfg.image_size, cfg.image_size, device=self.device)

    @torch.no_grad()
    def sample(self, n: int | None = None, use_ema: bool = True, z=None) -> torch.Tensor:
        """Euler-integrate ``dx/dt = v(x, t)`` from t=0 to 1; CPU images in [-1, 1]."""
        net = self.ema.model if use_ema else self.model
        was_training = net.training
        net.eval()
        if z is None:
            z = self.sample_z(n if n is not None else self.cfg.n_samples)
        x = z.to(self.device)
        steps = self.cfg.ode_steps
        dt = 1.0 / steps
        for k in range(steps):
            t = torch.full((x.shape[0],), k * dt, device=self.device)
            x = x + dt * net(x, t)
        if was_training:
            net.train()
        return x.clamp(-1.0, 1.0).cpu()

    # ---- one optimization step -------------------------------------------
    def _step(self, batch: torch.Tensor, i: int) -> dict:
        cfg = self.cfg
        x1 = batch  # the full 2B batch: the flow has no critic to split for
        x0 = torch.randn_like(x1)
        x1 = minibatch_coupling(x0, x1, cfg.cfm_coupling, cfg.cfm_eps, cfg.sinkhorn_iters)
        t = torch.rand(x1.shape[0], device=self.device)
        tb = t[:, None, None, None]
        xt = (1.0 - tb) * x0 + tb * x1
        loss = F.mse_loss(self.model(xt, t), x1 - x0)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        self.ema.update(self.model)
        return {"cfm_loss": loss.detach().item()}

    def _format_line(self, means: dict) -> str:
        return f"cfm_loss={means['cfm_loss']:.4f}"

    # ---- checkpoint state ------------------------------------------------
    def _checkpoint_state(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "opt": self.opt.state_dict(),
        }

    def _load_checkpoint_state(self, ckpt: dict) -> None:
        self.model.load_state_dict(ckpt["model"])
        self.ema.load_state_dict(ckpt["ema"])
        self.opt.load_state_dict(ckpt["opt"])
