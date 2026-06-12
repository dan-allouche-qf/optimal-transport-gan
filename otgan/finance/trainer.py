"""Step-based OT-GAN trainer for log-return paths — the same engine, smaller.

``ReturnsTrainer`` is intentionally a standalone class, not a ``BaseTrainer``
subclass: the image trainer's machinery (epochs over torchvision loaders,
FID/KID, sample grids, TensorBoard) is image-specific, while a returns model
trains in a few thousand steps on a tensor that fits in memory. What it shares
with the image pipeline it shares by direct import — ``otgan.energy
.compute_loss`` (and through it ``otgan.sinkhorn.sinkhorn``),
``otgan.ema.EMAGenerator``, ``otgan.data.split_real_pair``,
``otgan.device.seed_everything`` — so the objective and the critic-sign
semantics literally cannot diverge between images and markets.

Evaluation uses a debiased Sinkhorn divergence (Genevay, Peyre & Cuturi 2018;
Feydy et al. 2019) on *raw paths* with a squared-Euclidean cost. Per the
finance-track convention (see ``otgan/finance/config.py``), that cost matrix
is divided by its mean — ``C = C / C.mean()`` — before the solver, so epsilon
is dimensionless: ``epsilon=1.0`` means the same thing here as in the
cosine-cost-on-embeddings regime, and the metric is invariant to the overall
scale of the paths (standardized or not).
"""

from __future__ import annotations

import csv
from collections.abc import Iterator

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from otgan.data import split_real_pair
from otgan.device import resolve_device, seed_everything
from otgan.ema import EMAGenerator
from otgan.energy import EnergyTerms, compute_loss
from otgan.finance.config import FinanceConfig
from otgan.finance.models1d import build_finance_models
from otgan.finance.simulate import gbm_paths, gjr_garch_paths, heston_paths
from otgan.paths import resolve
from otgan.sinkhorn import sinkhorn


def _set_requires_grad(module: torch.nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(flag)


def _raw_path_divergence(
    x: torch.Tensor, y: torch.Tensor, epsilon: float, iters: int
) -> torch.Tensor:
    """Debiased Sinkhorn divergence between raw path batches ``(n, T)``.

    ``S = W(x, y) - 1/2 W(x, x) - 1/2 W(y, y)`` where each ``W`` is the
    entropic cost ``<M, C>`` under a squared-Euclidean cost normalized by its
    mean (``C = C / C.mean()``) — the dimensionless-epsilon convention. Each
    term is invariant under a common rescaling of both inputs, so the metric
    reads the same on standardized and de-standardized paths.
    """

    def w(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        C = torch.cdist(u, v) ** 2
        C = C / C.mean()
        n1, n2 = C.shape
        a = torch.full((n1,), 1.0 / n1, device=C.device, dtype=C.dtype)
        b = torch.full((n2,), 1.0 / n2, device=C.device, dtype=C.dtype)
        M = sinkhorn(a, b, C, epsilon, iters)
        return (M * C).sum()

    return w(x, y) - 0.5 * w(x, x) - 0.5 * w(y, y)


class ReturnsTrainer:
    """OT-GAN on 1D return paths: ``n_steps`` optimization steps, CPU-sized."""

    ckpt_name = "returns_gan.pt"

    def __init__(self, cfg: FinanceConfig):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        seed_everything(cfg.seed)

        train, held_out = self._build_target(cfg)
        # Standardize to unit variance; the scale travels with the checkpoint
        # so sample() can de-standardize (the generator head is linear).
        self.scale = float(train.std())
        self.train_paths = (train / self.scale).unsqueeze(1)  # (n_train, 1, T)
        self.eval_paths = (held_out / self.scale).unsqueeze(1)  # (n_eval, 1, T)

        self.generator, self.critic = build_finance_models(cfg)
        self.generator.to(self.device)
        self.critic.to(self.device)
        self.ema = EMAGenerator(self.generator, cfg.ema_decay).to(self.device)
        self.g_opt = optim.Adam(
            self.generator.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2)
        )
        self.c_opt = optim.Adam(
            self.critic.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2)
        )

        self.dataloader = DataLoader(
            TensorDataset(self.train_paths),
            batch_size=2 * cfg.batch_size,  # split into two independent halves per step
            shuffle=True,
            drop_last=True,  # keep every cost matrix exactly B x B
            num_workers=cfg.num_workers,
        )
        self.step = 0
        self.history: list[dict] = []

    # ---- target construction --------------------------------------------
    @staticmethod
    def _build_target(cfg: FinanceConfig) -> tuple[torch.Tensor, torch.Tensor]:
        """(train, held_out) raw log-return tensors of shape ``(n, seq_len)``."""
        if cfg.target == "csv":
            if cfg.csv_path is None:  # validated in FinanceConfig; narrows for mypy
                raise ValueError("target='csv' requires csv_path to be set")
            series = torch.from_numpy(np.loadtxt(cfg.csv_path, delimiter=",")).float().reshape(-1)
            n_windows = series.numel() // cfg.seq_len
            if n_windows < 2:
                raise ValueError(
                    f"csv at {cfg.csv_path!r} yields {n_windows} window(s) of seq_len="
                    f"{cfg.seq_len}; need at least 2 (one train, one held out)"
                )
            windows = series[: n_windows * cfg.seq_len].view(n_windows, cfg.seq_len)
            n_eval = max(1, min(cfg.n_eval_paths, n_windows // 2))
            return windows[: n_windows - n_eval], windows[n_windows - n_eval :]

        def sim(n: int, seed: int) -> torch.Tensor:
            if cfg.target == "gbm":
                return gbm_paths(n, cfg.seq_len, cfg.mu, cfg.sigma, cfg.dt, seed)
            if cfg.target == "heston":
                return heston_paths(
                    n,
                    cfg.seq_len,
                    cfg.mu,
                    cfg.dt,
                    cfg.heston_kappa,
                    cfg.heston_theta,
                    cfg.heston_xi,
                    cfg.heston_rho,
                    seed=seed,
                )
            return gjr_garch_paths(
                n,
                cfg.seq_len,
                cfg.garch_omega,
                cfg.garch_alpha,
                cfg.garch_beta,
                cfg.garch_gamma,
                seed,
            )

        # Held-out paths come from an independent seed: same process, fresh draws.
        return sim(cfg.n_train_paths, cfg.seed), sim(cfg.n_eval_paths, cfg.seed + 1)

    # ---- latent / sampling -----------------------------------------------
    def sample_z(self, n: int) -> torch.Tensor:
        return torch.randn(n, self.cfg.z_dim, device=self.device)

    @torch.no_grad()
    def sample(self, n: int) -> torch.Tensor:
        """``(n, 1, seq_len)`` de-standardized paths from the EMA generator (CPU)."""
        return (self.ema.model(self.sample_z(n)) * self.scale).detach().cpu()

    # ---- a single optimization step (same sign convention as the image run)
    def _optimize(self, real_1: torch.Tensor, real_2: torch.Tensor, critic_step: bool):
        """Line-for-line the sign convention of ``OTGANTrainer._optimize``: the
        critic ASCENDS the minibatch OT objective, the generator DESCENDS it,
        and ``critic_sign=False`` reproduces the original bug — so the 1D
        replay of the headline critic-sign ablation runs in minutes instead of
        16 hours."""
        z1, z2 = self.sample_z(real_1.shape[0]), self.sample_z(real_2.shape[0])
        if critic_step:
            _set_requires_grad(self.critic, True)
            _set_requires_grad(self.generator, False)
        else:
            _set_requires_grad(self.critic, False)
            _set_requires_grad(self.generator, True)

        fake_1, fake_2 = self.generator(z1), self.generator(z2)
        embeddings = (
            self.critic(real_1),
            self.critic(real_2),
            self.critic(fake_1),
            self.critic(fake_2),
        )
        terms = compute_loss(embeddings, self.cfg.epsilon, self.cfg.sinkhorn_iters, self.cfg.loss)

        if critic_step:
            # Critic MAXIMIZES the objective -> ascend (minimize its negation).
            # critic_sign=False reproduces the original bug (critic descends).
            objective = -terms.total if self.cfg.critic_sign else terms.total
            self.c_opt.zero_grad(set_to_none=True)
            objective.backward()
            self.c_opt.step()
        else:
            # Generator MINIMIZES the objective -> descend.
            self.g_opt.zero_grad(set_to_none=True)
            terms.total.backward()
            self.g_opt.step()
            self.ema.update(self.generator)
        return terms

    # ---- fit ---------------------------------------------------------------
    def _cycle(self) -> Iterator[list[torch.Tensor]]:
        while True:  # re-enters the loader so each pass is reshuffled
            yield from self.dataloader

    def fit(self) -> list[dict]:
        cfg = self.cfg
        cfg.print_config()
        eval_dir = resolve(cfg.eval_dir, create=True)
        self.generator.train()
        self.critic.train()

        batches = self._cycle()
        acc: dict[str, float] = {}
        n_acc = 0
        while self.step < cfg.n_steps:
            (batch,) = next(batches)
            batch = batch.to(self.device, dtype=torch.float32)
            real_1, real_2 = split_real_pair(batch)
            critic_step = (self.step + 1) % cfg.g2c_ratio == 0
            terms: EnergyTerms = self._optimize(real_1, real_2, critic_step)
            for k, v in terms.to_floats().items():
                acc[k] = acc.get(k, 0.0) + v
            n_acc += 1
            self.step += 1

            if self.step % cfg.eval_every == 0 or (self.step == cfg.n_steps and n_acc > 0):
                row: dict = {"step": self.step}
                row.update({k: v / n_acc for k, v in acc.items()})
                row["sinkhorn_divergence_metric"] = self._evaluate()
                self.history.append(row)
                line = " ".join(f"{k}={v:+.4f}" for k, v in row.items() if k != "step")
                print(f"step {self.step:>5} | {line}")
                acc, n_acc = {}, 0

        self._write_history_csv(eval_dir)
        self.save_checkpoint()
        return self.history

    @torch.no_grad()
    def _evaluate(self) -> float:
        """Sinkhorn divergence: fresh EMA samples vs held-out target paths."""
        n = min(self.cfg.n_samples, self.eval_paths.shape[0])
        fake = self.ema.model(self.sample_z(n)).squeeze(1).cpu()
        idx = torch.randperm(self.eval_paths.shape[0])[:n]
        real = self.eval_paths[idx].squeeze(1)
        return float(_raw_path_divergence(fake, real, self.cfg.epsilon, self.cfg.sinkhorn_iters))

    # ---- logging / artifacts ----------------------------------------------
    def _write_history_csv(self, eval_dir) -> None:
        if not self.history:
            return
        keys = sorted({k for row in self.history for k in row})
        with open(eval_dir / "history.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.history)

    # ---- checkpoint / resume -----------------------------------------------
    def save_checkpoint(self, name: str | None = None) -> str:
        ckpt_dir = resolve(self.cfg.ckpt_dir, create=True)
        path = ckpt_dir / (name or self.ckpt_name)
        state = {
            "step": self.step,
            "scale": self.scale,
            "rng": torch.get_rng_state(),
            "config": self.cfg.to_dict(),
            "generator": self.generator.state_dict(),
            "critic": self.critic.state_dict(),
            "ema": self.ema.state_dict(),
            "g_opt": self.g_opt.state_dict(),
            "c_opt": self.c_opt.state_dict(),
        }
        torch.save(state, str(path))
        return str(path)

    def load_checkpoint(self, path) -> None:
        ckpt = torch.load(str(path), map_location=self.device, weights_only=False)
        self.generator.load_state_dict(ckpt["generator"])
        self.critic.load_state_dict(ckpt["critic"])
        self.ema.load_state_dict(ckpt["ema"])
        self.g_opt.load_state_dict(ckpt["g_opt"])
        self.c_opt.load_state_dict(ckpt["c_opt"])
        self.scale = float(ckpt["scale"])
        self.step = int(ckpt["step"])
        torch.set_rng_state(ckpt["rng"].to("cpu"))
