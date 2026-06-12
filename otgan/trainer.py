"""Training loops.

``BaseTrainer`` owns everything model-agnostic — seeding, device placement,
the epoch loop, FID/KID evaluation, best-checkpoint tracking, fixed-noise
sample grids, TensorBoard + CSV logging, checkpoint/resume plumbing — so the
OT-GAN, the DCGAN baseline and the OT-CFM flow matcher all share the exact
same evaluation harness and differ only in how they build models and take an
optimization step.

The central OT-GAN correction over the original notebook lives in
``OTGANTrainer._optimize``: the critic ASCENDS the minibatch energy distance
(maximizes it) while the generator DESCENDS it (minimizes it). The original
code descended a single loss with both optimizers, training the critic in the
wrong direction.
"""

import csv
import time

import torch
import torch.optim as optim
from torchvision.utils import make_grid, save_image

from otgan.data import build_dataloader, split_real_pair
from otgan.device import resolve_device, seed_everything
from otgan.ema import EMAGenerator
from otgan.energy import compute_loss
from otgan.models import build_models
from otgan.paths import resolve

_SUMMARY_WRITER: "type | None"
try:  # TensorBoard is optional at runtime
    from torch.utils.tensorboard import SummaryWriter as _SUMMARY_WRITER
except Exception:  # pragma: no cover
    _SUMMARY_WRITER = None


def _set_requires_grad(module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(flag)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Map a generator/real tensor from [-1, 1] back to [0, 1] for display/saving."""
    return (x.clamp(-1, 1) + 1.0) / 2.0


class BaseTrainer:
    """Template: subclasses implement ``_build``, ``_step`` and checkpoint state."""

    ckpt_name = "model.pt"
    # Set by every subclass in _build(); declared here so the shared sampling
    # and evaluation paths type-check.
    generator: torch.nn.Module
    ema: EMAGenerator

    def __init__(self, cfg, dataloader=None):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        seed_everything(cfg.seed)

        self._build()  # models / optimizers / EMA — sets self.generator at least

        self.dataloader = dataloader  # lazily built in fit() if None
        self.start_epoch = 0
        self.history: list[dict] = []
        self.best_fid = float("inf")

        # Fixed latents for a stable sample-grid evolution across epochs.
        self.fixed_z = self.sample_z(cfg.n_samples)
        self._writer = None
        self._evaluator = None

    # ---- hooks for subclasses ------------------------------------------
    def _build(self) -> None:
        raise NotImplementedError

    def _step(self, batch: torch.Tensor, i: int) -> dict:
        """One optimization step on a ``2B`` image batch; returns float logs."""
        raise NotImplementedError

    def _modules(self) -> list:
        """Modules toggled to train mode each epoch."""
        raise NotImplementedError

    def _checkpoint_state(self) -> dict:
        raise NotImplementedError

    def _load_checkpoint_state(self, ckpt: dict) -> None:
        raise NotImplementedError

    # ---- latent / sampling ---------------------------------------------
    def sample_z(self, n: int) -> torch.Tensor:
        if self.cfg.latent == "uniform":
            z = torch.rand(n, self.cfg.z_dim, device=self.device) * 2.0 - 1.0
        else:
            z = torch.randn(n, self.cfg.z_dim, device=self.device)
        return z

    @torch.no_grad()
    def sample(self, n: int | None = None, use_ema: bool = True, z=None) -> torch.Tensor:
        """Generate images in [-1, 1] on CPU (from the EMA generator by default)."""
        gen = self.ema.model if use_ema else self.generator
        was_training = gen.training
        gen.eval()
        if z is None:
            z = self.sample_z(n if n is not None else self.cfg.n_samples)
        imgs = gen(z).detach().cpu()
        if was_training:
            gen.train()
        return imgs

    # ---- epoch / fit -----------------------------------------------------
    def _run_epoch(self, epoch: int) -> dict:
        for m in self._modules():
            m.train()
        acc: dict[str, float] = {}
        n = 0
        for i, (batch, _) in enumerate(self.dataloader):
            if self.cfg.max_batches is not None and i >= self.cfg.max_batches:
                break
            batch = batch.to(self.device, dtype=torch.float32)
            logs = self._step(batch, i)
            for k, v in logs.items():
                acc[k] = acc.get(k, 0.0) + v
            n += 1
        return {k: v / max(n, 1) for k, v in acc.items()}

    def fit(self):
        if self.dataloader is None:
            self.dataloader = build_dataloader(self.cfg, train=True)
        self.cfg.print_config()
        eval_dir = resolve(self.cfg.eval_dir, create=True)

        for epoch in range(self.start_epoch, self.cfg.n_epochs):
            t0 = time.time()
            means = self._run_epoch(epoch)
            means["epoch"] = epoch
            means["seconds"] = time.time() - t0

            if self.cfg.fid_every and (epoch + 1) % self.cfg.fid_every == 0:
                means.update(self._evaluate())

            self._log(epoch, means)
            if self.cfg.log_step and epoch % self.cfg.log_step == 0:
                self._save_sample_grid(epoch, eval_dir)
            self.save_checkpoint(epoch)
            if means.get("fid") is not None and means["fid"] < self.best_fid:
                # Keep the best-FID weights: the rolling checkpoint is
                # overwritten every epoch, and GAN FID is not monotonic.
                self.best_fid = means["fid"]
                self.save_checkpoint(epoch, name=self.best_ckpt_name())
            self.history.append(means)

        self._write_history_csv(eval_dir)
        self._plot_artifacts(eval_dir)
        if self._writer is not None:
            self._writer.close()
        return self.history

    @classmethod
    def best_ckpt_name(cls) -> str:
        stem, dot, ext = cls.ckpt_name.rpartition(".")
        return f"{stem}_best{dot}{ext}"

    def _plot_artifacts(self, eval_dir) -> None:
        try:
            from otgan.plotting import plot_curves, plot_sample_grid_evolution

            plot_curves(self.history, str(eval_dir / "curves.png"))
            plot_sample_grid_evolution(eval_dir, str(eval_dir / "evolution.png"))
        except Exception as exc:  # pragma: no cover - plotting must never crash training
            print(f"[warn] plotting skipped: {exc}")

    # ---- evaluation (FID/KID/IS) ----------------------------------------
    def _evaluate(self) -> dict:
        from otgan.metrics import FIDISEvaluator  # lazy: pulls torchmetrics

        if self._evaluator is None:
            self._evaluator = FIDISEvaluator(self.cfg, self.device)
        return self._evaluator.evaluate(self)

    # ---- logging / artifacts ---------------------------------------------
    def _format_line(self, means: dict) -> str:
        keys = [k for k in means if k not in ("epoch", "seconds")]
        return " ".join(f"{k}={means[k]:+.4f}" for k in keys)

    def _log(self, epoch: int, means: dict) -> None:
        msg = f"epoch {epoch:>3} | {self._format_line(means)} | {means['seconds']:.1f}s"
        if "fid" in means:
            msg += f" | FID={means['fid']:.2f}"
            if "kid_mean" in means:
                msg += f" KIDx1e3={1000 * means['kid_mean']:.2f}"
            if "is_mean" in means:
                msg += f" IS={means['is_mean']:.2f}"
        print(msg)
        writer = self._writer_obj()
        if writer is not None:
            for k, v in means.items():
                if k != "epoch":
                    writer.add_scalar(k, v, epoch)

    def _writer_obj(self):
        if self._writer is None and _SUMMARY_WRITER is not None:
            self._writer = _SUMMARY_WRITER(str(resolve(self.cfg.log_dir, create=True)))
        return self._writer

    def _save_sample_grid(self, epoch: int, eval_dir) -> None:
        imgs = denormalize(self.sample(z=self.fixed_z))
        grid = make_grid(imgs, nrow=8)
        save_image(grid, str(eval_dir / f"sample_epoch{epoch:03d}.png"))
        writer = self._writer_obj()
        if writer is not None:
            writer.add_image("samples", grid, epoch)

    def _write_history_csv(self, eval_dir) -> None:
        if not self.history:
            return
        keys = sorted({k for row in self.history for k in row})
        with open(eval_dir / "history.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.history)

    # ---- checkpoint / resume ----------------------------------------------
    def save_checkpoint(self, epoch: int, name: str | None = None) -> str:
        ckpt_dir = resolve(self.cfg.ckpt_dir, create=True)
        path = ckpt_dir / (name or self.ckpt_name)
        state = {
            "epoch": epoch,
            "rng": torch.get_rng_state(),
            "config": self.cfg.to_dict(),
        }
        state.update(self._checkpoint_state())
        torch.save(state, str(path))
        return str(path)

    def load_checkpoint(self, path) -> None:
        ckpt = torch.load(str(path), map_location=self.device, weights_only=False)
        self._load_checkpoint_state(ckpt)
        if isinstance(ckpt, dict) and "rng" in ckpt:
            torch.set_rng_state(ckpt["rng"].to("cpu"))
            self.start_epoch = ckpt["epoch"] + 1


class OTGANTrainer(BaseTrainer):
    ckpt_name = "ot_gan.pt"

    def _build(self) -> None:
        cfg = self.cfg
        self.generator, self.critic = build_models(cfg)
        self.generator.to(self.device)
        self.critic.to(self.device)
        self.ema = EMAGenerator(self.generator, cfg.ema_decay).to(self.device)
        self.g_opt = optim.Adam(
            self.generator.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2)
        )
        self.c_opt = optim.Adam(
            self.critic.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2)
        )

    def _modules(self) -> list:
        return [self.generator, self.critic]

    # ---- a single optimization step (THE critic-sign fix) -------------
    def _optimize(self, real_1, real_2, critic_step: bool):
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
            # Critic MAXIMIZES the energy distance -> ascend (minimize its negation).
            # critic_sign=False reproduces the original bug (critic descends).
            objective = -terms.total if self.cfg.critic_sign else terms.total
            self.c_opt.zero_grad(set_to_none=True)
            objective.backward()
            self.c_opt.step()
        else:
            # Generator MINIMIZES the energy distance -> descend.
            self.g_opt.zero_grad(set_to_none=True)
            terms.total.backward()
            self.g_opt.step()
            self.ema.update(self.generator)
        return terms

    def _step(self, batch: torch.Tensor, i: int) -> dict:
        real_1, real_2 = split_real_pair(batch)
        critic_step = (i + 1) % self.cfg.g2c_ratio == 0
        return self._optimize(real_1, real_2, critic_step).to_floats()

    def _format_line(self, means: dict) -> str:
        return (
            f"D^2={means['energy_distance']:+.4f} "
            f"| cross={means['cross']:.4f} rr={means['real_real']:.4f} "
            f"ff={means['fake_fake']:.4f}"
        )

    # ---- checkpoint state ------------------------------------------------
    def _checkpoint_state(self) -> dict:
        return {
            "generator": self.generator.state_dict(),
            "critic": self.critic.state_dict(),
            "ema": self.ema.state_dict(),
            "g_opt": self.g_opt.state_dict(),
            "c_opt": self.c_opt.state_dict(),
        }

    def _load_checkpoint_state(self, ckpt) -> None:
        if isinstance(ckpt, dict) and "generator" in ckpt:
            self.generator.load_state_dict(ckpt["generator"])
            self.critic.load_state_dict(ckpt["critic"])
            self.ema.load_state_dict(ckpt["ema"])
            self.g_opt.load_state_dict(ckpt["g_opt"])
            self.c_opt.load_state_dict(ckpt["c_opt"])
        elif isinstance(ckpt, dict) and "ema" in ckpt:
            # Slim release export (scripts/export_generator.py): EMA generator
            # weights only — load them into both the raw and EMA generators so
            # sample() works with either use_ema setting.
            self.generator.load_state_dict(ckpt["ema"])
            self.ema.load_state_dict(ckpt["ema"])
        else:  # back-compat: a bare generator state_dict
            self.generator.load_state_dict(ckpt)
            self.ema = EMAGenerator(self.generator, self.cfg.ema_decay).to(self.device)
            self.start_epoch = 0


def build_trainer(cfg) -> BaseTrainer:
    """Instantiate the trainer for ``cfg.model`` ('otgan' | 'dcgan' | 'cfm')."""
    if cfg.model == "dcgan":
        from otgan.baselines import DCGANTrainer  # lazy

        return DCGANTrainer(cfg)
    if cfg.model == "cfm":
        from otgan.cfm import CFMTrainer  # lazy

        return CFMTrainer(cfg)
    return OTGANTrainer(cfg)
