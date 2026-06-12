"""Smoke tests for the DCGAN calibration baseline — fast, offline, CPU-only.

Mirrors the synthetic-loader pattern of ``test_trainer_smoke.py``: a fake
``2B``-image dataloader stands in for MNIST so ``fit()`` exercises the full
shared harness (epoch loop, history CSV, checkpoints) without any download.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from otgan.baselines import DCGANDiscriminator, DCGANGenerator, DCGANTrainer
from otgan.config import Config
from otgan.trainer import build_trainer


def _tiny_cfg(**kw):
    base = dict(
        model="dcgan",
        dataset="mnist",
        channels=1,
        batch_size=8,
        z_dim=100,
        n_epochs=1,
        max_batches=3,
        fid_every=0,
        n_samples=4,
        device="cpu",
        seed=0,
        num_workers=0,
    )
    base.update(kw)
    return Config(**base)


def _fake_loader(cfg, n_batches=4):
    n = n_batches * 2 * cfg.batch_size
    imgs = torch.rand(n, cfg.channels, 32, 32) * 2 - 1
    labels = torch.zeros(n, dtype=torch.long)
    return DataLoader(TensorDataset(imgs, labels), batch_size=2 * cfg.batch_size, drop_last=True)


def test_generator_and_discriminator_shapes():
    gen = DCGANGenerator(z_dim=100, channels=1)
    disc = DCGANDiscriminator(channels=1)
    imgs = gen(torch.randn(4, 100))
    assert imgs.shape == (4, 1, 32, 32)
    assert imgs.min() >= -1.0 and imgs.max() <= 1.0  # Tanh output range
    logits = disc(imgs)
    assert logits.shape == (4,)  # raw logits, one per image


def test_build_trainer_factory_returns_dcgan():
    trainer = build_trainer(_tiny_cfg())
    assert isinstance(trainer, DCGANTrainer)


def test_fit_end_to_end_offline(tmp_path, monkeypatch):
    """Full fit() on a synthetic loader: runs, logs, checkpoints, and round-trips."""
    monkeypatch.setenv("OT_GAN_ROOT", str(tmp_path))
    cfg = _tiny_cfg(eval_dir="eval", ckpt_dir="ckpt", log_dir="logs")
    trainer = DCGANTrainer(cfg, dataloader=_fake_loader(cfg))
    history = trainer.fit()

    assert len(history) == 1
    assert torch.isfinite(torch.tensor(history[0]["d_loss"]))
    assert torch.isfinite(torch.tensor(history[0]["g_loss"]))
    assert (tmp_path / "eval" / "history.csv").exists()
    ckpt = tmp_path / "ckpt" / "dcgan.pt"
    assert ckpt.exists()

    # Resume into a fresh trainer and confirm generator weights round-trip exactly.
    trainer2 = DCGANTrainer(cfg, dataloader=_fake_loader(cfg))
    trainer2.load_checkpoint(ckpt)
    for p1, p2 in zip(trainer.generator.parameters(), trainer2.generator.parameters(), strict=True):
        assert torch.equal(p1, p2)
    assert trainer2.start_epoch == 1


def test_sample_shape_range_and_device(tmp_path, monkeypatch):
    monkeypatch.setenv("OT_GAN_ROOT", str(tmp_path))
    trainer = DCGANTrainer(_tiny_cfg())
    for use_ema in (True, False):
        imgs = trainer.sample(6, use_ema=use_ema)
        assert imgs.shape == (6, 1, 32, 32)
        assert imgs.device.type == "cpu"
        assert imgs.min() >= -1.0 and imgs.max() <= 1.0
