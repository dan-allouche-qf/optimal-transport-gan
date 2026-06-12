"""Behavioral tests for the trainer — most importantly, that the critic ASCENDS
the energy distance and the generator DESCENDS it (the audited sign fix)."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from otgan.config import Config
from otgan.energy import energy_distance
from otgan.trainer import OTGANTrainer


def _direction_cfg(**kw):
    base = dict(
        dataset="mnist",
        channels=1,
        batch_size=16,
        z_dim=100,
        sinkhorn_iters=20,
        epsilon=1.0,
        n_epochs=1,
        fid_every=0,
        device="cpu",
        seed=0,
        num_workers=0,
    )
    base.update(kw)
    return Config(**base)


def _fixed_z_patch(trainer, z1, z2):
    """Make sample_z deterministic: alternate z1, z2 on successive calls."""
    seq = [z1, z2]
    state = {"k": 0}

    def fixed(n):
        z = seq[state["k"] % 2]
        state["k"] += 1
        return z[:n]

    trainer.sample_z = fixed


@torch.no_grad()
def _measure(trainer, r1, r2, z1, z2):
    f1, f2 = trainer.generator(z1), trainer.generator(z2)
    emb = (trainer.critic(r1), trainer.critic(r2), trainer.critic(f1), trainer.critic(f2))
    return float(energy_distance(emb, trainer.cfg.epsilon, trainer.cfg.sinkhorn_iters).total)


def _setup(cfg):
    tr = OTGANTrainer(cfg)
    B = cfg.batch_size
    z1, z2 = torch.randn(B, cfg.z_dim), torch.randn(B, cfg.z_dim)
    _fixed_z_patch(tr, z1, z2)
    r1, r2 = torch.rand(B, 1, 32, 32) * 2 - 1, torch.rand(B, 1, 32, 32) * 2 - 1
    return tr, r1, r2, z1, z2


def test_critic_ascends_energy_distance():
    """With critic_sign=True, repeated critic steps INCREASE D^2 on a held batch."""
    tr, r1, r2, z1, z2 = _setup(_direction_cfg(critic_sign=True))
    before = _measure(tr, r1, r2, z1, z2)
    for _ in range(20):
        tr._optimize(r1, r2, critic_step=True)
    after = _measure(tr, r1, r2, z1, z2)
    assert after > before, f"critic should ascend: {before:.4f} -> {after:.4f}"


def test_generator_descends_energy_distance():
    """Repeated generator steps DECREASE D^2 on a held batch."""
    tr, r1, r2, z1, z2 = _setup(_direction_cfg(critic_sign=True))
    before = _measure(tr, r1, r2, z1, z2)
    for _ in range(20):
        tr._optimize(r1, r2, critic_step=False)
    after = _measure(tr, r1, r2, z1, z2)
    assert after < before, f"generator should descend: {before:.4f} -> {after:.4f}"


def test_critic_sign_false_reproduces_bug():
    """critic_sign=False is the original bug: the critic DESCENDS D^2 (wrong direction)."""
    tr, r1, r2, z1, z2 = _setup(_direction_cfg(critic_sign=False))
    before = _measure(tr, r1, r2, z1, z2)
    for _ in range(20):
        tr._optimize(r1, r2, critic_step=True)
    after = _measure(tr, r1, r2, z1, z2)
    assert after < before, f"buggy critic should descend: {before:.4f} -> {after:.4f}"


def _fake_loader(cfg, n_batches=4):
    n = n_batches * 2 * cfg.batch_size
    imgs = torch.rand(n, cfg.channels, 32, 32) * 2 - 1
    labels = torch.zeros(n, dtype=torch.long)
    return DataLoader(TensorDataset(imgs, labels), batch_size=2 * cfg.batch_size, drop_last=True)


def test_fit_end_to_end_offline(tmp_path, monkeypatch):
    """Full fit() on a synthetic loader: runs, checkpoints, and resumes."""
    monkeypatch.setenv("OT_GAN_ROOT", str(tmp_path))
    cfg = _direction_cfg(
        batch_size=8,
        sinkhorn_iters=5,
        n_epochs=1,
        max_batches=3,
        eval_dir="eval",
        ckpt_dir="ckpt",
        log_dir="logs",
    )
    tr = OTGANTrainer(cfg, dataloader=_fake_loader(cfg))
    history = tr.fit()
    assert len(history) == 1 and torch.isfinite(torch.tensor(history[0]["energy_distance"]))
    assert (tmp_path / "eval" / "history.csv").exists()
    ckpt = tmp_path / "ckpt" / "ot_gan.pt"
    assert ckpt.exists()

    # Resume into a fresh trainer and confirm weights round-trip exactly.
    tr2 = OTGANTrainer(cfg, dataloader=_fake_loader(cfg))
    tr2.load_checkpoint(ckpt)
    for p1, p2 in zip(tr.generator.parameters(), tr2.generator.parameters(), strict=True):
        assert torch.equal(p1, p2)
    assert tr2.start_epoch == 1


def test_load_slim_release_export(tmp_path, monkeypatch):
    """A slim export (EMA weights + config only) loads and samples via the CLI path."""
    monkeypatch.setenv("OT_GAN_ROOT", str(tmp_path))
    cfg = _direction_cfg(sinkhorn_iters=5)
    tr = OTGANTrainer(cfg)
    slim = {"ema": tr.ema.state_dict(), "config": cfg.to_dict(), "epoch": 0}
    path = tmp_path / "slim.pt"
    torch.save(slim, path)

    tr2 = OTGANTrainer(cfg)
    tr2.load_checkpoint(path)
    for p1, p2 in zip(tr.ema.model.parameters(), tr2.ema.model.parameters(), strict=True):
        assert torch.equal(p1, p2)
    imgs = tr2.sample(4)
    assert imgs.shape == (4, 1, 32, 32) and torch.isfinite(imgs).all()
