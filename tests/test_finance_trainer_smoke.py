"""Smoke + behavioral tests for the finance trainer: end-to-end fit on the
finance_smoke.yaml values, checkpoint round-trip, de-standardized sampling,
the dimensionless-epsilon convention (C / C.mean()), and the 1D replay of the
critic-sign direction test from tests/test_trainer_smoke.py."""

import math

import torch

from otgan.energy import energy_distance
from otgan.finance.config import FinanceConfig
from otgan.finance.trainer import ReturnsTrainer, _raw_path_divergence


def _smoke_cfg(**kw):
    """The configs/finance_smoke.yaml values, constructed directly."""
    base = dict(
        target="gjr_garch",
        seq_len=64,
        n_train_paths=256,
        n_eval_paths=64,
        batch_size=16,
        z_dim=32,
        learning_rate=3e-4,
        g2c_ratio=3,
        n_steps=20,
        epsilon=1.0,
        sinkhorn_iters=5,
        loss="energy_distance",
        ema_decay=0.9,
        critic_sign=True,
        eval_every=10,
        n_samples=32,
        eval_dir="finance/eval_smoke",
        ckpt_dir="finance/ckpt_smoke",
        log_dir="finance/logs_smoke",
        seed=0,
        device="cpu",
        num_workers=0,
    )
    base.update(kw)
    return FinanceConfig(**base)


def test_fit_end_to_end(tmp_path, monkeypatch):
    """fit() on smoke values: finite history (incl. the divergence metric),
    history.csv on disk, checkpoint round-trip, de-standardized sampling."""
    monkeypatch.setenv("OT_GAN_ROOT", str(tmp_path))
    cfg = _smoke_cfg()
    tr = ReturnsTrainer(cfg)
    history = tr.fit()

    assert len(history) == cfg.n_steps // cfg.eval_every  # logged at steps 10 and 20
    for row in history:
        assert {"step", "energy_distance", "cross", "real_real", "fake_fake"} <= set(row)
        assert "sinkhorn_divergence_metric" in row
        assert all(math.isfinite(v) for v in row.values())
    assert (tmp_path / "finance" / "eval_smoke" / "history.csv").exists()

    # Checkpoint round-trip restores generator weights + scale + step.
    ckpt = tmp_path / "finance" / "ckpt_smoke" / "returns_gan.pt"
    assert ckpt.exists()
    tr2 = ReturnsTrainer(cfg)
    assert tr2.step == 0
    tr2.load_checkpoint(ckpt)
    for p1, p2 in zip(tr.generator.parameters(), tr2.generator.parameters(), strict=True):
        assert torch.equal(p1, p2)
    assert tr2.scale == tr.scale
    assert tr2.step == cfg.n_steps

    # sample(n): (n, 1, seq_len), de-standardized = EMA output * scale.
    z = torch.randn(8, cfg.z_dim)
    monkeypatch.setattr(tr, "sample_z", lambda n: z[:n])
    out = tr.sample(8)
    assert out.shape == (8, 1, cfg.seq_len)
    with torch.no_grad():
        manual = tr.ema.model(z) * tr.scale
    assert torch.allclose(out, manual)
    assert tr.scale != 1.0  # GJR returns have std ~1e-2; standardization is real


def test_critic_sign_false_runs_one_step(tmp_path, monkeypatch):
    """critic_sign=False (the original bug) still optimizes without blowing up."""
    monkeypatch.setenv("OT_GAN_ROOT", str(tmp_path))
    tr = ReturnsTrainer(_smoke_cfg(critic_sign=False))
    B = tr.cfg.batch_size
    real_1, real_2 = tr.train_paths[:B], tr.train_paths[B : 2 * B]
    terms = tr._optimize(real_1, real_2, critic_step=True)
    assert torch.isfinite(terms.total)


def test_divergence_metric_epsilon_is_dimensionless():
    """THE finance-track convention: C = C / C.mean() before sinkhorn, so the
    raw-path divergence is invariant under a common rescaling of both inputs
    (epsilon=1.0 means the same thing at any path scale)."""
    x = torch.randn(32, 64) * 0.01  # return-sized paths
    y = torch.randn(32, 64) * 0.012 + 0.001
    d = float(_raw_path_divergence(x, y, epsilon=1.0, iters=20))
    d_scaled = float(_raw_path_divergence(100.0 * x, 100.0 * y, epsilon=1.0, iters=20))
    assert math.isfinite(d)
    assert math.isclose(d, d_scaled, rel_tol=1e-4, abs_tol=1e-6)


# ---- direction tests: the 1D replay of the headline critic-sign ablation ----


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


def _direction_setup(tmp_path, monkeypatch, critic_sign):
    monkeypatch.setenv("OT_GAN_ROOT", str(tmp_path))
    cfg = _smoke_cfg(critic_sign=critic_sign, sinkhorn_iters=20, n_train_paths=64)
    tr = ReturnsTrainer(cfg)
    B = cfg.batch_size
    z1, z2 = torch.randn(B, cfg.z_dim), torch.randn(B, cfg.z_dim)
    _fixed_z_patch(tr, z1, z2)
    r1, r2 = tr.train_paths[:B], tr.train_paths[B : 2 * B]
    return tr, r1, r2, z1, z2


def test_critic_ascends_energy_distance(tmp_path, monkeypatch):
    """With critic_sign=True, repeated critic steps INCREASE D^2 on held batches."""
    tr, r1, r2, z1, z2 = _direction_setup(tmp_path, monkeypatch, critic_sign=True)
    before = _measure(tr, r1, r2, z1, z2)
    for _ in range(20):
        tr._optimize(r1, r2, critic_step=True)
    after = _measure(tr, r1, r2, z1, z2)
    assert after > before, f"critic should ascend: {before:.4f} -> {after:.4f}"


def test_critic_sign_false_descends_energy_distance(tmp_path, monkeypatch):
    """critic_sign=False is the original bug: the critic DESCENDS D^2."""
    tr, r1, r2, z1, z2 = _direction_setup(tmp_path, monkeypatch, critic_sign=False)
    before = _measure(tr, r1, r2, z1, z2)
    for _ in range(20):
        tr._optimize(r1, r2, critic_step=True)
    after = _measure(tr, r1, r2, z1, z2)
    assert after < before, f"buggy critic should descend: {before:.4f} -> {after:.4f}"


def test_csv_target_windows_and_holds_out(tmp_path, monkeypatch):
    """target='csv' loads a comma-separated series, windows it into seq_len
    chunks and holds out the tail for evaluation."""
    monkeypatch.setenv("OT_GAN_ROOT", str(tmp_path))
    series = (torch.randn(16 * 16) * 0.01).numpy()
    csv_path = tmp_path / "returns.csv"
    csv_path.write_text(",".join(f"{v:.8f}" for v in series))
    cfg = _smoke_cfg(
        target="csv",
        csv_path=str(csv_path),
        seq_len=16,
        n_train_paths=12,
        n_eval_paths=4,
        batch_size=4,
        z_dim=16,
    )
    tr = ReturnsTrainer(cfg)
    assert tr.train_paths.shape == (12, 1, 16)
    assert tr.eval_paths.shape == (4, 1, 16)
