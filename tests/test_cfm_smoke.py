"""Fast, offline smoke tests for OT conditional flow matching (otgan/cfm.py):
the SmallUNet vector field, the minibatch OT coupling, and the CFMTrainer
end-to-end on a synthetic loader."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from otgan.cfm import CFMTrainer, SmallUNet, minibatch_coupling
from otgan.config import Config
from otgan.trainer import build_trainer


def _cfm_cfg(**kw):
    base = dict(
        model="cfm",
        dataset="mnist",
        channels=1,
        batch_size=8,
        n_epochs=1,
        max_batches=3,
        fid_every=0,
        ode_steps=5,
        sinkhorn_iters=5,
        n_samples=4,
        num_workers=0,
        device="cpu",
        seed=0,
    )
    base.update(kw)
    return Config(**base)


def _fake_loader(cfg, n_batches=3):
    n = n_batches * 2 * cfg.batch_size
    imgs = torch.rand(n, cfg.channels, 32, 32) * 2 - 1
    labels = torch.zeros(n, dtype=torch.long)
    return DataLoader(TensorDataset(imgs, labels), batch_size=2 * cfg.batch_size, drop_last=True)


# ---------------------------------------------------------------------------
# SmallUNet
# ---------------------------------------------------------------------------
def test_smallunet_forward_shape():
    net = SmallUNet(channels=1)
    x = torch.randn(4, 1, 32, 32)
    t = torch.rand(4)
    out = net(x, t)
    assert out.shape == (4, 1, 32, 32)
    assert torch.isfinite(out).all()


def test_smallunet_param_count():
    n_params = sum(p.numel() for p in SmallUNet(channels=1).parameters())
    assert 500_000 < n_params < 3_000_000, f"unexpected size: {n_params}"


# ---------------------------------------------------------------------------
# Minibatch coupling
# ---------------------------------------------------------------------------
def _two_clusters(n_per=16, d=8, gap=10.0):
    """Two far-separated Gaussian blobs around -gap and +gap."""
    lo = -gap + 0.1 * torch.randn(n_per, d)
    hi = gap + 0.1 * torch.randn(n_per, d)
    return torch.cat([lo, hi])


def _same_cluster_rate(x0, paired):
    """Fraction of rows whose partner came from the same (sign-coded) cluster."""
    return (torch.sign(x0.mean(1)) == torch.sign(paired.mean(1))).float().mean().item()


def test_coupling_none_is_identity():
    x0, x1 = torch.randn(8, 5), torch.randn(8, 5)
    assert torch.equal(minibatch_coupling(x0, x1, mode="none"), x1)


def test_coupling_sinkhorn_returns_rows_of_x1():
    x0, x1 = torch.randn(8, 1, 4, 4), torch.randn(8, 1, 4, 4)
    out = minibatch_coupling(x0, x1, mode="sinkhorn", eps=0.5, iters=20)
    assert out.shape == x1.shape
    eq = (out.flatten(1).unsqueeze(1) == x1.flatten(1).unsqueeze(0)).all(dim=-1)  # (8, 8)
    assert eq.any(dim=1).all(), "every output row must be some row of x1"


def test_coupling_sinkhorn_pairs_clusters_near_optimally():
    x0, x1 = _two_clusters(), _two_clusters()
    out = minibatch_coupling(x0, x1, mode="sinkhorn", eps=0.05, iters=50)
    assert _same_cluster_rate(x0, out) > 0.8


def test_coupling_exact_pairs_clusters():
    pytest.importorskip("ot")
    x0, x1 = _two_clusters(), _two_clusters()
    out = minibatch_coupling(x0, x1, mode="exact")
    eq = (out.unsqueeze(1) == x1.unsqueeze(0)).all(dim=-1)
    assert eq.any(dim=1).all()
    assert _same_cluster_rate(x0, out) > 0.8


def test_coupling_rejects_unknown_mode():
    x = torch.randn(4, 3)
    with pytest.raises(ValueError, match="mode"):
        minibatch_coupling(x, x, mode="bogus")


# ---------------------------------------------------------------------------
# CFMTrainer
# ---------------------------------------------------------------------------
def test_cfm_fit_checkpoint_and_sample(tmp_path, monkeypatch):
    monkeypatch.setenv("OT_GAN_ROOT", str(tmp_path))
    cfg = _cfm_cfg(eval_dir="eval", ckpt_dir="ckpt", log_dir="logs")
    tr = CFMTrainer(cfg, dataloader=_fake_loader(cfg))
    history = tr.fit()
    assert len(history) == 1
    assert torch.isfinite(torch.tensor(history[0]["cfm_loss"]))

    ckpt = tmp_path / "ckpt" / "ot_cfm.pt"
    assert ckpt.exists()
    tr2 = CFMTrainer(cfg, dataloader=_fake_loader(cfg))
    tr2.load_checkpoint(ckpt)
    for p1, p2 in zip(tr.model.parameters(), tr2.model.parameters(), strict=True):
        assert torch.equal(p1, p2)
    assert tr2.start_epoch == 1

    imgs = tr2.sample(4)
    assert imgs.shape == (4, 1, 32, 32)
    assert imgs.min() >= -1.0 and imgs.max() <= 1.0


def test_build_trainer_dispatches_cfm():
    assert isinstance(build_trainer(_cfm_cfg()), CFMTrainer)
