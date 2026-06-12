"""Shared pytest fixtures. Tests run on CPU by default for determinism and speed."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _seed():
    """Seed every test for reproducibility."""
    torch.manual_seed(0)
    yield


@pytest.fixture
def cpu():
    return torch.device("cpu")


@pytest.fixture
def tiny_config():
    """A minimal in-memory Config for fast tests (no disk/data dependency)."""
    from otgan.config import Config

    return Config(
        dataset="mnist",
        channels=1,
        image_size=32,
        batch_size=8,
        z_dim=100,
        n_epochs=1,
        epsilon=1.0,
        sinkhorn_iters=5,
        n_samples=4,
        n_eval=16,
        fid_every=0,
        max_batches=2,
        num_workers=0,
        device="cpu",
        seed=0,
    )
