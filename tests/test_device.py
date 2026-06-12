"""Tests for device resolution and seeding."""

import torch

from otgan.device import resolve_device, seed_everything


def test_resolve_explicit_cpu():
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_auto_returns_device():
    dev = resolve_device("auto")
    assert isinstance(dev, torch.device)
    assert dev.type in ("cuda", "mps", "cpu")


def test_seed_makes_sampling_deterministic_on_cpu():
    seed_everything(123)
    a = torch.randn(16)
    seed_everything(123)
    b = torch.randn(16)
    assert torch.equal(a, b)
