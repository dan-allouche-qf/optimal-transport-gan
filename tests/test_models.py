"""Tests for the generator and critic."""

import pytest
import torch

from otgan.models import OTGANCritic, OTGANGenerator, build_models


@pytest.mark.parametrize("channels", [1, 3])
def test_generator_shape_and_range(channels):
    g = OTGANGenerator(z_dim=100, channels=channels)
    out = g(torch.randn(4, 100))
    assert out.shape == (4, channels, 32, 32)
    assert out.min() >= -1.0 and out.max() <= 1.0  # Tanh output


@pytest.mark.parametrize("channels", [1, 3])
def test_critic_embedding_is_l2_normalized(channels):
    c = OTGANCritic(channels=channels)
    emb = c(torch.rand(4, channels, 32, 32) * 2 - 1)
    assert emb.shape == (4, 32768)
    norms = emb.norm(p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_generator_critic_compose_and_backprop():
    g, c = build_models(_TinyCfg())
    z = torch.randn(3, 100)
    emb = c(g(z))
    loss = emb.sum()
    loss.backward()
    assert next(g.parameters()).grad is not None


class _TinyCfg:
    z_dim = 100
    channels = 1
