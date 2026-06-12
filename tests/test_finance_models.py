"""Behavioral tests for the 1D returns models: unbounded generator head,
unit-norm critic embeddings, gradient flow through the OT objective, and the
dilated receptive field."""

import pytest
import torch

from otgan.energy import energy_distance
from otgan.finance.config import FinanceConfig
from otgan.finance.models1d import ReturnsCritic, ReturnsGenerator, build_finance_models

Z_DIM, SEQ_LEN = 32, 64


def test_generator_output_shape_and_finite():
    gen = ReturnsGenerator(z_dim=Z_DIM, seq_len=SEQ_LEN)
    out = gen(torch.randn(4, Z_DIM))
    assert out.shape == (4, 1, SEQ_LEN)
    assert torch.isfinite(out).all()


def test_generator_head_is_unbounded():
    """The output head is LINEAR: scaling its weights by 10 scales the output by
    exactly 10 and pushes |values| past 1 — impossible with a tanh head."""
    gen = ReturnsGenerator(z_dim=Z_DIM, seq_len=SEQ_LEN)
    z = torch.randn(8, Z_DIM)
    with torch.no_grad():
        base = gen(z)
        gen.head.weight.mul_(10.0)
        assert gen.head.bias is not None
        gen.head.bias.mul_(10.0)
        scaled = gen(z)
    assert torch.allclose(scaled, 10.0 * base, atol=1e-5)  # affine head, no saturation
    assert scaled.abs().max() > 1.0


def test_critic_embedding_rows_unit_norm():
    critic = ReturnsCritic(seq_len=SEQ_LEN)
    emb = critic(torch.randn(8, 1, SEQ_LEN))
    assert emb.shape == (8, critic.embed_dim)
    norms = emb.norm(dim=1)
    assert torch.allclose(norms, torch.ones(8), atol=1e-5)


def test_gradient_flows_generator_to_energy_distance():
    """One backward through critic(generator(z)) populates generator gradients."""
    gen = ReturnsGenerator(z_dim=Z_DIM, seq_len=SEQ_LEN)
    critic = ReturnsCritic(seq_len=SEQ_LEN)
    real_1, real_2 = torch.randn(8, 1, SEQ_LEN), torch.randn(8, 1, SEQ_LEN)
    fake_1, fake_2 = gen(torch.randn(8, Z_DIM)), gen(torch.randn(8, Z_DIM))
    emb = (critic(real_1), critic(real_2), critic(fake_1), critic(fake_2))
    terms = energy_distance(emb, epsilon=1.0, iters=5)
    terms.total.backward()
    grads = [p.grad for p in gen.parameters()]
    assert all(g is not None for g in grads)
    assert sum(g.abs().sum().item() for g in grads if g is not None) > 0.0


def test_receptive_field_covers_first_lag():
    """Perturbing x[..., 0] changes the embedding: the dilated stack is wired
    so even the most distant lag reaches the (stride-reduced) feature map."""
    critic = ReturnsCritic(seq_len=SEQ_LEN)
    x = torch.randn(1, 1, SEQ_LEN)
    perturbed = x.clone()
    perturbed[..., 0] += 1.0
    with torch.no_grad():
        delta = (critic(x) - critic(perturbed)).abs().max()
    assert delta > 1e-6


def test_build_finance_models_from_config():
    cfg = FinanceConfig(z_dim=16, seq_len=32)
    gen, critic = build_finance_models(cfg)
    out = gen(torch.randn(2, 16))
    assert out.shape == (2, 1, 32)
    assert critic(out).shape == (2, 256 * (32 // 4))


@pytest.mark.parametrize("cls", [ReturnsGenerator, ReturnsCritic])
def test_seq_len_must_be_divisible_by_four(cls):
    with pytest.raises(ValueError, match="divisible by 4"):
        cls(seq_len=30)
