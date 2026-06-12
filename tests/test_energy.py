"""Tests for the minibatch energy distance assembly."""

import torch
import torch.nn.functional as F

from otgan.energy import energy_distance


def _emb(n=8, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(n, d, generator=g), dim=1)


def test_identical_distributions_near_zero():
    """If reals and fakes are the same embeddings, D^2 ~ 0."""
    e = _emb()
    terms = energy_distance((e, e, e, e), epsilon=1.0, iters=200)
    assert abs(float(terms.total)) < 1e-3


def test_assembly_identity():
    """total == cross - 2*real_real - 2*fake_fake exactly (single source of truth)."""
    embs = (_emb(seed=1), _emb(seed=2), _emb(seed=3), _emb(seed=4))
    t = energy_distance(embs, epsilon=1.0, iters=50)
    expected = t.cross - 2.0 * t.real_real - 2.0 * t.fake_fake
    assert torch.allclose(t.total, expected, atol=1e-6)


def test_differentiable_wrt_embeddings():
    """D^2 is finite and differentiable w.r.t. the (fake) critic embeddings."""
    cr1, cr2 = _emb(seed=1), _emb(seed=2)
    cf1 = _emb(seed=3).clone().requires_grad_(True)
    cf2 = _emb(seed=4).clone().requires_grad_(True)
    t = energy_distance((cr1, cr2, cf1, cf2), epsilon=1.0, iters=50)
    assert torch.isfinite(t.total)
    t.total.backward()
    assert cf1.grad is not None and torch.isfinite(cf1.grad).all()


def _cluster(direction, n=8, d=16, seed=0):
    """n embeddings tightly clustered around a unit `direction`."""
    g = torch.Generator().manual_seed(seed)
    base = torch.zeros(d)
    base[direction] = 1.0
    return F.normalize(base + 0.01 * torch.randn(n, d, generator=g), dim=1)


def test_separated_distributions_positive():
    """Reals and fakes clustered around orthogonal directions give a clearly positive D^2."""
    real1, real2 = _cluster(0, seed=1), _cluster(0, seed=2)  # both near e_0
    fake1, fake2 = _cluster(1, seed=3), _cluster(1, seed=4)  # both near e_1 (orthogonal)
    t = energy_distance((real1, real2, fake1, fake2), epsilon=1.0, iters=200)
    assert float(t.total) > 0.5
