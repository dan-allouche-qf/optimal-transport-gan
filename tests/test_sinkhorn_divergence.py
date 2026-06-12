"""Tests for the debiased Sinkhorn divergence and POT cross-validation of sinkhorn().

The POT tests validate our hand-rolled log-domain solver elementwise against
``ot.sinkhorn``, the reference implementation — skipped if POT is not installed
(it ships in the ``dev`` extra).
"""

import pytest
import torch
import torch.nn.functional as F

from otgan.energy import compute_loss, energy_distance, sinkhorn_divergence
from otgan.sinkhorn import cost, sinkhorn


def _unit_embeddings(n: int, d: int, shift: float = 0.0, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g) + shift
    return F.normalize(x, dim=1, p=2)


def _four(shift_fake: float, seed: int = 0):
    return (
        _unit_embeddings(16, 32, 0.0, seed),
        _unit_embeddings(16, 32, 0.0, seed + 1),
        _unit_embeddings(16, 32, shift_fake, seed + 2),
        _unit_embeddings(16, 32, shift_fake, seed + 3),
    )


def test_divergence_exactly_zero_when_fake_equals_real():
    """S(a, a) = 0 for identical empirical measures: with all four embeddings
    equal to the same batch, cross terms cancel the self-terms exactly (the
    debiasing property of Feydy et al. 2019)."""
    x = _unit_embeddings(16, 32, 0.0, seed=0)
    terms = sinkhorn_divergence((x, x, x, x), epsilon=1.0, iters=100)
    assert abs(float(terms.total)) < 1e-5


def test_divergence_positive_and_orders_distributions():
    """Positive on finite samples, and much larger for separated distributions
    than for two independent samples of the same distribution."""
    near = sinkhorn_divergence(_four(shift_fake=0.0), epsilon=1.0, iters=100)
    far = sinkhorn_divergence(_four(shift_fake=3.0), epsilon=1.0, iters=100)
    assert float(near.total) > 0.0
    assert float(far.total) > 2.0 * float(near.total)


def test_self_terms_use_same_batch_not_independent_batches():
    """The implementation difference vs the energy distance: at small epsilon
    the same-batch self-cost W(X, X) is ~0 (plan ~ identity), while the
    independent-minibatch self-cost W(X, X') stays at finite-sample scale."""
    emb = _four(shift_fake=0.0)
    sd = sinkhorn_divergence(emb, epsilon=0.1, iters=200)
    ed = energy_distance(emb, epsilon=0.1, iters=200)
    assert float(sd.real_real) < 0.1 * float(ed.real_real)


def test_divergence_gradient_flows_through_embeddings():
    emb = tuple(e.clone().requires_grad_(True) for e in _four(shift_fake=1.0))
    terms = sinkhorn_divergence(emb, epsilon=1.0, iters=20)
    terms.total.backward()
    assert all(e.grad is not None and torch.isfinite(e.grad).all() for e in emb)


def test_compute_loss_dispatch():
    emb = _four(shift_fake=1.0)
    ed = compute_loss(emb, 1.0, 20, "energy_distance")
    sd = compute_loss(emb, 1.0, 20, "sinkhorn_divergence")
    assert torch.equal(ed.total, energy_distance(emb, 1.0, 20).total)
    assert torch.equal(sd.total, sinkhorn_divergence(emb, 1.0, 20).total)
    with pytest.raises(ValueError, match="loss must be one of"):
        compute_loss(emb, 1.0, 20, "wasserstein_gp")


# ---- POT cross-validation -------------------------------------------------


@pytest.mark.parametrize("epsilon", [0.1, 1.0])
def test_sinkhorn_plan_matches_pot(epsilon):
    """Elementwise agreement of our log-domain plan with POT's ot.sinkhorn."""
    ot = pytest.importorskip("ot")
    g = torch.Generator().manual_seed(0)
    e1 = F.normalize(torch.randn(8, 16, generator=g), dim=1)
    e2 = F.normalize(torch.randn(8, 16, generator=g), dim=1)
    C = cost(e1, e2)
    a = torch.full((8,), 1.0 / 8, dtype=torch.float64)
    b = torch.full((8,), 1.0 / 8, dtype=torch.float64)
    C64 = C.to(torch.float64)

    ours = sinkhorn(a, b, C64, epsilon, iters=2000)
    theirs = torch.as_tensor(
        ot.sinkhorn(a.numpy(), b.numpy(), C64.numpy(), reg=epsilon, numItermax=20000)
    )
    assert torch.allclose(ours, theirs, atol=1e-5)
    # And the transported costs <M, C> agree too.
    assert abs(float((ours * C64).sum()) - float((theirs * C64).sum())) < 1e-6
