"""Tests for the log-domain Sinkhorn solver and cosine cost."""

import torch

from otgan.sinkhorn import cost, sinkhorn


def _uniform(n):
    return torch.full((n,), 1.0 / n)


def test_plan_marginals():
    """Plan rows sum to a, columns sum to b."""
    torch.manual_seed(0)
    n = 8
    C = torch.rand(n, n)
    a, b = _uniform(n), _uniform(n)
    M = sinkhorn(a, b, C, epsilon=0.1, iters=200)
    assert torch.allclose(M.sum(1), a, atol=1e-4)
    assert torch.allclose(M.sum(0), b, atol=1e-4)


def test_plan_nonnegative():
    C = torch.rand(6, 6)
    M = sinkhorn(_uniform(6), _uniform(6), C, epsilon=0.5, iters=100)
    assert (M >= 0).all()


def test_no_overflow_small_epsilon():
    """The raw-exp solver overflows here; the log-domain one must not."""
    C = torch.rand(8, 8) * 2.0  # cosine distances live in [0, 2]
    M = sinkhorn(_uniform(8), _uniform(8), C, epsilon=0.01, iters=100)
    assert torch.isfinite(M).all()


def test_matches_raw_domain_at_eps1():
    """At epsilon=1 the raw-exp reference is safe; log-domain must agree."""
    torch.manual_seed(1)
    n = 8
    C = torch.rand(n, n)
    a, b = _uniform(n), _uniform(n)

    # Reference: original notebook's raw-domain iteration.
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    K = torch.exp(-C / 1.0)
    for _ in range(300):
        u = a / (K @ v + 1e-8)
        v = b / (K.t() @ u + 1e-8)
    M_raw = torch.diag(u) @ K @ torch.diag(v)

    M_log = sinkhorn(a, b, C, epsilon=1.0, iters=300)
    assert torch.allclose(M_log, M_raw, atol=1e-5)


def test_sharpening_concentrates_on_diagonal():
    """With a distance that is minimal on the diagonal, small epsilon -> near-identity plan."""
    n = 6
    C = 1.0 - torch.eye(n)  # 0 on the diagonal, 1 off-diagonal
    M = sinkhorn(_uniform(n), _uniform(n), C, epsilon=0.01, iters=300)
    assert M.diag().sum() > 0.95 * M.sum()


def test_plan_detached_gradient_flows_through_cost():
    """Plan carries no gradient; the OT cost is differentiable w.r.t. C."""
    torch.manual_seed(2)
    emb1 = torch.randn(5, 16, requires_grad=True)
    emb2 = torch.randn(5, 16)
    C = cost(torch.nn.functional.normalize(emb1, dim=1), torch.nn.functional.normalize(emb2, dim=1))
    M = sinkhorn(_uniform(5), _uniform(5), C, epsilon=0.5, iters=50)
    assert M.requires_grad is False
    ot_cost = (M * C).sum()
    assert ot_cost.requires_grad is True
    ot_cost.backward()
    assert emb1.grad is not None and torch.isfinite(emb1.grad).all()
