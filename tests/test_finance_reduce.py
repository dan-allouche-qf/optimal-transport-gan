"""Tests for otgan.finance.reduce: scenario reduction by entropic OT.

Fast, CPU-only, deterministic: every routine takes an explicit seed and uses a
local generator, so thresholds here are stable margins, not flaky statistics.
The scale-invariance test locks the repo's cost-normalization convention
(``C = C / C.mean()`` before sinkhorn): it uses ``epsilon=0.1``, where the
plan has real structure — at ``epsilon=1.0`` entropic smoothing makes the
plan near-uniform and a correlation between near-constant tensors would only
measure float noise.
"""

import pytest
import torch

from otgan.finance.reduce import (
    holdout_split,
    kmeans_reduce,
    pairwise_sqeuclidean,
    random_subsample,
    reduction_report,
    sinkhorn_reduce,
)
from otgan.finance.simulate import gjr_garch_paths

GJR_KW = {"omega": 5e-6, "alpha": 0.05, "beta": 0.90, "gamma": 0.05}

REDUCERS = {
    "kmeans": lambda X, K: kmeans_reduce(X, K, seed=0),
    "random": lambda X, K: random_subsample(X, K, seed=0),
    "sinkhorn": lambda X, K: sinkhorn_reduce(X, K, seed=0),
}


@pytest.fixture(scope="module")
def paths():
    return gjr_garch_paths(256, 32, seed=11, **GJR_KW)


# ---- cost matrix ----------------------------------------------------------


def test_pairwise_sqeuclidean_matches_cdist_and_flattens():
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(7, 4, 3, generator=gen)
    y = torch.randn(5, 4, 3, generator=gen)
    got = pairwise_sqeuclidean(x, y)
    want = torch.cdist(x.reshape(7, -1), y.reshape(5, -1)) ** 2
    assert got.shape == (7, 5)
    assert torch.allclose(got, want, atol=1e-5)
    assert (got >= 0).all()
    self_cost = pairwise_sqeuclidean(x, x)
    assert torch.allclose(self_cost.diagonal(), torch.zeros(7), atol=1e-5)


# ---- common ReducedSet invariants -----------------------------------------


@pytest.mark.parametrize("name", sorted(REDUCERS))
def test_weights_nonnegative_and_sum_to_one(paths, name):
    red = REDUCERS[name](paths, 16)
    assert red.paths.shape == (16, paths.shape[1])
    assert (red.weights >= 0).all()
    assert torch.isclose(red.weights.sum(), torch.tensor(1.0), atol=1e-5)
    assert red.distortion >= 0.0
    assert torch.isfinite(torch.tensor(red.distortion))


@pytest.mark.parametrize("name", sorted(REDUCERS))
def test_k_out_of_range_raises(paths, name):
    with pytest.raises(ValueError, match="K must be"):
        REDUCERS[name](paths, paths.shape[0] + 1)


def test_sinkhorn_plan_has_correct_marginals(paths):
    n = paths.shape[0]
    red = sinkhorn_reduce(paths, 16, seed=0)
    assert red.plan is not None
    assert red.plan.shape == (n, 16)
    assert torch.allclose(red.plan.sum(dim=1), torch.full((n,), 1.0 / n), atol=1e-6)
    assert torch.allclose(red.plan.sum(dim=0), red.weights, atol=1e-6)


# ---- distortion ------------------------------------------------------------


def test_sinkhorn_reduce_beats_random_subsample(paths):
    sk = sinkhorn_reduce(paths, 16, seed=0)
    rs = random_subsample(paths, 16, seed=0)
    assert sk.distortion <= rs.distortion


def test_k_equals_n_gives_near_zero_distortion(paths):
    X = paths[:64]
    assert kmeans_reduce(X, 64, seed=1).distortion < 1e-6
    assert random_subsample(X, 64, seed=1).distortion < 1e-6
    # eps -> 0 limit: the plan hardens to the identity matching, centroids
    # collapse onto the scenarios themselves (classical reduction recovered).
    sk = sinkhorn_reduce(X, 64, epsilon=0.01, sinkhorn_iters=300, seed=1)
    assert sk.distortion < 1e-6


# ---- determinism -----------------------------------------------------------


def test_sinkhorn_reduce_deterministic_given_seed(paths):
    first = sinkhorn_reduce(paths, 8, seed=3)
    torch.manual_seed(999)  # perturb the GLOBAL rng: local generators must not care
    torch.randn(100)
    second = sinkhorn_reduce(paths, 8, seed=3)
    assert torch.equal(first.paths, second.paths)
    assert torch.equal(first.weights, second.weights)
    assert first.plan is not None and second.plan is not None
    assert torch.equal(first.plan, second.plan)
    assert first.distortion == second.distortion


def test_kmeans_and_subsample_deterministic_given_seed(paths):
    km1, km2 = (kmeans_reduce(paths, 8, seed=5) for _ in range(2))
    assert torch.equal(km1.paths, km2.paths)
    rs1, rs2 = (random_subsample(paths, 8, seed=5) for _ in range(2))
    assert torch.equal(rs1.paths, rs2.paths)


def test_different_seeds_differ(paths):
    a = random_subsample(paths, 8, seed=0)
    b = random_subsample(paths, 8, seed=1)
    assert not torch.equal(a.paths, b.paths)


# ---- cost-normalization convention (locked) --------------------------------


def test_plan_invariant_to_global_scale_of_paths(paths):
    """C / C.mean() makes epsilon dimensionless: scaling every path by 10
    multiplies raw costs by 100 but leaves the normalized cost — hence the
    plan and weights — unchanged. The distortion, reported in data units,
    must scale by exactly the squared factor."""
    base = sinkhorn_reduce(paths, 16, epsilon=0.1, seed=0)
    scaled = sinkhorn_reduce(10.0 * paths, 16, epsilon=0.1, seed=0)
    assert base.plan is not None and scaled.plan is not None
    stacked = torch.stack([base.plan.reshape(-1), scaled.plan.reshape(-1)])
    assert torch.corrcoef(stacked)[0, 1] > 0.99
    assert torch.allclose(base.weights, scaled.weights, atol=1e-4)
    assert scaled.distortion / base.distortion == pytest.approx(100.0, rel=1e-3)


# ---- reduction report -------------------------------------------------------


def test_reduction_report_keys_and_finite_cvar(paths):
    fit, held_out = holdout_split(paths)
    assert fit.shape[0] + held_out.shape[0] == paths.shape[0]
    red = sinkhorn_reduce(fit, 16, epsilon=0.1, seed=0)
    report = reduction_report(held_out, red, alphas=(0.95, 0.99))
    expected = {
        "mean_full",
        "mean_reduced",
        "std_full",
        "std_reduced",
        "q0.95_full",
        "q0.95_reduced",
        "q0.99_full",
        "q0.99_reduced",
        "cvar0.95_full",
        "cvar0.95_reduced",
        "cvar0.99_full",
        "cvar0.99_reduced",
    }
    assert set(report) == expected
    assert all(torch.isfinite(torch.tensor(v)) for v in report.values())
    # CVaR is monotone in alpha and dominates the corresponding VaR.
    assert report["cvar0.99_full"] >= report["cvar0.95_full"]
    assert report["cvar0.99_reduced"] >= report["cvar0.95_reduced"]
    assert report["std_full"] > 0
