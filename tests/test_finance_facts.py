"""Tests for otgan.finance.evaluate: stylized facts and the Sinkhorn metric.

Deterministic by construction (seeded simulators, seeded local generators),
so the statistical thresholds are stable margins. The centerpiece is the
GARCH-vs-GJR trap: a symmetric GARCH and a GJR-GARCH with nearly identical
volatility clustering (ACF of r^2) are separated ONLY by the leverage
correlation — any evaluation table without that row would call them equal.
"""

import pytest
import torch

from otgan.finance.evaluate import (
    acf,
    excess_kurtosis,
    leverage_corr,
    sinkhorn_divergence_metric,
    stylized_facts_table,
    to_markdown,
)
from otgan.finance.simulate import gbm_paths, gjr_garch_paths

GBM_KW = {"mu": 0.05, "sigma": 0.2, "dt": 1 / 252}
GJR_KW = {"omega": 5e-6, "alpha": 0.05, "beta": 0.90, "gamma": 0.05}
# Same persistence-style symmetric GARCH: alpha absorbs gamma/2 so the
# unconditional variance matches GJR_KW exactly (0.075 + 0.90 = 0.975).
SYM_KW = {"omega": 5e-6, "alpha": 0.075, "beta": 0.90, "gamma": 0.0}


def _white_noise(n: int, t: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n, t, generator=gen)


@pytest.fixture(scope="module")
def gjr():
    return gjr_garch_paths(2048, 128, seed=7, **GJR_KW)


# ---- acf -------------------------------------------------------------------


def test_acf_lag0_is_one_and_white_noise_has_no_vol_clustering():
    wn = _white_noise(512, 256, seed=0)
    a = acf(wn, 10)
    assert a.shape == (11,)
    assert a[0].item() == 1.0
    assert acf(wn.abs(), 10)[1:].abs().max().item() < 0.02
    assert acf(wn, 10)[1:].abs().max().item() < 0.02


def test_acf_rejects_bad_max_lag():
    wn = _white_noise(4, 16, seed=0)
    with pytest.raises(ValueError, match="max_lag"):
        acf(wn, 16)
    with pytest.raises(ValueError, match="max_lag"):
        acf(wn, -1)


def test_gjr_has_volatility_clustering(gjr):
    assert acf(gjr, 1)[1].abs().item() < 0.02  # raw returns: no linear memory
    assert acf(gjr**2, 5)[1].item() > 0.03  # squared returns: clustered
    assert acf(gjr.abs(), 5)[1].item() > 0.03


# ---- excess kurtosis ---------------------------------------------------------


def test_excess_kurtosis_gaussian_near_zero_gjr_fat(gjr):
    assert abs(excess_kurtosis(_white_noise(512, 256, seed=1))) < 0.1
    assert excess_kurtosis(gjr) > 0.5


# ---- leverage correlation: the GARCH-vs-GJR trap ----------------------------


def test_leverage_separates_gjr_from_symmetric_garch(gjr):
    sym = gjr_garch_paths(2048, 128, seed=7, **SYM_KW)
    # Both cluster volatility about equally: ACF(r^2) cannot tell them apart.
    assert acf(sym**2, 1)[1].item() > 0.03
    assert acf(gjr**2, 1)[1].item() > 0.03
    # Only the leverage correlation separates them: gamma > 0 means bad news
    # raises future variance more, so corr(r_t, r_{t+1}^2) < 0.
    assert leverage_corr(gjr, 5)[0].item() < -0.01
    assert abs(leverage_corr(sym, 5)[0].item()) < 0.05


def test_leverage_corr_rejects_bad_max_lag():
    wn = _white_noise(4, 16, seed=0)
    with pytest.raises(ValueError, match="max_lag"):
        leverage_corr(wn, 16)
    with pytest.raises(ValueError, match="max_lag"):
        leverage_corr(wn, 0)


# ---- sinkhorn divergence metric ---------------------------------------------


def test_sinkhorn_divergence_small_same_process_large_across():
    gbm_a = gbm_paths(1024, 16, seed=1, **GBM_KW)
    gbm_b = gbm_paths(1024, 16, seed=2, **GBM_KW)
    gjr_c = gjr_garch_paths(1024, 16, seed=3, **GJR_KW)

    # eps=1.0 converges in well under 50 iterations; 50 keeps the test fast.
    d_self = sinkhorn_divergence_metric(gbm_a, gbm_a, iters=50)
    d_same = sinkhorn_divergence_metric(gbm_a, gbm_b, iters=50)
    d_cross = sinkhorn_divergence_metric(gbm_a, gjr_c, iters=50)

    assert abs(d_self) < 1e-6  # identical samples: debiasing cancels exactly
    assert 0.0 < d_same < 0.01  # same process: small finite-sample residual
    assert d_cross > 3.0 * d_same  # different process: clearly separated


def test_sinkhorn_divergence_deterministic_and_seeded_subsampling():
    big = gjr_garch_paths(1100, 16, seed=4, **GJR_KW)  # > 1024: subsampling kicks in
    small = gbm_paths(256, 16, seed=5, **GBM_KW)
    first = sinkhorn_divergence_metric(big, small, iters=20, seed=0)
    torch.manual_seed(999)  # perturb the GLOBAL rng: local generators must not care
    torch.randn(100)
    second = sinkhorn_divergence_metric(big, small, iters=20, seed=0)
    assert first == second
    assert sinkhorn_divergence_metric(big, small, iters=20, seed=1) != first


# ---- table + markdown ---------------------------------------------------------


def test_stylized_facts_table_rows_and_markdown_rendering():
    real = gjr_garch_paths(256, 64, seed=5, **GJR_KW)
    fake = gbm_paths(256, 64, seed=6, **GBM_KW)
    table = stylized_facts_table(real, fake, lags=(1, 5))

    expected_rows = [
        "excess kurtosis",
        "ACF(r) lag-1",
        "ACF(|r|) lag-1",
        "ACF(|r|) lag-5",
        "ACF(r^2) lag-1",
        "ACF(r^2) lag-5",
        "leverage corr lag-1",
        "leverage corr lag-5",
    ]
    assert list(table) == expected_rows
    for cols in table.values():
        assert set(cols) == {"target", "generated"}
        assert all(torch.isfinite(torch.tensor(v)) for v in cols.values())

    md = to_markdown(table)
    lines = md.splitlines()
    assert lines[0] == "| statistic | target | generated |"
    assert len(lines) == 2 + len(expected_rows)
    for row in expected_rows:
        assert f"| {row} |" in md
