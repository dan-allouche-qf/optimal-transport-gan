"""Tests for otgan.finance.simulate and FinanceConfig validation.

Fast, CPU-only, offline: simulators are seeded with local generators, so every
assertion here is deterministic and independent of torch's global RNG state.
"""

import pytest
import torch

from otgan.finance.config import FinanceConfig
from otgan.finance.simulate import (
    gbm_paths,
    gjr_garch_paths,
    heston_paths,
    prices_from_returns,
)

GBM_KW = {"mu": 0.05, "sigma": 0.2, "dt": 1 / 252}
HESTON_KW = {"mu": 0.05, "dt": 1 / 252, "kappa": 2.0, "theta": 0.04, "xi": 0.3, "rho": -0.7}
GJR_KW = {"omega": 5e-6, "alpha": 0.05, "beta": 0.90, "gamma": 0.05}

SIMULATORS = {
    "gbm": lambda n, t, seed: gbm_paths(n, t, seed=seed, **GBM_KW),
    "heston": lambda n, t, seed: heston_paths(n, t, seed=seed, **HESTON_KW),
    "gjr_garch": lambda n, t, seed: gjr_garch_paths(n, t, seed=seed, **GJR_KW),
}


def _kurtosis(x: torch.Tensor) -> float:
    x = x.flatten().double()
    c = x - x.mean()
    return ((c**4).mean() / (c**2).mean() ** 2).item()


# ---- shapes / dtype -----------------------------------------------------


@pytest.mark.parametrize("name", sorted(SIMULATORS))
def test_shape_dtype_finite(name):
    r = SIMULATORS[name](7, 16, seed=0)
    assert r.shape == (7, 16)
    assert r.dtype == torch.float32
    assert torch.isfinite(r).all()


# ---- seeding ------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(SIMULATORS))
def test_same_seed_is_deterministic(name):
    first = SIMULATORS[name](8, 12, seed=42)
    torch.manual_seed(999)  # perturb the GLOBAL rng: local generators must not care
    torch.randn(100)
    second = SIMULATORS[name](8, 12, seed=42)
    assert torch.equal(first, second)


@pytest.mark.parametrize("name", sorted(SIMULATORS))
def test_different_seeds_differ(name):
    a = SIMULATORS[name](8, 12, seed=0)
    b = SIMULATORS[name](8, 12, seed=1)
    assert not torch.equal(a, b)


# ---- FinanceConfig validation -------------------------------------------


def test_config_defaults_are_valid():
    cfg = FinanceConfig()
    assert cfg.target == "gjr_garch"
    assert cfg.device == "cpu"


def test_config_rejects_nonstationary_gjr():
    # alpha + beta + gamma/2 = 0.2 + 0.85 + 0.05 = 1.1 >= 1
    with pytest.raises(ValueError, match=r"garch_alpha \+ garch_beta \+ garch_gamma/2"):
        FinanceConfig(garch_alpha=0.2, garch_beta=0.85, garch_gamma=0.1)


def test_config_csv_target_requires_path():
    with pytest.raises(ValueError, match="csv_path"):
        FinanceConfig(target="csv")


def test_simulator_rejects_nonstationary_gjr():
    with pytest.raises(ValueError, match=r"alpha \+ beta \+ gamma/2 < 1"):
        gjr_garch_paths(4, 8, omega=5e-6, alpha=0.2, beta=0.85, gamma=0.1, seed=0)


# ---- GJR-GARCH statistics ------------------------------------------------


@pytest.fixture(scope="module")
def gjr_sample():
    return gjr_garch_paths(4096, 128, seed=7, **GJR_KW)


def test_gjr_unconditional_variance_matches_analytic(gjr_sample):
    # omega / (1 - alpha - beta - gamma/2) = 5e-6 / 0.025 = 2e-4
    analytic = GJR_KW["omega"] / (1 - GJR_KW["alpha"] - GJR_KW["beta"] - GJR_KW["gamma"] / 2)
    empirical = gjr_sample.double().var().item()
    assert abs(empirical / analytic - 1.0) < 0.25


def test_gjr_has_fat_tails_gbm_does_not(gjr_sample):
    assert _kurtosis(gjr_sample) > 3.5
    gbm = gbm_paths(4096, 128, seed=7, **GBM_KW)
    assert abs(_kurtosis(gbm) - 3.0) < 0.3


# ---- Heston --------------------------------------------------------------


def test_heston_variance_nonnegative_throughout():
    returns, variances = heston_paths(256, 64, seed=3, return_variance=True, **HESTON_KW)
    assert variances.shape == returns.shape
    assert (variances >= 0).all()
    assert torch.isfinite(returns).all()


def test_heston_full_truncation_survives_feller_violation():
    # 2*kappa*theta = 0.04 < xi^2 = 1.0: Feller violated, scheme must stay finite.
    kw = dict(HESTON_KW, kappa=0.5, xi=1.0)
    returns, variances = heston_paths(256, 64, seed=3, return_variance=True, **kw)
    assert (variances >= 0).all()
    assert torch.isfinite(returns).all()


# ---- prices --------------------------------------------------------------


def test_prices_from_returns_monotone_and_anchored():
    returns = torch.full((3, 5), 0.01)
    prices = prices_from_returns(returns, s0=50.0)
    assert prices.shape == (3, 6)  # initial price prepended
    assert torch.allclose(prices[:, 0], torch.full((3,), 50.0))
    assert (prices.diff(dim=-1) > 0).all()  # positive log-returns => monotone up
    assert torch.allclose(prices[:, -1], torch.full((3,), 50.0 * torch.tensor(0.05).exp()))
