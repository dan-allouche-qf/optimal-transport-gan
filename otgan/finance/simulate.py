"""Seeded market-path simulators: GBM, Heston and GJR-GARCH (torch-only).

These produce the ``(n_paths, seq_len)`` float32 **log-return** tensors that
the finance OT-GAN trains on — the same minibatch energy-distance engine as
the image experiments (``otgan.energy``), pointed at markets instead of
pixels. Each simulator takes an explicit ``seed`` and draws from a *local*
``torch.Generator``, never the global RNG: an explicit seed in, a bit-exact
tensor out, regardless of what other code (or a concurrent image run) does to
torch's global state. Everything is vectorized across paths; only time is
looped. Tensors live on CPU — paths are tiny, and CPU avoids MPS float64
quirks; callers move them to a device if needed.

The three processes form a difficulty ladder of stylized facts:

- **GBM** (Black & Scholes 1973): iid Gaussian log-returns via the exact
  discretization ``r = (mu - sigma^2/2) dt + sigma sqrt(dt) Z``. The control —
  no fat tails, no volatility clustering, nothing for the critic to find.
- **Heston** (Heston 1993): stochastic variance
  ``dv = kappa (theta - v) dt + xi sqrt(v) dW_v`` with
  ``corr(dW_S, dW_v) = rho``, discretized by full-truncation Euler
  (Lord, Koekkoek & van Dijk 2010). Volatility clusters; ``rho < 0`` adds a
  diffusive leverage effect.
- **GJR-GARCH(1,1)** (Glosten, Jagannathan & Runkle 1993): discrete-time
  conditional variance with an asymmetric ARCH term — fat tails, clustering
  and a sharp leverage asymmetry, the hardest target of the three.
"""

from __future__ import annotations

from typing import Literal, overload

import torch


def _local_generator(seed: int) -> torch.Generator:
    """A CPU ``torch.Generator`` seeded in isolation (global RNG untouched)."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return gen


def gbm_paths(n: int, seq_len: int, mu: float, sigma: float, dt: float, seed: int) -> torch.Tensor:
    """Geometric Brownian motion log-returns, exact discretization.

    Under GBM ``dS/S = mu dt + sigma dW``, log-returns over a step ``dt`` are
    exactly ``r = (mu - sigma^2/2) dt + sigma sqrt(dt) Z`` with ``Z ~ N(0, 1)``
    — no discretization error at any ``dt``. Returns ``(n, seq_len)`` float32.
    """
    gen = _local_generator(seed)
    z = torch.randn(n, seq_len, generator=gen)
    drift = (mu - 0.5 * sigma**2) * dt
    return drift + sigma * dt**0.5 * z


@overload
def heston_paths(
    n: int,
    seq_len: int,
    mu: float,
    dt: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    v0: float | None = ...,
    *,
    seed: int,
    return_variance: Literal[False] = ...,
) -> torch.Tensor: ...


@overload
def heston_paths(
    n: int,
    seq_len: int,
    mu: float,
    dt: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    v0: float | None = ...,
    *,
    seed: int,
    return_variance: Literal[True],
) -> tuple[torch.Tensor, torch.Tensor]: ...


def heston_paths(
    n: int,
    seq_len: int,
    mu: float,
    dt: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    v0: float | None = None,
    *,
    seed: int,
    return_variance: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Heston log-returns via the full-truncation Euler scheme.

    Model: ``d log S = (mu - v/2) dt + sqrt(v) dW_S``,
    ``dv = kappa (theta - v) dt + xi sqrt(v) dW_v`` with
    ``corr(dW_S, dW_v) = rho``. The Feller condition ``2 kappa theta > xi^2``
    guarantees ``v > 0`` in continuous time, but the Euler discretization can
    drive ``v`` negative even when it holds — and realistic calibrations often
    violate it outright. Full truncation (Lord, Koekkoek & van Dijk 2010)
    handles both cases: the update keeps the *untruncated* ``v`` as state but
    feeds ``v+ = max(v, 0)`` into the drift, the diffusion and the return, so
    the scheme stays well defined (no sqrt of a negative) and exhibits the
    smallest discretization bias among the simple Euler fixes.

    ``v0`` defaults to the long-run variance ``theta``. Returns ``(n, seq_len)``
    float32 log-returns; with ``return_variance=True`` also returns the
    ``(n, seq_len)`` truncated variance ``v+`` actually used at each step
    (nonnegative by construction).
    """
    if v0 is None:
        v0 = theta
    gen = _local_generator(seed)
    z = torch.randn(2, n, seq_len, generator=gen)
    z_v = z[0]
    z_s = rho * z_v + (1.0 - rho**2) ** 0.5 * z[1]

    v = torch.full((n,), float(v0))
    returns = torch.empty(n, seq_len)
    variances = torch.empty(n, seq_len)
    for t in range(seq_len):
        v_plus = v.clamp(min=0.0)
        variances[:, t] = v_plus
        vol = torch.sqrt(v_plus * dt)
        returns[:, t] = (mu - 0.5 * v_plus) * dt + vol * z_s[:, t]
        v = v + kappa * (theta - v_plus) * dt + xi * vol * z_v[:, t]
    if return_variance:
        return returns, variances
    return returns


def gjr_garch_paths(
    n: int,
    seq_len: int,
    omega: float,
    alpha: float,
    beta: float,
    gamma: float,
    seed: int,
    burn_in: int = 200,
) -> torch.Tensor:
    """GJR-GARCH(1,1) log-returns (Glosten, Jagannathan & Runkle 1993).

    Model: ``r_t = sigma_t z_t`` with ``z_t ~ N(0, 1)`` and

        ``sigma_t^2 = omega + (alpha + gamma * 1[r_{t-1} < 0]) r_{t-1}^2
                      + beta sigma_{t-1}^2``.

    ``gamma > 0`` creates the leverage effect that symmetric GARCH lacks:
    negative returns raise next-period variance more than positive returns of
    the same size. Covariance stationarity (with symmetric innovations)
    requires ``alpha + beta + gamma/2 < 1``, in which case the unconditional
    variance is ``omega / (1 - alpha - beta - gamma/2)`` — used here to
    initialize ``sigma_0^2``. The first ``burn_in`` steps are simulated and
    discarded so the retained window is drawn from (approximately) the
    stationary distribution. Returns ``(n, seq_len)`` float32.
    """
    persistence = alpha + beta + gamma / 2.0
    if persistence >= 1.0:
        raise ValueError(
            "GJR-GARCH stationarity requires alpha + beta + gamma/2 < 1, "
            f"got {alpha} + {beta} + {gamma}/2 = {persistence}"
        )
    gen = _local_generator(seed)
    total = burn_in + seq_len
    z = torch.randn(n, total, generator=gen)

    sigma2 = torch.full((n,), omega / (1.0 - persistence))
    returns = torch.empty(n, total)
    for t in range(total):
        r = torch.sqrt(sigma2) * z[:, t]
        returns[:, t] = r
        sigma2 = omega + (alpha + gamma * (r < 0).float()) * r**2 + beta * sigma2
    return returns[:, burn_in:].contiguous()


def prices_from_returns(returns: torch.Tensor, s0: float = 100.0) -> torch.Tensor:
    """Price paths from log-returns: ``P_t = s0 * exp(sum_{u<=t} r_u)``.

    Prepends the initial price, so a ``(..., T)`` return tensor maps to a
    ``(..., T + 1)`` price tensor with ``prices[..., 0] == s0``.
    """
    prices = s0 * torch.exp(torch.cumsum(returns, dim=-1))
    first = torch.full_like(prices[..., :1], s0)
    return torch.cat([first, prices], dim=-1)
