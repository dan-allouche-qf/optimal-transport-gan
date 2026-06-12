"""Critic-free evaluation: stylized-facts table and a debiased Sinkhorn divergence.

GAN loss values are useless for model comparison, so the finance track is
judged the way a quant desk would judge a market simulator: do the generated
paths reproduce the stylized facts of asset returns (Cont 2001)? Near-zero
linear autocorrelation, fat tails (positive excess kurtosis), volatility
clustering (slowly decaying ACF of ``|r|`` and ``r^2``), and the leverage
effect — negative ``corr(r_t, r_{t+k}^2)`` (Glosten, Jagannathan & Runkle
1993; Cont's fact #8). The leverage row is the designed trap: a *symmetric*
GARCH matches kurtosis and vol clustering yet has exactly zero leverage
correlation, so a table without that row cannot tell it apart from
GJR-GARCH — the tests encode precisely this failure mode.

``sinkhorn_divergence_metric`` compresses distributional fit into one number:
the debiased entropic divergence

    S_eps(X, Y) = W_eps(X, Y) - W_eps(X, X) / 2 - W_eps(Y, Y) / 2

(Genevay, Peyre & Cuturi, AISTATS 2018; same-batch self-terms following Feydy
et al., AISTATS 2019), computed on standardized raw paths with the repo's
mean-normalized squared-Euclidean cost so ``epsilon`` is dimensionless. Up to
swapping critic embeddings for raw paths, the training objective satisfies
``D^2 = 2 x this divergence`` (the energy distance is two cross terms minus
the two self terms — see ``otgan/energy.py``), so eval-time numbers live on
the training objective's scale.
"""

from __future__ import annotations

import torch

from otgan.finance.reduce import pairwise_sqeuclidean
from otgan.sinkhorn import sinkhorn


def acf(x: torch.Tensor, max_lag: int) -> torch.Tensor:
    """Mean-over-paths autocorrelation function of ``(n_paths, T)`` series.

    Per path: demean, estimate ``rho_k = sum_t xc_t xc_{t+k} / sum_t xc_t^2``,
    then average over paths. Returns a ``(max_lag + 1,)`` float64 tensor whose
    lag-0 entry is identically 1. Vectorized across paths; only the (short)
    lag loop is Python.
    """
    if x.ndim != 2:
        raise ValueError(f"expected (n_paths, T), got shape {tuple(x.shape)}")
    if not 0 <= max_lag < x.shape[1]:
        raise ValueError(f"max_lag must be in [0, T={x.shape[1]}), got {max_lag}")
    xd = x.double()
    xc = xd - xd.mean(dim=1, keepdim=True)
    denom = (xc * xc).sum(dim=1).clamp(min=1e-30)
    out = torch.empty(max_lag + 1, dtype=torch.float64)
    out[0] = 1.0
    for k in range(1, max_lag + 1):
        out[k] = ((xc[:, :-k] * xc[:, k:]).sum(dim=1) / denom).mean()
    return out


def excess_kurtosis(x: torch.Tensor) -> float:
    """Pooled Fisher excess kurtosis ``m4 / m2^2 - 3`` (Gaussian -> 0).

    Pools all paths and time steps into one sample: the stylized fact is about
    the unconditional return distribution, and pooling is what makes GARCH-
    style mixtures of normals show their fat tails.
    """
    v = x.reshape(-1).double()
    c = v - v.mean()
    m2 = (c**2).mean()
    m4 = (c**4).mean()
    return (m4 / m2**2 - 3.0).item()


def leverage_corr(r: torch.Tensor, max_lag: int) -> torch.Tensor:
    """Leverage-effect correlations ``corr(r_t, r_{t+k}^2)`` for k = 1..max_lag.

    Pools over paths and time. Negative values mean negative returns raise
    future volatility more than positive ones — true for GJR-GARCH with
    ``gamma > 0`` and for Heston with ``rho < 0``, but ~0 for a *symmetric*
    GARCH (``gamma = 0``), which still clusters volatility. That asymmetry is
    exactly what ACF-of-``r^2`` rows cannot see and this statistic can — the
    trap the tests encode. Returns a ``(max_lag,)`` float64 tensor.
    """
    if r.ndim != 2:
        raise ValueError(f"expected (n_paths, T), got shape {tuple(r.shape)}")
    if not 1 <= max_lag < r.shape[1]:
        raise ValueError(f"max_lag must be in [1, T={r.shape[1]}), got {max_lag}")
    rd = r.double()
    out = torch.empty(max_lag, dtype=torch.float64)
    for k in range(1, max_lag + 1):
        a = rd[:, :-k].reshape(-1)
        b = (rd[:, k:] ** 2).reshape(-1)
        ac = a - a.mean()
        bc = b - b.mean()
        denom = (ac.square().mean() * bc.square().mean()).sqrt().clamp(min=1e-30)
        out[k - 1] = (ac * bc).mean() / denom
    return out


def stylized_facts_table(
    real: torch.Tensor,
    fake: torch.Tensor,
    lags: tuple[int, ...] = (1, 5, 10, 20),
) -> dict[str, dict[str, float]]:
    """Side-by-side stylized-facts statistics for target vs generated paths.

    Rows: pooled excess kurtosis; ACF of raw returns at lag 1 (should be ~0
    for both — markets are hard to predict, returns are easy to decorrelate);
    ACF of ``|r|`` and ``r^2`` at ``lags`` (volatility clustering); leverage
    correlation at lags 1 and 5 (the GARCH-vs-GJR discriminator). Returns
    ``{row_label: {"target": float, "generated": float}}``, insertion-ordered
    for rendering by ``to_markdown``.
    """
    max_lag = max(*lags, 5)
    table: dict[str, dict[str, float]] = {}

    def row(label: str, target: float, generated: float) -> None:
        table[label] = {"target": float(target), "generated": float(generated)}

    row("excess kurtosis", excess_kurtosis(real), excess_kurtosis(fake))
    acf_r = (acf(real, 1), acf(fake, 1))
    row("ACF(r) lag-1", acf_r[0][1].item(), acf_r[1][1].item())
    acf_abs = (acf(real.abs(), max_lag), acf(fake.abs(), max_lag))
    for k in lags:
        row(f"ACF(|r|) lag-{k}", acf_abs[0][k].item(), acf_abs[1][k].item())
    acf_sq = (acf(real**2, max_lag), acf(fake**2, max_lag))
    for k in lags:
        row(f"ACF(r^2) lag-{k}", acf_sq[0][k].item(), acf_sq[1][k].item())
    lev = (leverage_corr(real, 5), leverage_corr(fake, 5))
    for k in (1, 5):
        row(f"leverage corr lag-{k}", lev[0][k - 1].item(), lev[1][k - 1].item())
    return table


def to_markdown(table: dict[str, dict[str, float]]) -> str:
    """Render a ``stylized_facts_table`` dict as a GitHub-flavored markdown table."""
    lines = ["| statistic | target | generated |", "| --- | ---: | ---: |"]
    lines += [
        f"| {label} | {cols['target']:+.4f} | {cols['generated']:+.4f} |"
        for label, cols in table.items()
    ]
    return "\n".join(lines)


def _subsample(x: torch.Tensor, cap: int, gen: torch.Generator) -> torch.Tensor:
    if x.shape[0] <= cap:
        return x
    return x[torch.randperm(x.shape[0], generator=gen)[:cap]]


def sinkhorn_divergence_metric(
    X: torch.Tensor,
    Y: torch.Tensor,
    epsilon: float = 1.0,
    iters: int = 100,
    seed: int = 0,
) -> float:
    """Debiased Sinkhorn divergence between two path sets, on raw paths.

    ``S = W(X, Y) - 0.5 W(X, X) - 0.5 W(Y, Y)`` where each ``W`` is the primal
    entropic cost ``<M, C>`` with uniform marginals and *same-batch* self-terms
    (Feydy et al. 2019; divergence form from Genevay et al. 2018). The
    training objective is exactly ``D^2 = 2 x`` this divergence, with critic
    embeddings in place of raw paths (see ``otgan/energy.py``), so the metric
    reads on the objective's scale. Near 0 for two samples of the same
    process, clearly positive across processes. Caveat for interpretation:
    the finite-sample floor of the same-process value grows with path
    dimension and tail weight (Genevay et al. 2019's sample-complexity
    constants), so heavy-tailed pairs read higher than Gaussian ones even
    when the processes match.

    Conventions: paths are jointly standardized (pooled mean/std over both
    sets, preserving any scale mismatch between them); all three blocks share
    ONE squared-Euclidean cost normalized by the pooled mean cost — a single
    normalizer, otherwise the three terms would live on different scales and
    debiasing would break. Each side is subsampled to <= 1024 paths with a
    seeded local generator for tractability; deterministic given ``seed``.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    xs = _subsample(X, 1024, gen).reshape(-1, X[0].numel())
    ys = _subsample(Y, 1024, gen).reshape(-1, Y[0].numel())

    pooled = torch.cat([xs.reshape(-1), ys.reshape(-1)])
    mu, sd = pooled.mean(), pooled.std().clamp(min=1e-12)
    z = torch.cat([(xs - mu) / sd, (ys - mu) / sd], dim=0)

    C = pairwise_sqeuclidean(z, z)
    C = C / C.mean()
    n = xs.shape[0]

    def w(block: torch.Tensor) -> torch.Tensor:
        n1, n2 = block.shape
        a = torch.full((n1,), 1.0 / n1, dtype=block.dtype)
        b = torch.full((n2,), 1.0 / n2, dtype=block.dtype)
        M = sinkhorn(a, b, block, epsilon, iters)
        return (M * block).sum()

    s = w(C[:n, n:]) - 0.5 * w(C[:n, :n]) - 0.5 * w(C[n:, n:])
    return s.item()
