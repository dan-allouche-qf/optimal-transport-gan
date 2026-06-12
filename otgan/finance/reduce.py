"""Scenario reduction by entropic optimal transport — the zero-training quant demo.

Scenario reduction asks: given N simulated market paths, pick K << N
representative paths and reweight them so that downstream risk numbers (tail
quantiles, CVaR) computed on the K scenarios match the full set. Classical
algorithms (Dupacova, Groewe-Kuska & Roemisch 2003; Heitsch & Roemisch 2003)
prune scenarios under a Wasserstein-type probability metric and move the
deleted mass to the nearest survivor. This module solves the same problem with
the exact solver that trains the image GAN — ``otgan.sinkhorn.sinkhorn`` —
which is the point of the finance track: the same OT engine, from images to
markets.

``sinkhorn_reduce`` runs a Lloyd-style alternating minimization of the
entropic transport cost between the empirical path distribution and a free
K-point support ("Sinkhorn k-means"; the centroid update is the barycentric
projection used for free-support Wasserstein barycenters, Cuturi & Doucet
2014):

    repeat:  C    = ||x_i - c_j||^2, mean-normalized (see below)
             M    = sinkhorn(1/N, w, C, epsilon)
             c_j  = sum_i M_ij x_i / M.sum(0)_j      (barycentric projection)
             w    = M.sum(0)                          (column masses, sum to 1)

As ``epsilon -> 0`` the plan hardens to nearest-scenario assignment and the
iteration recovers classical scenario reduction: centroids become cluster
means and the weights become cluster probabilities — the optimal-
redistribution rule of Dupacova et al. Entropic smoothing (``epsilon > 0``)
trades a little distortion for dense, smooth assignments and fast, stable
solves.

Cost-normalization convention (shared with the finance trainer; documented in
``otgan/finance/config.py`` and locked by tests): squared-Euclidean cost
matrices on raw paths are divided by their mean, ``C = C / C.mean()``, before
the solver, so ``epsilon`` is dimensionless and ``epsilon=1.0`` means the same
thing as in the cosine-cost-on-embeddings regime (range ``[0, 2]``).
Distortions are reported on the UNNORMALIZED cost so they keep data units
(squared summed log-return) and stay comparable across reduction methods.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from otgan.sinkhorn import sinkhorn


def pairwise_sqeuclidean(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Squared-Euclidean cost matrix between two batches of (flattened) paths.

    ``X`` is ``(n, ...)`` and ``Y`` is ``(m, ...)``; trailing dimensions are
    flattened. Returns an ``(n, m)`` matrix, clamped at 0 to absorb the small
    negative values the expanded form ``|x|^2 + |y|^2 - 2<x, y>`` can produce.
    """
    xf = X.reshape(X.shape[0], -1)
    yf = Y.reshape(Y.shape[0], -1)
    sq = (xf**2).sum(dim=1, keepdim=True) + (yf**2).sum(dim=1) - 2.0 * (xf @ yf.t())
    return sq.clamp_(min=0.0)


@dataclass
class ReducedSet:
    """A reduced scenario set: K paths, a probability vector, and a fit measure.

    ``distortion`` is the transport cost of representing the full set by
    ``paths`` under the method's assignment, on the UNNORMALIZED squared-
    Euclidean cost (data units). For hard methods this is the mean squared
    distance to the assigned representative, i.e. ``<M_hard, C>`` with row
    mass ``1/N``; for ``sinkhorn_reduce`` it is ``<M, C>`` for the entropic
    plan — total mass 1 in both cases, so the numbers are comparable.
    """

    paths: torch.Tensor  # (K, ...) representative paths, trailing shape of the input
    weights: torch.Tensor  # (K,) nonnegative, sums to 1
    distortion: float  # <plan, C_unnormalized>, in data units
    plan: torch.Tensor | None = None  # (N, K) entropic transport plan, if available


def _local_generator(seed: int) -> torch.Generator:
    """A CPU ``torch.Generator`` seeded in isolation (global RNG untouched)."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return gen


def _kmeans_pp_init(xf: torch.Tensor, K: int, gen: torch.Generator) -> torch.Tensor:
    """Seeded k-means++ initialization (Arthur & Vassilvitskii 2007) on flat rows."""
    n = xf.shape[0]
    if not 1 <= K <= n:
        raise ValueError(f"K must be in [1, n_paths={n}], got {K}")
    first = int(torch.randint(n, (1,), generator=gen).item())
    chosen = [first]
    d2 = pairwise_sqeuclidean(xf, xf[first : first + 1]).squeeze(1)
    for _ in range(K - 1):
        total = d2.sum()
        # All remaining points coincide with a chosen centroid: fall back to uniform.
        probs = torch.full_like(d2, 1.0 / n) if total <= 0 else d2 / total
        nxt = int(torch.multinomial(probs, 1, generator=gen).item())
        chosen.append(nxt)
        d2 = torch.minimum(d2, pairwise_sqeuclidean(xf, xf[nxt : nxt + 1]).squeeze(1))
    return xf[torch.tensor(chosen, dtype=torch.long)].clone()


def kmeans_reduce(X: torch.Tensor, K: int, seed: int, iters: int = 50) -> ReducedSet:
    """Classical hard reduction: seeded k-means++ init + Lloyd iterations.

    The ``epsilon -> 0`` reference point for ``sinkhorn_reduce``: hard
    nearest-centroid assignments, centroids = cluster means, weights = cluster
    frequencies (Dupacova et al.'s optimal-redistribution rule). Empty
    clusters keep their previous centroid (and get weight 0). Stops early once
    assignments stabilize.
    """
    xf = X.reshape(X.shape[0], -1)
    n = xf.shape[0]
    centroids = _kmeans_pp_init(xf, K, _local_generator(seed))
    assign: torch.Tensor | None = None
    for _ in range(iters):
        new_assign = pairwise_sqeuclidean(xf, centroids).argmin(dim=1)
        if assign is not None and torch.equal(new_assign, assign):
            break
        assign = new_assign
        counts = torch.bincount(assign, minlength=K)
        sums = torch.zeros_like(centroids)
        sums.index_add_(0, assign, xf)
        nonempty = (counts > 0).unsqueeze(1)
        means = sums / counts.clamp(min=1).to(xf.dtype).unsqueeze(1)
        centroids = torch.where(nonempty, means, centroids)
    d2min, assign = pairwise_sqeuclidean(xf, centroids).min(dim=1)
    weights = torch.bincount(assign, minlength=K).to(xf.dtype) / n
    return ReducedSet(
        paths=centroids.reshape(K, *X.shape[1:]),
        weights=weights,
        distortion=d2min.mean().item(),
    )


def random_subsample(X: torch.Tensor, K: int, seed: int) -> ReducedSet:
    """Baseline: keep K uniformly sampled paths with uniform weights ``1/K``.

    Distortion is the nearest-scenario transport cost of the FULL set onto the
    kept paths (mean over paths of the squared distance to the closest kept
    one) — the optimal hard reassignment for this fixed support, which is the
    fairest possible reading of the baseline.
    """
    xf = X.reshape(X.shape[0], -1)
    n = xf.shape[0]
    if not 1 <= K <= n:
        raise ValueError(f"K must be in [1, n_paths={n}], got {K}")
    idx = torch.randperm(n, generator=_local_generator(seed))[:K]
    kept = xf[idx]
    distortion = pairwise_sqeuclidean(xf, kept).min(dim=1).values.mean().item()
    return ReducedSet(
        paths=X[idx].clone(),
        weights=torch.full((K,), 1.0 / K, dtype=xf.dtype),
        distortion=distortion,
    )


def sinkhorn_reduce(
    X: torch.Tensor,
    K: int,
    epsilon: float = 1.0,
    sinkhorn_iters: int = 100,
    outer_iters: int = 20,
    seed: int = 0,
) -> ReducedSet:
    """Entropic-OT Lloyd scenario reduction ('Sinkhorn k-means').

    Initializes the K-point support with seeded k-means++, then alternates the
    entropic plan (via ``otgan.sinkhorn.sinkhorn``, uniform source marginal,
    current weights as target marginal) with the barycentric-projection
    centroid update and the column-mass weight update — see the module
    docstring for the scheme and its ``epsilon -> 0`` classical limit.

    Per the repo convention, the solver sees the mean-normalized cost
    ``C / C.mean()`` (dimensionless ``epsilon``); the returned ``distortion``
    is ``<M, C>`` on the unnormalized cost (data units). A final solve against
    the final centroids makes ``plan``, ``weights`` and ``distortion``
    mutually consistent. Deterministic given ``seed``.
    """
    xf = X.reshape(X.shape[0], -1)
    n = xf.shape[0]
    a = torch.full((n,), 1.0 / n, dtype=xf.dtype)
    w = torch.full((K,), 1.0 / K, dtype=xf.dtype)
    centroids = _kmeans_pp_init(xf, K, _local_generator(seed))

    def solve(c_raw: torch.Tensor) -> torch.Tensor:
        return sinkhorn(a, w, c_raw / c_raw.mean(), epsilon, sinkhorn_iters)

    for _ in range(outer_iters):
        M = solve(pairwise_sqeuclidean(xf, centroids))
        col = M.sum(dim=0)
        centroids = (M.t() @ xf) / col.clamp(min=1e-12).unsqueeze(1)
        w = col / col.sum()  # renormalize away finite-iteration mass error
    c_raw = pairwise_sqeuclidean(xf, centroids)
    M = solve(c_raw)
    col = M.sum(dim=0)
    return ReducedSet(
        paths=centroids.reshape(K, *X.shape[1:]),
        weights=col / col.sum(),
        distortion=(M * c_raw).sum().item(),
        plan=M,
    )


def holdout_split(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic even/odd split of a path set into (fit, held_out) halves.

    Fit the reduction on the first half and evaluate ``reduction_report`` on
    the second, so the report measures generalization of the reduced set
    rather than in-sample overfit of K support points to N paths.
    """
    return X[0::2], X[1::2]


def _weighted_quantile(values: torch.Tensor, weights: torch.Tensor, alpha: float) -> torch.Tensor:
    """Quantile of a discrete weighted distribution: smallest v with CDF(v) >= alpha."""
    order = torch.argsort(values)
    sorted_values = values[order]
    cdf = torch.cumsum(weights[order], dim=0)
    idx = int(torch.searchsorted(cdf, torch.tensor(alpha, dtype=cdf.dtype)).item())
    return sorted_values[min(idx, len(sorted_values) - 1)]


def reduction_report(
    full: torch.Tensor,
    reduced: ReducedSet,
    alphas: tuple[float, ...] = (0.95, 0.99),
) -> dict[str, float]:
    """Risk-statistics comparison: weighted reduced set vs a held-out full set.

    Out-of-sample protocol (enforced by convention, documented here): ``full``
    must be the HELD-OUT half of the simulated paths — split deterministically
    by even/odd index via ``holdout_split`` — and ``reduced`` must have been
    fit on the *other* half. Passing the fit half instead silently turns the
    report into an in-sample comparison.

    The compared statistic is the terminal log-return sum ``t = sum_u r_u``
    (log of the gross return over the horizon). Reports mean, std, the
    ``alpha``-quantiles of ``t``, and the CVaR of the terminal loss
    ``L = -t``: ``CVaR_alpha = E[L | L >= VaR_alpha]`` with
    ``VaR_alpha = quantile(L, alpha)`` (discrete-tail estimator in the spirit
    of Rockafellar & Uryasev 2000). Keys: ``mean_full``, ``mean_reduced``,
    ``std_full``, ``std_reduced``, and per alpha ``q{alpha}_full/_reduced``
    and ``cvar{alpha}_full/_reduced``. All torch, no scipy.
    """
    t_full = full.reshape(full.shape[0], -1).sum(dim=1).double()
    t_red = reduced.paths.reshape(reduced.paths.shape[0], -1).sum(dim=1).double()
    w = reduced.weights.double()
    w = w / w.sum()

    mean_red = (w * t_red).sum()
    report: dict[str, float] = {
        "mean_full": t_full.mean().item(),
        "mean_reduced": mean_red.item(),
        "std_full": t_full.std(correction=0).item(),
        "std_reduced": (w * (t_red - mean_red) ** 2).sum().sqrt().item(),
    }
    loss_full, loss_red = -t_full, -t_red
    for alpha in alphas:
        report[f"q{alpha:g}_full"] = torch.quantile(t_full, alpha).item()
        report[f"q{alpha:g}_reduced"] = _weighted_quantile(t_red, w, alpha).item()

        var_full = torch.quantile(loss_full, alpha)
        report[f"cvar{alpha:g}_full"] = loss_full[loss_full >= var_full].mean().item()
        var_red = _weighted_quantile(loss_red, w, alpha)
        tail = loss_red >= var_red
        tail_mass = w[tail].sum()
        report[f"cvar{alpha:g}_reduced"] = ((w[tail] * loss_red[tail]).sum() / tail_mass).item()
    return report
