"""Entropic optimal transport via the Sinkhorn algorithm (log-domain, stable).

Two fixes over the original notebook:

1. **Log-domain iterations.** The original computed ``K = exp(-C / epsilon)`` and
   iterated divisions in the raw domain, which under/overflows for small epsilon.
   Here the scaling vectors live in log-space and updates use ``logsumexp``, so the
   solver is stable for any epsilon > 0 and can produce sharp plans.

2. **Detached plan (intentional).** The iterations run under ``no_grad`` and the
   returned plan ``M`` carries no gradient. By Danskin's / the envelope theorem,
   the gradient of the entropic OT cost equals the cost contracted with the fixed
   optimal plan, so gradients correctly flow through the cost matrix ``C`` (i.e.
   through the critic) while the plan is treated as a constant.
"""

import torch


def cost(emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
    """Cosine distance between two batches of (L2-normalized) embeddings.

    Returns an ``(n1, n2)`` matrix in ``[0, 2]``. Valid as a cosine distance only
    because the critic L2-normalizes its output; do not normalize again here.
    """
    return 1.0 - emb1 @ emb2.t()


def sinkhorn(
    a: torch.Tensor, b: torch.Tensor, C: torch.Tensor, epsilon: float, iters: int
) -> torch.Tensor:
    """Optimal transport plan between marginals ``a`` and ``b`` for cost ``C``.

    The plan ``M`` satisfies (approximately) ``M.sum(1) == a`` and ``M.sum(0) == b``.
    Detached from autograd by construction.
    """
    with torch.no_grad():
        log_a = torch.log(a + 1e-10)
        log_b = torch.log(b + 1e-10)
        log_K = -C / epsilon
        log_u = torch.zeros_like(a)
        log_v = torch.zeros_like(b)
        for _ in range(iters):
            # u_i = a_i / sum_j K_ij v_j   (in log-space)
            log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
            # v_j = b_j / sum_i K_ij u_i
            log_v = log_b - torch.logsumexp(log_K.t() + log_u.unsqueeze(0), dim=1)
        # M_ij = u_i K_ij v_j
        M = torch.exp(log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0))
    return M
