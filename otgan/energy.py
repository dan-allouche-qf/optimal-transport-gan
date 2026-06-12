"""Minibatch OT objectives: energy distance and debiased Sinkhorn divergence.

**Energy distance** (Salimans et al. 2018, eq. for D^2):

    D^2(p, g) = 2 E[W_c(X, Y)] - E[W_c(X, X')] - E[W_c(Y, Y')]

estimated from two independent real minibatches (X, X') and two independent
generated minibatches (Y, Y'). With the four cross pairs and the two within
pairs this is assembled as:

    D^2 = W02 + W03 + W12 + W13 - 2*W01 - 2*W23

**Sinkhorn divergence** (Genevay, Peyre & Cuturi, AISTATS 2018; theory in
Feydy et al., AISTATS 2019):

    S_eps(p, g) = W_eps(p, g) - 1/2 W_eps(p, p) - 1/2 W_eps(g, g)

The energy distance above is exactly 2x a Sinkhorn divergence whose self-terms
are estimated with *independent* minibatches (W(X, X')); the debiased variant
implemented by ``sinkhorn_divergence`` uses *same-batch* self-terms (W(X, X)),
which is Feydy et al.'s estimator. Two deviations from their exact framework
are intentional and documented: (1) the per-pair cost here is the biased primal
entropic cost <M, C> (as in the OT-GAN paper), not the dual-potential form;
(2) marginals are uniform over the minibatch. Both objectives are assembled at
the same 4-cross-pair scale so their training curves are directly comparable.

The generator MINIMIZES the objective; the critic MAXIMIZES it (see
otgan/trainer.py). This module is the single source of truth for the objective,
used by both training and evaluation so the sign convention can never diverge.
"""

from dataclasses import dataclass

import torch

from otgan.sinkhorn import cost, sinkhorn


@dataclass
class EnergyTerms:
    total: torch.Tensor  # D^2
    cross: torch.Tensor  # W02 + W03 + W12 + W13  (estimates 2 E[W(X,Y)])
    real_real: torch.Tensor  # W01  (E[W(X,X')])
    fake_fake: torch.Tensor  # W23  (E[W(Y,Y')])

    def to_floats(self) -> dict:
        return {
            "energy_distance": self.total.detach().item(),
            "cross": self.cross.detach().item(),
            "real_real": self.real_real.detach().item(),
            "fake_fake": self.fake_fake.detach().item(),
        }


def _wasserstein(e1: torch.Tensor, e2: torch.Tensor, epsilon: float, iters: int) -> torch.Tensor:
    """Entropic OT cost <M, C> between two embedding batches (uniform marginals)."""
    C = cost(e1, e2)
    n1, n2 = C.shape
    a = torch.full((n1,), 1.0 / n1, device=C.device, dtype=C.dtype)
    b = torch.full((n2,), 1.0 / n2, device=C.device, dtype=C.dtype)
    M = sinkhorn(a, b, C, epsilon, iters)  # detached
    return (M * C).sum()


def energy_distance(embeddings, epsilon: float, iters: int) -> EnergyTerms:
    """Assemble D^2 from the four critic embeddings (cr1, cr2, cf1, cf2)."""
    cr1, cr2, cf1, cf2 = embeddings

    def w(x, y):
        return _wasserstein(x, y, epsilon, iters)

    real_real = w(cr1, cr2)  # W01
    fake_fake = w(cf1, cf2)  # W23
    cross = w(cr1, cf1) + w(cr1, cf2) + w(cr2, cf1) + w(cr2, cf2)  # W02+W03+W12+W13
    total = cross - 2.0 * real_real - 2.0 * fake_fake
    return EnergyTerms(total=total, cross=cross, real_real=real_real, fake_fake=fake_fake)


def sinkhorn_divergence(embeddings, epsilon: float, iters: int) -> EnergyTerms:
    """Debiased Sinkhorn divergence at the same scale as ``energy_distance``.

    Self-terms use the SAME batch (W(X, X), W(Y, Y)) following Feydy et al.
    (AISTATS 2019) — for entropic OT these are strictly positive, and
    subtracting them removes the entropic bias that makes the raw cost
    collapse toward a constant for large epsilon. Assembled as

        total = cross - 2 * mean[W(Xi, Xi)] - 2 * mean[W(Yi, Yi)]

    (~ 4 * S_eps) so curves are directly comparable with the energy distance.
    ``real_real``/``fake_fake`` log the same-batch self-terms.
    """
    cr1, cr2, cf1, cf2 = embeddings

    def w(x, y):
        return _wasserstein(x, y, epsilon, iters)

    real_real = 0.5 * (w(cr1, cr1) + w(cr2, cr2))  # ~ W_eps(p, p)
    fake_fake = 0.5 * (w(cf1, cf1) + w(cf2, cf2))  # ~ W_eps(g, g)
    cross = w(cr1, cf1) + w(cr1, cf2) + w(cr2, cf1) + w(cr2, cf2)
    total = cross - 2.0 * real_real - 2.0 * fake_fake
    return EnergyTerms(total=total, cross=cross, real_real=real_real, fake_fake=fake_fake)


LOSSES = {
    "energy_distance": energy_distance,
    "sinkhorn_divergence": sinkhorn_divergence,
}


def compute_loss(
    embeddings, epsilon: float, iters: int, loss: str = "energy_distance"
) -> EnergyTerms:
    """Dispatch to the configured minibatch OT objective (see ``Config.loss``)."""
    try:
        fn = LOSSES[loss]
    except KeyError:
        raise ValueError(f"loss must be one of {sorted(LOSSES)}, got {loss!r}") from None
    return fn(embeddings, epsilon, iters)
