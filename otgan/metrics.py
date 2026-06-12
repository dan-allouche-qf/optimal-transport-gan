"""Quantitative evaluation: FID, KID and FID-LeNet (plus IS as a sanity check).

A single InceptionV3 forward pass per image batch — torchmetrics'
``NoTrainInceptionV3`` with ``features_list=['2048', 'logits_unbiased']`` —
yields both the 2048-d pool features and the unbiased class logits, which feed
three metrics without re-running the backbone:

- **FID** (Heusel et al., NeurIPS 2017): Frechet distance between Gaussians
  fitted to the 2048-d features, computed in float64 by ``_frechet`` (the
  matrix-square-root trace comes from the eigenvalues of ``sigma1 @ sigma2``,
  the same closed form torchmetrics uses; cross-checked against
  ``FrechetInceptionDistance`` in tests/test_metrics.py).
- **KID** (Binkowski et al., "Demystifying MMD GANs", ICLR 2018): the unbiased
  MMD^2 estimator with the cubic polynomial kernel ``k(x, y) = (x.y/d + 1)^3``,
  averaged over ``KID_SUBSETS`` random subsets of ``cfg.kid_subset_size``
  samples per side — the semantics of torchmetrics'
  ``KernelInceptionDistance``. Reported as ``kid_mean``/``kid_std``.
- **IS** (Salimans et al., NeurIPS 2016) from the unbiased logits. IS is kept
  as a SANITY CHECK ONLY: it scores ImageNet class confidence/diversity, which
  is close to meaningless for 32x32 digits (Barratt & Sharma, 2018) — it is
  reported (``is_mean``/``is_std``) but models should not be compared on it.

**FID-LeNet** complements Inception FID with a domain-relevant feature space:
the same ``_frechet`` computation in the 128-d penultimate features of a LeNet
trained on MNIST (``otgan/lenet.py``). The Inception path consumes 3-channel
uint8 images (``_to_three_channels`` + ``_to_uint8``); the LeNet consumes
``[0, 1]`` grayscale directly. ``fid_lenet`` is only computed when
``cfg.dataset == 'mnist'`` (the key is simply absent otherwise).

Real-side features are computed once per (dataset, split, n_eval, feature
space) and disk-cached under ``resolve(cfg.eval_cache_dir)`` as
``{dataset}_{split}_n{n_eval}_{space}.pt`` holding ``{mu, sigma}`` — plus the
raw ``features`` for the Inception space, which KID's subset resampling needs.
The filename carries the full cache key, so changing any component recomputes;
cache hits log ``[eval] using cached real features ...``.

All statistics run on CPU (float64) when the training device is MPS, which
lacks float64 support; generated images are produced in ``cfg.batch_size``
chunks. KID/IS subset shuffling uses a ``torch.Generator`` seeded from
``cfg.seed`` so repeated evaluations of the same model are reproducible.
"""

from pathlib import Path

import torch
from torchmetrics.image.fid import NoTrainInceptionV3

from otgan.data import build_dataloader
from otgan.lenet import MNISTLeNet, train_or_load_lenet
from otgan.paths import resolve
from otgan.trainer import denormalize

KID_SUBSETS = 100  # number of random subsets; torchmetrics' default
IS_SPLITS = 10  # torchmetrics' default for InceptionScore


def _to_three_channels(imgs01: torch.Tensor) -> torch.Tensor:
    """Expand grayscale to 3 channels for InceptionV3 and clamp to [0, 1]."""
    if imgs01.shape[1] == 1:
        imgs01 = imgs01.repeat(1, 3, 1, 1)
    return imgs01.clamp(0, 1)


def _to_uint8(imgs01: torch.Tensor) -> torch.Tensor:
    """[0, 1] floats -> uint8, exactly as torchmetrics' ``normalize=True`` path."""
    return (imgs01 * 255).byte()


def _metric_device(device: torch.device) -> torch.device:
    """Metrics need float64 covariance math, which MPS lacks — use CPU there."""
    return torch.device("cpu") if device.type == "mps" else device


def _mean_cov(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """float64 mean and unbiased covariance of an ``(N, d)`` feature matrix."""
    f = features.double()
    mu = f.mean(dim=0)
    centered = f - mu
    return mu, centered.t().mm(centered) / (f.shape[0] - 1)


def _frechet(
    mu1: torch.Tensor, sigma1: torch.Tensor, mu2: torch.Tensor, sigma2: torch.Tensor
) -> float:
    """Frechet distance ||mu1-mu2||^2 + Tr(S1 + S2 - 2 (S1 S2)^1/2) in float64.

    The square-root trace is the sum of the (clamped-real) square roots of the
    eigenvalues of ``S1 @ S2`` — the same closed form as torchmetrics'
    ``_compute_fid``, avoiding an iterative ``sqrtm``.
    """
    mu1, mu2 = mu1.double(), mu2.double()
    sigma1, sigma2 = sigma1.double(), sigma2.double()
    a = (mu1 - mu2).square().sum()
    b = sigma1.trace() + sigma2.trace()
    c = torch.linalg.eigvals(sigma1 @ sigma2).sqrt().real.sum()
    return float(a + b - 2.0 * c)


def _poly_mmd(f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
    """Unbiased MMD^2 with the cubic polynomial kernel (Binkowski et al. 2018).

    ``k(x, y) = (x . y / d + 1)^3`` with ``d`` the feature dimension; the
    within-set kernel diagonals are dropped (the unbiased estimator, eq. 2 of
    the paper — matching torchmetrics' ``poly_mmd``). Both batches must share
    the same size ``m``; computed in float64.
    """
    f1, f2 = f1.double(), f2.double()
    m, d = f1.shape
    k11 = (f1 @ f1.t() / d + 1.0) ** 3
    k22 = (f2 @ f2.t() / d + 1.0) ** 3
    k12 = (f1 @ f2.t() / d + 1.0) ** 3
    within = (k11.sum() - k11.trace() + k22.sum() - k22.trace()) / (m * (m - 1))
    return within - 2.0 * k12.sum() / (m * m)


def _kid(
    real: torch.Tensor,
    fake: torch.Tensor,
    subset_size: int,
    subsets: int,
    generator: torch.Generator,
) -> tuple[float, float]:
    """KID mean/std over random subsets (``KernelInceptionDistance`` semantics).

    ``subset_size`` is clamped to the available sample count so tiny smoke
    configurations still evaluate; when the clamp is active every subset is a
    permutation of the full set and the std collapses to ~0. The subset draws
    use the supplied seeded ``generator`` (reproducible, unlike torchmetrics'
    global-RNG draws).
    """
    m = min(subset_size, real.shape[0], fake.shape[0])
    scores = []
    for _ in range(subsets):
        idx_real = torch.randperm(real.shape[0], generator=generator)[:m].to(real.device)
        idx_fake = torch.randperm(fake.shape[0], generator=generator)[:m].to(fake.device)
        scores.append(_poly_mmd(real[idx_real], fake[idx_fake]))
    stacked = torch.stack(scores)
    return float(stacked.mean()), float(stacked.std(correction=0))


def _inception_score(
    logits: torch.Tensor, splits: int, generator: torch.Generator
) -> tuple[float, float]:
    """IS = exp(E_x KL(p(y|x) || p(y))) over ``splits`` chunks (Salimans et al. 2016).

    Operates on the *unbiased* logits head, as torchmetrics does. The pre-split
    shuffle uses the supplied seeded ``generator`` so evaluations are
    reproducible. Sanity-check only on MNIST — see the module docstring.
    """
    idx = torch.randperm(logits.shape[0], generator=generator).to(logits.device)
    logits = logits[idx].double()
    prob = logits.softmax(dim=1)
    log_prob = logits.log_softmax(dim=1)
    scores = []
    for p, lp in zip(prob.chunk(splits), log_prob.chunk(splits), strict=True):
        kl = (p * (lp - p.mean(dim=0, keepdim=True).log())).sum(dim=1).mean()
        scores.append(kl.exp())
    stacked = torch.stack(scores)
    std = float(stacked.std()) if stacked.numel() > 1 else 0.0
    return float(stacked.mean()), std


class FIDISEvaluator:
    """Shared evaluator for training (``BaseTrainer._evaluate``) and the CLI.

    ``evaluate(trainer)`` -> ``{fid, kid_mean, kid_std, fid_lenet, is_mean,
    is_std}`` (``fid_lenet`` present only on MNIST). ``fid_floor()`` -> the
    train-vs-test floors ``{fid, fid_lenet}``. Real-side statistics for the
    test split are computed (or loaded from the disk cache) once at
    construction; each ``evaluate`` call only re-runs the generated side.
    """

    def __init__(self, cfg, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.metric_device = _metric_device(device)
        self.inception = NoTrainInceptionV3(
            name="inception-v3-compat", features_list=["2048", "logits_unbiased"]
        ).to(self.metric_device)
        # FID-LeNet is MNIST-specific: the classifier is trained on MNIST digits.
        self.lenet: MNISTLeNet | None = (
            train_or_load_lenet(cfg, device=self.metric_device) if cfg.dataset == "mnist" else None
        )
        self._real = self._real_stats(train=False)

    # ---- cache layout ----------------------------------------------------
    def _spaces(self) -> list[str]:
        return ["inception", "lenet"] if self.lenet is not None else ["inception"]

    def _cache_path(self, split: str, space: str) -> Path:
        cache_dir = resolve(self.cfg.eval_cache_dir, create=True)
        return cache_dir / f"{self.cfg.dataset}_{split}_n{self.cfg.n_eval}_{space}.pt"

    # ---- feature extraction ------------------------------------------------
    def _inception_pass(self, imgs01: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One shared InceptionV3 forward -> (2048-d features, unbiased logits).

        ``_torch_fidelity_forward`` is the only entry point that returns every
        head in ``features_list`` (the public ``forward`` keeps just the first).
        """
        x = _to_uint8(_to_three_channels(imgs01)).to(self.metric_device)
        feats, logits = self.inception._torch_fidelity_forward(x)
        return feats, logits

    def _lenet_pass(self, imgs01: torch.Tensor) -> torch.Tensor:
        """LeNet features from [0, 1] grayscale images (no 3-channel expansion)."""
        assert self.lenet is not None
        return self.lenet.features(imgs01.clamp(0, 1).to(self.metric_device))

    # ---- real side ---------------------------------------------------------
    @torch.no_grad()
    def _real_features(self, train: bool, spaces: list[str]) -> dict[str, torch.Tensor]:
        """One pass over the real split, extracting every requested feature space."""
        loader = build_dataloader(self.cfg, train=train)
        chunks: dict[str, list[torch.Tensor]] = {space: [] for space in spaces}
        seen = 0
        for batch, _ in loader:
            imgs01 = denormalize(batch)
            if "inception" in chunks:
                chunks["inception"].append(self._inception_pass(imgs01)[0])
            if "lenet" in chunks:
                chunks["lenet"].append(self._lenet_pass(imgs01))
            seen += int(imgs01.shape[0])
            if seen >= self.cfg.n_eval:
                break
        if seen == 0:
            raise RuntimeError("real dataloader yielded no batches")
        return {space: torch.cat(c)[: self.cfg.n_eval] for space, c in chunks.items()}

    def _real_stats(self, train: bool) -> dict[str, dict[str, torch.Tensor]]:
        """Load (or compute and disk-cache) real-side stats for every feature space."""
        split = "train" if train else "test"
        stats: dict[str, dict[str, torch.Tensor]] = {}
        missing = []
        for space in self._spaces():
            path = self._cache_path(split, space)
            if path.exists():
                print(f"[eval] using cached real features {path}")
                payload = torch.load(str(path), map_location="cpu", weights_only=True)
                stats[space] = {k: v.to(self.metric_device) for k, v in payload.items()}
            else:
                missing.append(space)
        if missing:
            features = self._real_features(train, missing)
            for space in missing:
                mu, sigma = _mean_cov(features[space])
                payload = {"mu": mu, "sigma": sigma}
                if space == "inception":  # KID resamples subsets from raw features
                    payload["features"] = features[space]
                path = self._cache_path(split, space)
                torch.save({k: v.cpu() for k, v in payload.items()}, str(path))
                stats[space] = payload
        return stats

    # ---- public API ----------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, trainer) -> dict:
        """Score ``cfg.n_eval`` generated samples (drawn in ``cfg.batch_size`` chunks)."""
        inception_chunks, logit_chunks, lenet_chunks = [], [], []
        remaining = self.cfg.n_eval
        while remaining > 0:
            n = min(self.cfg.batch_size, remaining)
            imgs01 = denormalize(trainer.sample(n))
            feats, logits = self._inception_pass(imgs01)
            inception_chunks.append(feats)
            logit_chunks.append(logits)
            if self.lenet is not None:
                lenet_chunks.append(self._lenet_pass(imgs01))
            remaining -= n
        fake = torch.cat(inception_chunks)
        generator = torch.Generator().manual_seed(self.cfg.seed)

        real = self._real["inception"]
        mu_fake, sigma_fake = _mean_cov(fake)
        kid_mean, kid_std = _kid(
            real["features"], fake, self.cfg.kid_subset_size, KID_SUBSETS, generator
        )
        out = {
            "fid": _frechet(real["mu"], real["sigma"], mu_fake, sigma_fake),
            "kid_mean": kid_mean,
            "kid_std": kid_std,
        }
        if self.lenet is not None:
            mu_lenet, sigma_lenet = _mean_cov(torch.cat(lenet_chunks))
            real_lenet = self._real["lenet"]
            out["fid_lenet"] = _frechet(
                real_lenet["mu"], real_lenet["sigma"], mu_lenet, sigma_lenet
            )
        out["is_mean"], out["is_std"] = _inception_score(
            torch.cat(logit_chunks), IS_SPLITS, generator
        )
        return out

    @torch.no_grad()
    def fid_floor(self) -> dict:
        """Train-vs-test FID floors: what a 'perfect' generator would score.

        Finite-sample bias keeps the FID between two disjoint *real* samples
        well above zero; Lucic et al. ("Are GANs Created Equal?", NeurIPS 2018)
        measured train-vs-test floors on the order of ~1.25 Inception-FID at
        comparable sample sizes. Model FIDs should therefore be read relative
        to this floor, not to zero. Uses ``cfg.n_eval`` samples per split; the
        train-split statistics are disk-cached like the test-split ones.
        """
        train_stats = self._real_stats(train=True)
        out = {
            "fid": _frechet(
                train_stats["inception"]["mu"],
                train_stats["inception"]["sigma"],
                self._real["inception"]["mu"],
                self._real["inception"]["sigma"],
            )
        }
        if self.lenet is not None:
            out["fid_lenet"] = _frechet(
                train_stats["lenet"]["mu"],
                train_stats["lenet"]["sigma"],
                self._real["lenet"]["mu"],
                self._real["lenet"]["sigma"],
            )
        return out
