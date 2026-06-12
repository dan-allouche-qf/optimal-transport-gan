"""Sanity and cross-check tests for FID/KID/IS. Marked slow: most of them build
the real InceptionV3 (weights ~100MB, downloaded on first run) and are skipped
if that download fails (offline CI)."""

import math

import pytest
import torch

pytestmark = pytest.mark.slow


def _fid():
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance

        return FrechetInceptionDistance(feature=2048, normalize=True)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"InceptionV3 unavailable: {exc}")


def _kid_metric(subset_size: int, subsets: int):
    try:
        from torchmetrics.image.kid import KernelInceptionDistance

        return KernelInceptionDistance(subset_size=subset_size, subsets=subsets, normalize=True)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"InceptionV3 unavailable: {exc}")


def test_fid_identical_is_near_zero():
    torch.manual_seed(0)
    imgs = torch.rand(32, 3, 32, 32)
    fid = _fid()
    fid.update(imgs, real=True)
    fid.update(imgs, real=False)
    assert float(fid.compute()) < 1.0


def test_fid_increases_with_noise():
    torch.manual_seed(0)
    real = torch.rand(48, 3, 32, 32)
    noisy = (real + 0.5 * torch.rand_like(real)).clamp(0, 1)

    fid_clean = _fid()
    fid_clean.update(real, real=True)
    fid_clean.update(real.clone(), real=False)

    fid_noisy = _fid()
    fid_noisy.update(real, real=True)
    fid_noisy.update(noisy, real=False)

    assert float(fid_noisy.compute()) > float(fid_clean.compute())


def test_inception_score_runs():
    try:
        from torchmetrics.image.inception import InceptionScore
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"InceptionV3 unavailable: {exc}")
    torch.manual_seed(0)
    metric = InceptionScore(normalize=True)
    metric.update(torch.rand(32, 3, 32, 32))
    mean, _std = metric.compute()
    assert torch.isfinite(mean) and float(mean) >= 1.0


# ---- cross-checks of otgan.metrics' direct estimators vs torchmetrics --------


def test_frechet_matches_torchmetrics_compute_fid():
    """Pure math: our float64 _frechet vs torchmetrics' _compute_fid."""
    from torchmetrics.image.fid import _compute_fid

    from otgan.metrics import _frechet, _mean_cov

    torch.manual_seed(0)
    f1 = torch.randn(256, 32)
    f2 = torch.randn(256, 32) * 1.3 + 0.5
    mu1, s1 = _mean_cov(f1)
    mu2, s2 = _mean_cov(f2)
    expected = float(_compute_fid(mu1, s1, mu2, s2))
    assert math.isclose(_frechet(mu1, s1, mu2, s2), expected, rel_tol=1e-6, abs_tol=1e-6)


def test_poly_mmd_matches_torchmetrics():
    """Pure math: our float64 _poly_mmd vs torchmetrics' poly_mmd."""
    from torchmetrics.image.kid import poly_mmd

    from otgan.metrics import _poly_mmd

    torch.manual_seed(0)
    f1 = torch.randn(64, 16)
    f2 = torch.randn(64, 16) + 0.3
    expected = float(poly_mmd(f1, f2))
    assert math.isclose(float(_poly_mmd(f1, f2)), expected, rel_tol=1e-4, abs_tol=1e-6)


def test_fid_pipeline_matches_vanilla_torchmetrics():
    """Same images through vanilla FrechetInceptionDistance and through our
    uint8 conversion + feature pass + _frechet (sharing the same backbone)."""
    from otgan.metrics import _frechet, _mean_cov, _to_uint8

    fid = _fid()
    torch.manual_seed(0)
    real = torch.rand(64, 3, 32, 32)
    fake = (real + 0.3 * torch.rand_like(real)).clamp(0, 1)
    fid.update(real, real=True)
    fid.update(fake, real=False)
    expected = float(fid.compute())

    with torch.no_grad():
        feats_real = fid.inception(_to_uint8(real))
        feats_fake = fid.inception(_to_uint8(fake))
    mu_r, s_r = _mean_cov(feats_real)
    mu_f, s_f = _mean_cov(feats_fake)
    got = _frechet(mu_r, s_r, mu_f, s_f)
    assert math.isclose(got, expected, rel_tol=1e-3, abs_tol=1e-3), f"{got} vs {expected}"


def test_kid_pipeline_matches_vanilla_torchmetrics():
    """With subset_size == n_samples the subset draws are permutation-invariant,
    so vanilla KernelInceptionDistance and our _kid must agree exactly
    (independent of which RNG drew the subsets)."""
    from otgan.metrics import _kid, _to_uint8

    n = 64
    kid = _kid_metric(subset_size=n, subsets=8)
    torch.manual_seed(0)
    real = torch.rand(n, 3, 32, 32)
    fake = (real + 0.3 * torch.rand_like(real)).clamp(0, 1)
    kid.update(real, real=True)
    kid.update(fake, real=False)
    expected_mean, expected_std = (float(t) for t in kid.compute())

    with torch.no_grad():
        feats_real = kid.inception(_to_uint8(real))
        feats_fake = kid.inception(_to_uint8(fake))
    generator = torch.Generator().manual_seed(0)
    got_mean, got_std = _kid(feats_real, feats_fake, subset_size=n, subsets=8, generator=generator)
    assert math.isclose(got_mean, expected_mean, rel_tol=1e-4, abs_tol=1e-6)
    assert abs(got_std) < 1e-6 and abs(expected_std) < 1e-6


def test_evaluator_end_to_end_real_backbone(monkeypatch, tmp_path, tiny_config):
    """FIDISEvaluator with the real InceptionV3 (LeNet untrained, loader stubbed):
    the shared dual-head pass works and every metric key comes back finite."""
    import dataclasses

    from otgan.lenet import MNISTLeNet
    from otgan.metrics import FIDISEvaluator

    monkeypatch.setenv("OT_GAN_ROOT", str(tmp_path))
    cfg = dataclasses.replace(tiny_config, n_eval=32, batch_size=16)

    def fake_build_dataloader(config, train=True):
        gen = torch.Generator().manual_seed(5)
        return [
            (torch.rand(16, 1, 32, 32, generator=gen) * 2.0 - 1.0, torch.zeros(16).long())
            for _ in range(3)
        ]

    monkeypatch.setattr("otgan.metrics.build_dataloader", fake_build_dataloader)
    monkeypatch.setattr(
        "otgan.metrics.train_or_load_lenet", lambda config, device="cpu": MNISTLeNet().eval()
    )

    class _Trainer:
        def sample(self, n):
            return torch.rand(n, 1, 32, 32) * 2.0 - 1.0

    try:
        evaluator = FIDISEvaluator(cfg, torch.device("cpu"))
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"InceptionV3 unavailable: {exc}")
    result = evaluator.evaluate(_Trainer())
    assert list(result) == ["fid", "kid_mean", "kid_std", "fid_lenet", "is_mean", "is_std"]
    assert all(math.isfinite(v) for v in result.values())
    assert result["fid"] > 0.0
    floor = evaluator.fid_floor()
    assert set(floor) == {"fid", "fid_lenet"}
    assert all(math.isfinite(v) for v in floor.values())
