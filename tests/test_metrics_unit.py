"""Fast unit tests for ``otgan.metrics``. The InceptionV3 feature extractor,
the FID-LeNet and the MNIST dataloader are replaced by lightweight recording
stubs, so nothing is downloaded and no network weights are ever loaded."""

import dataclasses
import math

import pytest
import torch

from otgan.metrics import (
    FIDISEvaluator,
    _frechet,
    _inception_score,
    _kid,
    _mean_cov,
    _metric_device,
    _poly_mmd,
    _to_three_channels,
    _to_uint8,
)


class _StubInception:
    """Mimics NoTrainInceptionV3: records calls, emits deterministic features."""

    def __init__(self, name, features_list):
        self.name = name
        self.features_list = list(features_list)
        self.to_devices: list[torch.device] = []
        self.forward_shapes: list[tuple[int, ...]] = []
        self._gen = torch.Generator().manual_seed(99)

    def to(self, device):
        self.to_devices.append(torch.device(device))
        return self

    def _torch_fidelity_forward(self, x):
        assert x.dtype == torch.uint8, "torch-fidelity expects uint8 input"
        self.forward_shapes.append(tuple(x.shape))
        n = int(x.shape[0])
        feats = torch.rand(n, 8, generator=self._gen)
        logits = torch.rand(n, 6, generator=self._gen)
        return feats, logits


class _StubLeNet:
    """Mimics MNISTLeNet.features: asserts the [0,1]-grayscale input contract."""

    def __init__(self, device):
        self.device_arg = torch.device(device)
        self.feature_shapes: list[tuple[int, ...]] = []
        self._gen = torch.Generator().manual_seed(7)

    def features(self, x):
        assert x.shape[1] == 1, "LeNet consumes grayscale images (no 3-channel expansion)"
        assert float(x.min()) >= 0.0 and float(x.max()) <= 1.0, "LeNet consumes [0, 1] images"
        self.feature_shapes.append(tuple(x.shape))
        return torch.rand(int(x.shape[0]), 4, generator=self._gen)


class _StubTrainer:
    def __init__(self, channels: int = 1, image_size: int = 32):
        self.sample_sizes: list[int] = []
        self.channels = channels
        self.image_size = image_size

    def sample(self, n: int) -> torch.Tensor:
        self.sample_sizes.append(n)
        return torch.rand(n, self.channels, self.image_size, self.image_size) * 2.0 - 1.0


@pytest.fixture(autouse=True)
def stub_backends(monkeypatch, tmp_path):
    """Stub the heavy backends; returns the list of _StubLeNet instances built."""
    monkeypatch.setenv("OT_GAN_ROOT", str(tmp_path))
    monkeypatch.setattr("otgan.metrics.NoTrainInceptionV3", _StubInception)
    lenets: list[_StubLeNet] = []

    def fake_train_or_load(cfg, device="cpu"):
        lenets.append(_StubLeNet(device))
        return lenets[-1]

    monkeypatch.setattr("otgan.metrics.train_or_load_lenet", fake_train_or_load)
    return lenets


@pytest.fixture
def install_loader(monkeypatch):
    """Replace build_dataloader (as imported in otgan.metrics) with fixed batches.

    Returns the list of ``train`` flags it was called with.
    """

    def _install(n_batches: int, batch_size: int, channels: int = 1):
        calls: list[bool] = []

        def fake_build_dataloader(cfg, train=True):
            calls.append(train)
            gen = torch.Generator().manual_seed(5)
            return [
                (
                    torch.rand(batch_size, channels, cfg.image_size, cfg.image_size, generator=gen)
                    * 2.0
                    - 1.0,
                    torch.zeros(batch_size, dtype=torch.long),
                )
                for _ in range(n_batches)
            ]

        monkeypatch.setattr("otgan.metrics.build_dataloader", fake_build_dataloader)
        return calls

    return _install


EXPECTED_KEYS = ["fid", "kid_mean", "kid_std", "fid_lenet", "is_mean", "is_std"]


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


class TestImageHelpers:
    def test_expands_single_channel_to_three(self):
        x = torch.rand(2, 1, 4, 4)
        out = _to_three_channels(x)
        assert out.shape == (2, 3, 4, 4)
        for c in range(3):
            assert torch.equal(out[:, c], x[:, 0])

    def test_clamps_to_unit_range(self):
        x = torch.tensor([[[[-0.5, 0.25], [0.75, 1.5]]]])
        out = _to_three_channels(x)
        assert torch.equal(out[0, 0], torch.tensor([[0.0, 0.25], [0.75, 1.0]]))

    def test_three_channel_input_keeps_shape(self):
        x = torch.rand(2, 3, 4, 4) * 2.0 - 0.5
        out = _to_three_channels(x)
        assert out.shape == x.shape
        assert torch.equal(out, x.clamp(0, 1))

    def test_to_uint8_matches_torchmetrics_normalize_path(self):
        x = torch.tensor([[[[0.0, 0.5, 1.0]]]])
        out = _to_uint8(x)
        assert out.dtype == torch.uint8
        assert out.flatten().tolist() == [0, 127, 255]


class TestMetricDevice:
    @pytest.mark.parametrize(
        ("device_type", "expected"),
        [("cpu", "cpu"), ("cuda", "cuda"), ("mps", "cpu")],
    )
    def test_mps_routes_to_cpu_others_keep_device(self, device_type, expected):
        assert _metric_device(torch.device(device_type)) == torch.device(expected)


class TestFrechet:
    def test_identical_gaussians_give_zero(self):
        mu = torch.randn(6, dtype=torch.float64)
        a = torch.randn(6, 6, dtype=torch.float64)
        sigma = a @ a.t() + torch.eye(6, dtype=torch.float64)
        assert abs(_frechet(mu, sigma, mu, sigma)) < 1e-8

    def test_diagonal_closed_form(self):
        # For diagonal covariances: ||mu1-mu2||^2 + sum_i (sqrt(a_i) - sqrt(b_i))^2.
        mu1 = torch.tensor([0.0, 1.0, -2.0], dtype=torch.float64)
        mu2 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        a = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)
        b = torch.tensor([4.0, 1.0, 1.0], dtype=torch.float64)
        expected = float((mu1 - mu2).square().sum() + (a.sqrt() - b.sqrt()).square().sum())
        got = _frechet(mu1, torch.diag(a), mu2, torch.diag(b))
        assert math.isclose(got, expected, rel_tol=1e-9)

    def test_symmetric(self):
        f1, f2 = torch.randn(64, 5), torch.randn(64, 5) + 0.5
        mu1, s1 = _mean_cov(f1)
        mu2, s2 = _mean_cov(f2)
        assert math.isclose(_frechet(mu1, s1, mu2, s2), _frechet(mu2, s2, mu1, s1), rel_tol=1e-6)


class TestMeanCov:
    def test_matches_unbiased_covariance(self):
        f = torch.randn(32, 4)
        mu, sigma = _mean_cov(f)
        assert mu.dtype == torch.float64 and sigma.dtype == torch.float64
        assert torch.allclose(mu, f.double().mean(0))
        assert torch.allclose(sigma, torch.cov(f.double().t()))


class TestKid:
    def test_poly_mmd_matches_bruteforce(self):
        torch.manual_seed(3)
        m, d = 6, 4
        f1, f2 = torch.randn(m, d, dtype=torch.float64), torch.randn(m, d, dtype=torch.float64)

        def k(x, y):
            return (float(x @ y) / d + 1.0) ** 3

        within = sum(
            k(f1[i], f1[j]) + k(f2[i], f2[j]) for i in range(m) for j in range(m) if i != j
        ) / (m * (m - 1))
        cross = sum(k(f1[i], f2[j]) for i in range(m) for j in range(m)) / (m * m)
        assert math.isclose(float(_poly_mmd(f1, f2)), within - 2.0 * cross, rel_tol=1e-9)

    def test_deterministic_given_seeded_generator(self):
        real, fake = torch.randn(20, 4), torch.randn(20, 4)
        first = _kid(real, fake, subset_size=10, subsets=5, generator=_gen(0))
        second = _kid(real, fake, subset_size=10, subsets=5, generator=_gen(0))
        assert first == second

    def test_subset_size_clamped_to_sample_count(self):
        # subset_size=1000 >> 12 samples: every subset is a permutation of the
        # full set, so all scores coincide and the std collapses to 0.
        real, fake = torch.randn(12, 4), torch.randn(12, 4) + 1.0
        mean, std = _kid(real, fake, subset_size=1000, subsets=4, generator=_gen(0))
        assert math.isfinite(mean) and abs(std) < 1e-12
        assert math.isclose(mean, float(_poly_mmd(real, fake)), rel_tol=1e-9)


class TestInceptionScore:
    def test_uniform_logits_score_one(self):
        logits = torch.zeros(40, 10)  # p(y|x) uniform == p(y) -> KL 0 -> IS 1
        mean, std = _inception_score(logits, splits=4, generator=_gen(0))
        assert math.isclose(mean, 1.0, rel_tol=1e-9)
        assert abs(std) < 1e-12

    def test_deterministic_given_seeded_generator(self):
        logits = torch.randn(30, 6)
        assert _inception_score(logits, 5, _gen(1)) == _inception_score(logits, 5, _gen(1))


class TestRealFeatureCache:
    def test_loader_called_once_for_test_split(self, tiny_config, install_loader, cpu):
        # tiny_config: n_eval=16, loader batches of 8 -> only 2 of 5 batches consumed.
        calls = install_loader(n_batches=5, batch_size=8)
        ev = FIDISEvaluator(tiny_config, cpu)
        assert calls == [False]
        assert ev._real["inception"]["features"].shape == (16, 8)

    def test_cache_files_keyed_by_dataset_split_neval_space(
        self, tiny_config, install_loader, cpu, tmp_path
    ):
        install_loader(n_batches=5, batch_size=8)
        FIDISEvaluator(tiny_config, cpu)
        cache = tmp_path / tiny_config.eval_cache_dir
        assert (cache / "mnist_test_n16_inception.pt").exists()
        assert (cache / "mnist_test_n16_lenet.pt").exists()

    def test_cached_payload_contents(self, tiny_config, install_loader, cpu, tmp_path):
        install_loader(n_batches=5, batch_size=8)
        FIDISEvaluator(tiny_config, cpu)
        cache = tmp_path / tiny_config.eval_cache_dir
        inception = torch.load(cache / "mnist_test_n16_inception.pt", weights_only=True)
        # KID needs the raw real features; FID only needs mu/sigma.
        assert set(inception) == {"mu", "sigma", "features"}
        assert inception["mu"].shape == (8,)
        assert inception["sigma"].shape == (8, 8)
        assert inception["features"].shape == (16, 8)
        lenet = torch.load(cache / "mnist_test_n16_lenet.pt", weights_only=True)
        assert set(lenet) == {"mu", "sigma"}
        assert lenet["mu"].shape == (4,)

    def test_second_evaluator_hits_cache(self, tiny_config, install_loader, cpu, capsys):
        calls = install_loader(n_batches=5, batch_size=8)
        FIDISEvaluator(tiny_config, cpu)
        assert calls == [False]
        FIDISEvaluator(tiny_config, cpu)
        assert calls == [False]  # no second real pass
        assert "[eval] using cached real features" in capsys.readouterr().out

    def test_changing_n_eval_invalidates_cache(self, tiny_config, install_loader, cpu, tmp_path):
        calls = install_loader(n_batches=5, batch_size=8)
        FIDISEvaluator(tiny_config, cpu)
        FIDISEvaluator(dataclasses.replace(tiny_config, n_eval=8), cpu)
        assert calls == [False, False]  # different key -> recompute
        cache = tmp_path / tiny_config.eval_cache_dir
        assert (cache / "mnist_test_n16_inception.pt").exists()
        assert (cache / "mnist_test_n8_inception.pt").exists()

    def test_evaluate_never_recomputes_real_features(self, tiny_config, install_loader, cpu):
        calls = install_loader(n_batches=5, batch_size=8)
        ev = FIDISEvaluator(tiny_config, cpu)
        ev.evaluate(_StubTrainer())
        ev.evaluate(_StubTrainer())
        assert calls == [False]

    def test_real_images_expanded_for_inception_only(self, tiny_config, install_loader, cpu):
        install_loader(n_batches=5, batch_size=8, channels=1)
        ev = FIDISEvaluator(tiny_config, cpu)
        assert all(shape[1] == 3 for shape in ev.inception.forward_shapes)
        assert ev.lenet is not None
        assert all(shape[1] == 1 for shape in ev.lenet.feature_shapes)


class TestEvaluate:
    def test_output_keys_and_types(self, tiny_config, install_loader, cpu):
        install_loader(n_batches=5, batch_size=8)
        ev = FIDISEvaluator(tiny_config, cpu)
        result = ev.evaluate(_StubTrainer())
        assert list(result) == EXPECTED_KEYS
        assert all(isinstance(v, float) and math.isfinite(v) for v in result.values())

    def test_generated_side_batches_by_batch_size(self, tiny_config, install_loader, cpu):
        install_loader(n_batches=5, batch_size=8)
        cfg = dataclasses.replace(tiny_config, n_eval=20)  # batch_size=8 -> 8 + 8 + 4
        ev = FIDISEvaluator(cfg, cpu)
        trainer = _StubTrainer()
        ev.evaluate(trainer)
        assert trainer.sample_sizes == [8, 8, 4]

    def test_non_mnist_skips_fid_lenet(self, tiny_config, install_loader, cpu, stub_backends):
        install_loader(n_batches=5, batch_size=8, channels=3)
        cfg = dataclasses.replace(tiny_config, dataset="cifar10", channels=3)
        ev = FIDISEvaluator(cfg, cpu)
        assert ev.lenet is None
        assert stub_backends == []  # train_or_load_lenet never called
        result = ev.evaluate(_StubTrainer(channels=3))
        assert list(result) == ["fid", "kid_mean", "kid_std", "is_mean", "is_std"]


class TestDeviceRouting:
    @pytest.mark.parametrize(("device_type", "expected"), [("cpu", "cpu"), ("mps", "cpu")])
    def test_construction_routes_metrics(
        self, tiny_config, install_loader, stub_backends, device_type, expected
    ):
        install_loader(n_batches=5, batch_size=8)
        ev = FIDISEvaluator(tiny_config, torch.device(device_type))
        assert ev.metric_device == torch.device(expected)
        assert ev.inception.to_devices == [torch.device(expected)]
        assert stub_backends[-1].device_arg == torch.device(expected)
        assert ev.device == torch.device(device_type)


class TestFidFloor:
    def test_floor_keys_and_train_split_caching(self, tiny_config, install_loader, cpu, tmp_path):
        calls = install_loader(n_batches=5, batch_size=8)
        ev = FIDISEvaluator(tiny_config, cpu)
        floor = ev.fid_floor()
        assert list(floor) == ["fid", "fid_lenet"]
        assert all(isinstance(v, float) and math.isfinite(v) for v in floor.values())
        assert calls == [False, True]  # one extra pass over the train split
        cache = tmp_path / tiny_config.eval_cache_dir
        assert (cache / "mnist_train_n16_inception.pt").exists()
        # A second floor call reuses the disk cache: no further loader calls.
        ev.fid_floor()
        assert calls == [False, True]
