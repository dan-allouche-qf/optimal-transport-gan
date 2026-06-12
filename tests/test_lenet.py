"""Tests for the FID-LeNet feature network (``otgan.lenet``). Everything that
needs the real MNIST dataset is marked slow; the fast tests run offline by
stubbing the dataset (or the whole training loop)."""

import pytest
import torch
from torch.utils.data import TensorDataset

import otgan.lenet as lenet_mod
from otgan.lenet import LENET_FILENAME, MNISTLeNet, train_or_load_lenet


class TestArchitecture:
    def test_features_shape_is_n_by_128(self):
        net = MNISTLeNet()
        x = torch.rand(5, 1, 32, 32)  # [0, 1] grayscale: the documented input contract
        assert net.features(x).shape == (5, 128)

    def test_forward_emits_10_logits(self):
        net = MNISTLeNet()
        assert net(torch.rand(3, 1, 32, 32)).shape == (3, 10)

    def test_rejects_three_channel_input(self):
        # LeNet consumes grayscale directly — no 3-channel expansion like Inception.
        with pytest.raises(RuntimeError):
            MNISTLeNet().features(torch.rand(2, 3, 32, 32))


def _fresh_net(seed: int = 7) -> MNISTLeNet:
    torch.manual_seed(seed)
    return MNISTLeNet()


class TestTrainOrLoad:
    def test_trains_once_then_loads_from_cache(self, tiny_config, monkeypatch, tmp_path):
        monkeypatch.setenv("OT_GAN_ROOT", str(tmp_path))
        calls: list[int] = []

        def fake_train(cfg):
            calls.append(1)
            if len(calls) > 1:
                pytest.fail("training ran twice despite a warm cache")
            return _fresh_net()

        monkeypatch.setattr("otgan.lenet._train_lenet", fake_train)

        first = train_or_load_lenet(tiny_config)
        assert (tmp_path / tiny_config.eval_cache_dir / LENET_FILENAME).exists()
        second = train_or_load_lenet(tiny_config)  # must load, not retrain
        assert calls == [1]
        for key, value in first.state_dict().items():
            assert torch.equal(value, second.state_dict()[key])

    def test_returns_eval_mode_on_requested_device(self, tiny_config, monkeypatch, tmp_path):
        monkeypatch.setenv("OT_GAN_ROOT", str(tmp_path))
        monkeypatch.setattr("otgan.lenet._train_lenet", lambda cfg: _fresh_net())
        model = train_or_load_lenet(tiny_config, device="cpu")
        assert not model.training
        assert next(model.parameters()).device == torch.device("cpu")

    def test_cache_hit_is_logged(self, tiny_config, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("OT_GAN_ROOT", str(tmp_path))
        monkeypatch.setattr("otgan.lenet._train_lenet", lambda cfg: _fresh_net())
        train_or_load_lenet(tiny_config)
        train_or_load_lenet(tiny_config)
        assert "[eval] using cached FID-LeNet weights" in capsys.readouterr().out


def _install_fake_mnist(monkeypatch, n: int = 64):
    """Replace torchvision's MNIST with a tiny in-memory dataset (offline)."""

    def fake_mnist(root, train=True, download=False, transform=None):
        gen = torch.Generator().manual_seed(0)
        imgs = torch.rand(n, 1, 32, 32, generator=gen)
        labels = torch.randint(0, 10, (n,), generator=gen)
        return TensorDataset(imgs, labels)

    monkeypatch.setattr(lenet_mod.datasets, "MNIST", fake_mnist)


class TestTrainingLoopOffline:
    def test_training_runs_and_restores_global_rng(self, tiny_config, monkeypatch, tmp_path):
        """The real 2-epoch loop on a stubbed MNIST: trains, caches, and leaves
        the surrounding run's RNG stream untouched (seed_everything(0) inside)."""
        monkeypatch.setenv("OT_GAN_ROOT", str(tmp_path))
        _install_fake_mnist(monkeypatch)
        torch.manual_seed(1234)
        state_before = torch.get_rng_state()
        model = train_or_load_lenet(tiny_config)
        assert torch.equal(torch.get_rng_state(), state_before)
        assert not model.training
        assert (tmp_path / tiny_config.eval_cache_dir / LENET_FILENAME).exists()

    def test_training_is_deterministic(self, tiny_config, monkeypatch, tmp_path):
        monkeypatch.setenv("OT_GAN_ROOT", str(tmp_path))
        _install_fake_mnist(monkeypatch)
        first = train_or_load_lenet(tiny_config).state_dict()
        (tmp_path / tiny_config.eval_cache_dir / LENET_FILENAME).unlink()
        second = train_or_load_lenet(tiny_config).state_dict()
        for key, value in first.items():
            assert torch.equal(value, second[key]), f"nondeterministic parameter {key}"


@pytest.mark.slow
def test_real_mnist_training_reaches_reasonable_accuracy(tmp_path):
    """2 CPU epochs on the real MNIST train split should classify well above
    chance on held-out test digits. Slow: touches the real dataset."""
    from pathlib import Path

    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    from otgan.config import Config

    repo_data = Path(__file__).resolve().parent.parent / "data"
    cfg = Config(
        dataset="mnist",
        data_root=str(repo_data),  # absolute: reuses the repo's MNIST download
        eval_cache_dir=str(tmp_path / "eval_cache"),
        num_workers=0,
        device="cpu",
    )
    model = train_or_load_lenet(cfg)

    transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
    test_set = datasets.MNIST(str(repo_data), train=False, download=True, transform=transform)
    loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            correct += int((model(imgs).argmax(dim=1) == labels).sum())
            total += int(labels.shape[0])
            if total >= 2048:
                break
    assert correct / total > 0.9, f"FID-LeNet test accuracy too low: {correct / total:.3f}"
