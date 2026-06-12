"""Tests for the data pipeline (offline: torchvision datasets are stubbed, no downloads)."""

import dataclasses

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils import data as torch_data
from torch.utils.data import RandomSampler, SequentialSampler

import otgan.data as otdata
from otgan.data import build_dataloader, build_transform, split_real_pair


# ---- split_real_pair ----------------------------------------------------
def test_split_real_pair_exact_disjoint_halves():
    batch = torch.arange(16, dtype=torch.float32).view(16, 1, 1, 1)
    x, x_prime = split_real_pair(batch)
    assert torch.equal(x, batch[:8])
    assert torch.equal(x_prime, batch[8:16])
    # Disjoint: no element of X appears in X' (values are unique by construction).
    assert not set(x.flatten().tolist()) & set(x_prime.flatten().tolist())


def test_split_real_pair_odd_size_truncates_last():
    batch = torch.arange(7, dtype=torch.float32).view(7, 1)
    x, x_prime = split_real_pair(batch)
    assert x.shape[0] == x_prime.shape[0] == 3
    assert torch.equal(x, batch[:3])
    assert torch.equal(x_prime, batch[3:6])  # element 6 is dropped


# ---- build_transform ----------------------------------------------------
def _gradient_image(size: int = 64, channels: int = 1) -> Image.Image:
    """A uint8 gradient spanning the full [0, 255] range."""
    row = np.linspace(0, 255, size, dtype=np.uint8)
    plane = np.tile(row, (size, 1))
    if channels == 3:
        return Image.fromarray(np.stack([plane] * 3, axis=-1), mode="RGB")
    return Image.fromarray(plane, mode="L")


def test_build_transform_grayscale_shape_and_range(tiny_config):
    out = build_transform(tiny_config)(_gradient_image(channels=1))
    assert out.shape == (1, tiny_config.image_size, tiny_config.image_size)
    assert out.min().item() >= -1.0 and out.max().item() <= 1.0
    # The full input range must map onto (nearly) the full [-1, 1] output range.
    assert out.min().item() < -0.9 and out.max().item() > 0.9


def test_build_transform_rgb_shape_and_range(tiny_config):
    cfg = dataclasses.replace(tiny_config, dataset="cifar10", channels=3)
    out = build_transform(cfg)(_gradient_image(channels=3))
    assert out.shape == (3, cfg.image_size, cfg.image_size)
    assert out.min().item() >= -1.0 and out.max().item() <= 1.0


# ---- build_dataloader ---------------------------------------------------
def test_build_dataloader_unknown_dataset_raises(tiny_config):
    tiny_config.dataset = "svhn"  # bypass Config validation to hit the loader's own check
    with pytest.raises(ValueError, match="Unknown dataset"):
        build_dataloader(tiny_config)


def _stub_dataset_cls(created: list):
    """In-memory stand-in for a torchvision dataset (same constructor signature)."""

    class _StubDataset(torch_data.Dataset):
        def __init__(self, root, train=True, download=False, transform=None):
            self.root = root
            self.train = train
            self.download = download
            self.transform = transform
            created.append(self)

        def __len__(self):
            return 20  # not a multiple of 2 * batch_size, so drop_last matters

        def __getitem__(self, index):
            img = Image.fromarray(np.full((32, 32), index, dtype=np.uint8), mode="L")
            if self.transform is not None:
                img = self.transform(img)
            return img, index % 10

    return _StubDataset


def test_build_dataloader_wiring_train(tiny_config, tmp_path, monkeypatch):
    monkeypatch.setenv("OT_GAN_ROOT", str(tmp_path))
    created: list = []
    monkeypatch.setattr(otdata, "_DATASETS", {"mnist": _stub_dataset_cls(created)})

    loader = build_dataloader(tiny_config, train=True)

    # Dataset constructed once, rooted under OT_GAN_ROOT, in train mode.
    (dataset,) = created
    assert dataset.root == str(tmp_path / tiny_config.data_root)
    assert (tmp_path / tiny_config.data_root).is_dir()  # resolve(create=True)
    assert dataset.train is True
    assert dataset.download is True
    assert dataset.transform is not None

    # Loader yields double batches with the tail dropped, shuffled for training.
    assert loader.batch_size == 2 * tiny_config.batch_size
    assert loader.drop_last is True
    assert isinstance(loader.sampler, RandomSampler)
    assert len(loader) == 1  # 20 // 16 with drop_last

    images, labels = next(iter(loader))
    assert images.shape == (16, 1, tiny_config.image_size, tiny_config.image_size)
    assert labels.shape == (16,)


def test_build_dataloader_wiring_eval_is_sequential(tiny_config, tmp_path, monkeypatch):
    monkeypatch.setenv("OT_GAN_ROOT", str(tmp_path))
    created: list = []
    monkeypatch.setattr(otdata, "_DATASETS", {"mnist": _stub_dataset_cls(created)})

    loader = build_dataloader(tiny_config, train=False)

    assert created[0].train is False
    assert isinstance(loader.sampler, SequentialSampler)
