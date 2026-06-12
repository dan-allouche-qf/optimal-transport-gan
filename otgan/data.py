"""Data pipeline for MNIST / CIFAR-10.

Two corrections over the original notebook:

1. **Range match.** Images are normalized to ``[-1, 1]`` (``Normalize(0.5, 0.5)``)
   so they share the generator's ``Tanh`` output range. Without this the critic
   could trivially separate real from fake on global brightness.

2. **Independent real minibatches.** The loader yields ``2 * batch_size`` images
   per step; the trainer splits them into two *independent* halves ``X`` and ``X'``.
   The original ``DoubleBatchDataset(batch, batch)`` made ``X == X'``, collapsing
   the ``E[W(X, X')]`` variance-reduction term to ~0.
"""

import torch
from torch.utils import data
from torchvision import datasets, transforms

from otgan.paths import resolve

_DATASETS = {"mnist": datasets.MNIST, "cifar10": datasets.CIFAR10}


def build_transform(cfg) -> transforms.Compose:
    mean = (0.5,) * cfg.channels
    std = (0.5,) * cfg.channels
    return transforms.Compose(
        [
            transforms.Resize(cfg.image_size),
            transforms.ToTensor(),  # -> [0, 1]
            transforms.Normalize(mean, std),  # -> [-1, 1]
        ]
    )


def build_dataloader(cfg, train: bool = True) -> data.DataLoader:
    if cfg.dataset not in _DATASETS:
        raise ValueError(f"Unknown dataset {cfg.dataset!r}; expected one of {list(_DATASETS)}")
    root = str(resolve(cfg.data_root, create=True))
    dataset = _DATASETS[cfg.dataset](
        root, train=train, download=True, transform=build_transform(cfg)
    )
    return data.DataLoader(
        dataset,
        batch_size=2 * cfg.batch_size,  # split into two independent halves per step
        shuffle=train,
        drop_last=True,  # keep every cost matrix exactly B x B
        num_workers=cfg.num_workers,
    )


def split_real_pair(batch: torch.Tensor):
    """Split a ``2B`` batch into two independent real minibatches ``(X, X')``."""
    half = batch.shape[0] // 2
    return batch[:half], batch[half : 2 * half]
