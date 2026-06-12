"""FID-LeNet: a small MNIST classifier whose penultimate layer provides a
domain-relevant feature space for Frechet distances.

FID (Heusel et al., NeurIPS 2017) embeds images with an ImageNet-trained
InceptionV3, a feature space only weakly aligned with 32x32 grayscale digits;
Binkowski et al. (ICLR 2018) discuss how the choice of feature space shapes
what these metrics actually measure. The standard remedy on MNIST is to train
a small LeNet-style classifier on MNIST itself and compute the Frechet
distance in its penultimate features — here a 128-d space ("FID-LeNet"),
consumed by ``otgan.metrics.FIDISEvaluator``.

Input convention (deliberately different from the Inception path): the network
consumes single-channel 32x32 images in ``[0, 1]`` — i.e. *after*
``otgan.trainer.denormalize`` — with NO 3-channel expansion and NO uint8
conversion. The training transform below therefore stops at ``ToTensor()``
instead of reusing ``otgan.data.build_transform`` (which maps to ``[-1, 1]``).

Training is deliberately cheap and deterministic: 2 epochs of Adam (lr 1e-3,
batch 256) on the MNIST train split, CPU only, seeded with
``seed_everything(0)``. The resulting ``state_dict`` is cached at
``resolve(cfg.eval_cache_dir) / 'lenet_mnist.pt'`` so it is trained at most
once per machine; global RNG state is snapshotted and restored around training
so a cache miss in the middle of a GAN run does not perturb that run's random
stream relative to a cache-hit run.
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from otgan.device import seed_everything
from otgan.paths import resolve

LENET_FILENAME = "lenet_mnist.pt"


class MNISTLeNet(nn.Module):
    """conv(1->32, k3) ReLU pool -> conv(32->64, k3) ReLU pool -> fc 4096->128 -> fc 128->10.

    Expects 1 x 32 x 32 inputs in ``[0, 1]``; ``features`` returns the 128-d
    penultimate activations used for FID-LeNet.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """128-d penultimate features for ``x`` in [0, 1], shape (N, 1, 32, 32)."""
        x = self.pool(F.relu(self.conv1(x)))  # N x 32 x 16 x 16
        x = self.pool(F.relu(self.conv2(x)))  # N x 64 x 8 x 8
        x = torch.flatten(x, start_dim=1)  # N x 4096
        return F.relu(self.fc1(x))  # N x 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.features(x))


def _train_lenet(cfg) -> MNISTLeNet:
    """Train the LeNet for 2 epochs on the MNIST train split (CPU, deterministic)."""
    # Snapshot global RNG: seed_everything(0) below must not change the
    # surrounding run's random stream (a cache miss can happen mid-training).
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    try:
        seed_everything(0)
        transform = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.ToTensor(),  # -> [0, 1]: the LeNet input convention
            ]
        )
        root = str(resolve(cfg.data_root, create=True))
        dataset = datasets.MNIST(root, train=True, download=True, transform=transform)
        loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)

        model = MNISTLeNet().to("cpu")
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for epoch in range(2):
            correct, total = 0, 0
            for imgs, labels in loader:
                logits = model(imgs)
                loss = F.cross_entropy(logits, labels)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                correct += int((logits.argmax(dim=1) == labels).sum())
                total += int(labels.shape[0])
            print(f"[lenet] epoch {epoch}: train acc {correct / max(total, 1):.4f}")
        return model
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)


def train_or_load_lenet(cfg, device: str | torch.device = "cpu") -> MNISTLeNet:
    """Return an eval-mode ``MNISTLeNet``, training and caching it on first use.

    The state_dict is cached at ``resolve(cfg.eval_cache_dir)/'lenet_mnist.pt'``;
    when the file exists it is loaded instead of retraining, which also makes
    the feature space stable across evaluations. Training always runs on CPU;
    the returned model is moved to ``device``.
    """
    path = resolve(cfg.eval_cache_dir, create=True) / LENET_FILENAME
    if path.exists():
        print(f"[eval] using cached FID-LeNet weights {path}")
        model = MNISTLeNet()
        model.load_state_dict(torch.load(str(path), map_location="cpu", weights_only=True))
    else:
        model = _train_lenet(cfg)
        torch.save(model.state_dict(), str(path))
        print(f"[eval] trained FID-LeNet and cached weights at {path}")
    model = model.to(torch.device(device))
    model.eval()
    return model
