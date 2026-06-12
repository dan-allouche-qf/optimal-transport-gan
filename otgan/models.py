"""Generator and critic architectures from Salimans et al. (2018).

Kept faithful to the original notebook: a GLU generator and a CReLU critic that
emits an L2-normalized embedding (so cosine distance lives in [0, 2]). The only
change is that the number of image channels is parametrized (1 for MNIST, 3 for
CIFAR-10) and the manual ``split + sigmoid`` GLU is expressed with ``F.glu``,
which is numerically identical and avoids in-place autograd pitfalls.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OTGANGenerator(nn.Module):
    """z -> linear+GLU -> 512x8x8 -> (upsample, conv, GLU) x2 -> conv -> Tanh."""

    def __init__(self, z_dim: int = 100, channels: int = 1, kernel_size: int = 5):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.linear = nn.Linear(z_dim, 2 * 512 * 8 * 8)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = nn.Conv2d(512, 2 * 256, kernel_size, padding=pad)
        self.conv2 = nn.Conv2d(256, 2 * 128, kernel_size, padding=pad)
        self.conv3 = nn.Conv2d(128, channels, kernel_size, padding=pad)
        self.activ_out = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.glu(self.linear(x), dim=1)  # B x 32768
        x = x.view(x.shape[0], 512, 8, 8)  # B x 512 x 8 x 8
        x = F.glu(self.conv1(self.upsample(x)), 1)  # B x 256 x 16 x 16
        x = F.glu(self.conv2(self.upsample(x)), 1)  # B x 128 x 32 x 32
        x = self.activ_out(self.conv3(x))  # B x C x 32 x 32, in [-1, 1]
        return x


class OTGANCritic(nn.Module):
    """Image -> (conv, CReLU) x3 -> flatten -> L2-normalized 32768-d embedding."""

    def __init__(self, channels: int = 1, kernel_size: int = 5):
        super().__init__()
        c = 64
        pad = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(channels, c, kernel_size, padding=pad)
        self.conv2 = nn.Conv2d(c * 2, c * 2, kernel_size, stride=2, padding=pad)
        self.conv3 = nn.Conv2d(c * 4, c * 4, kernel_size, stride=2, padding=pad)

    @staticmethod
    def _crelu(x: torch.Tensor) -> torch.Tensor:
        # Concatenated ReLU: doubles capacity without extra parameters.
        return torch.cat((F.relu(x), F.relu(-x)), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._crelu(self.conv1(x))  # B x 128 x 32 x 32
        x = self._crelu(self.conv2(x))  # B x 256 x 16 x 16
        x = self._crelu(self.conv3(x))  # B x 512 x 8 x 8
        x = torch.flatten(x, start_dim=1)  # B x 32768
        x = F.normalize(x, dim=1, p=2)  # unit L2 norm per row
        return x


def build_models(cfg):
    """Construct (generator, critic) from a Config."""
    gen = OTGANGenerator(z_dim=cfg.z_dim, channels=cfg.channels)
    critic = OTGANCritic(channels=cfg.channels)
    return gen, critic
