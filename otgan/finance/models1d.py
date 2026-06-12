"""1D generator and critic for log-return paths — a deliberate structural rhyme
with ``otgan/models.py``.

The point of the finance track is that the OT engine is unchanged, so the
architectures translate the image pair (Salimans et al. 2018) dimension by
dimension: the generator is the same GLU pyramid (linear+GLU into a coarse
feature map, two nearest-neighbor upsamples with conv+GLU), the critic the
same CReLU stack (Shang et al. 2016) ending in an L2-normalized embedding so
the cosine cost in ``otgan.sinkhorn`` stays valid in [0, 2]. Two adaptations
are domain-driven rather than cosmetic:

1. **Linear output head (no tanh).** Log-returns are unbounded, unlike pixels
   in [-1, 1]. The trainer standardizes the target to unit variance and stores
   the scale in the checkpoint, so the generator works in standardized space
   and its head must remain affine.

2. **Dilated convolutions in the critic** (a la WaveNet, van den Oord et al.
   2016). With kernel size 5 and dilations 1, 2, 4 (stride 2 between blocks),
   the embedding's receptive field is 5 -> 13 -> 45 lags — comfortably more
   than 25, so the critic can see volatility clustering (Cont 2001), the
   defining long-memory stylized fact of returns, not just the marginal
   distribution of single steps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReturnsGenerator(nn.Module):
    """z -> linear+GLU -> 128 x (T/4) -> (upsample, conv, GLU) x2 -> conv -> (B, 1, T).

    Widths 128 -> 64 -> 32; the final ``Conv1d(32, 1)`` head is LINEAR (see the
    module docstring: log-returns are unbounded, the data is standardized).
    """

    def __init__(self, z_dim: int = 64, seq_len: int = 64, kernel_size: int = 5):
        super().__init__()
        if seq_len % 4 != 0:
            raise ValueError(f"seq_len must be divisible by 4 (two x2 upsamples), got {seq_len}")
        pad = (kernel_size - 1) // 2
        self.seq_len = seq_len
        self.base_len = seq_len // 4
        self.linear = nn.Linear(z_dim, 2 * 128 * self.base_len)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = nn.Conv1d(128, 2 * 64, kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(64, 2 * 32, kernel_size, padding=pad)
        self.head = nn.Conv1d(32, 1, kernel_size, padding=pad)  # linear: no tanh

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = F.glu(self.linear(z), dim=1)  # B x (128 * T/4)
        x = x.view(x.shape[0], 128, self.base_len)  # B x 128 x T/4
        x = F.glu(self.conv1(self.upsample(x)), dim=1)  # B x 64 x T/2
        x = F.glu(self.conv2(self.upsample(x)), dim=1)  # B x 32 x T
        return self.head(x)  # B x 1 x T, unbounded


class ReturnsCritic(nn.Module):
    """Path -> (dilated conv, CReLU) x3 -> flatten -> L2-normalized embedding.

    Conv widths 32/64/128 with stride 2 between blocks and dilations 1, 2, 4
    (kernel 5): receptive field 45 lags > 25, enough to resolve volatility
    clustering. CReLU doubles channels for free, exactly like the image critic,
    and the row-normalized output makes ``otgan.sinkhorn.cost`` a true cosine
    distance.
    """

    def __init__(self, seq_len: int = 64, kernel_size: int = 5):
        super().__init__()
        if seq_len % 4 != 0:
            raise ValueError(f"seq_len must be divisible by 4 (two stride-2 blocks), got {seq_len}")
        k = kernel_size

        def pad(dilation: int) -> int:
            return dilation * (k - 1) // 2  # 'same'-style padding under dilation

        self.conv1 = nn.Conv1d(1, 32, k, dilation=1, padding=pad(1))
        self.conv2 = nn.Conv1d(64, 64, k, stride=2, dilation=2, padding=pad(2))
        self.conv3 = nn.Conv1d(128, 128, k, stride=2, dilation=4, padding=pad(4))
        self.embed_dim = 256 * (seq_len // 4)

    @staticmethod
    def _crelu(x: torch.Tensor) -> torch.Tensor:
        # Concatenated ReLU: doubles capacity without extra parameters.
        return torch.cat((F.relu(x), F.relu(-x)), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._crelu(self.conv1(x))  # B x 64 x T
        x = self._crelu(self.conv2(x))  # B x 128 x T/2
        x = self._crelu(self.conv3(x))  # B x 256 x T/4
        x = torch.flatten(x, start_dim=1)  # B x 256*(T/4)
        x = F.normalize(x, dim=1, p=2)  # unit L2 norm per row
        return x


def build_finance_models(cfg) -> tuple[ReturnsGenerator, ReturnsCritic]:
    """Construct (generator, critic) from a FinanceConfig."""
    gen = ReturnsGenerator(z_dim=cfg.z_dim, seq_len=cfg.seq_len)
    critic = ReturnsCritic(seq_len=cfg.seq_len)
    return gen, critic
