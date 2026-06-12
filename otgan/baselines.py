"""DCGAN baseline (Radford et al., 2016) on the shared evaluation harness.

This is a calibration baseline so the FID harness has a same-evaluator
reference point (Lucic et al. 2018 report WGAN ~6.7, WGAN-GP ~20.3 on MNIST
with large budgets). Published FID numbers are not comparable across feature
extractors and sample counts, so the honest way to situate the OT-GAN is to
train a well-understood baseline through the *exact same* ``BaseTrainer``
pipeline — same data range, same EMA sampling, same FID/KID evaluator — and
compare against that.

The implementation is deliberately textbook: the standard 32x32 DCGAN
transposed-conv generator / strided-conv discriminator with BatchNorm and
LeakyReLU(0.2), trained with the non-saturating GAN loss (Goodfellow et al.,
2014) via ``BCEWithLogitsLoss``. No tricks beyond those architectural
guidelines, so the number it produces reflects the baseline, not tuning.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from otgan.ema import EMAGenerator
from otgan.trainer import BaseTrainer


class DCGANGenerator(nn.Module):
    """``z -> 1x1 -> 4x4(256) -> 8x8(128) -> 16x16(64) -> 32x32(C)``, Tanh output.

    Transposed-conv layers follow the DCGAN guidelines: BatchNorm + ReLU on every
    block except the output, which maps straight to ``Tanh`` so samples share the
    data range ``[-1, 1]``. Convs feeding a BatchNorm drop their bias (it would be
    absorbed by the normalization anyway).
    """

    def __init__(self, z_dim: int = 100, channels: int = 1):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 256, 4, 1, 0, bias=False),  # B x 256 x 4 x 4
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # B x 128 x 8 x 8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),  # B x 64 x 16 x 16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, channels, 4, 2, 1),  # B x C x 32 x 32
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.view(z.shape[0], self.z_dim, 1, 1))


class DCGANDiscriminator(nn.Module):
    """Mirror of the generator: stride-2 convs ``C -> 64 -> 128 -> 256 -> 1`` logit.

    LeakyReLU(0.2) throughout; BatchNorm on every block except the first (the
    DCGAN guideline). Emits a raw logit — NO sigmoid — so the trainer can use
    the numerically stable ``BCEWithLogitsLoss``.
    """

    def __init__(self, channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1),  # B x 64 x 16 x 16 (no BN on first layer)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # B x 128 x 8 x 8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # B x 256 x 4 x 4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0),  # B x 1 x 1 x 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)  # B raw logits


class DCGANTrainer(BaseTrainer):
    """Non-saturating DCGAN sharing the OT-GAN evaluation harness.

    Calibration baseline so the FID harness has a same-evaluator reference point
    (Lucic et al. 2018 report WGAN ~6.7, WGAN-GP ~20.3 on MNIST with large
    budgets). The dataloader yields ``2B`` images per step (the OT-GAN needs two
    independent real halves); DCGAN simply trains on the full ``2B`` batch.
    """

    ckpt_name = "dcgan.pt"

    def _build(self) -> None:
        cfg = self.cfg
        self.generator = DCGANGenerator(z_dim=cfg.z_dim, channels=cfg.channels).to(self.device)
        self.discriminator = DCGANDiscriminator(channels=cfg.channels).to(self.device)
        self.ema = EMAGenerator(self.generator, cfg.ema_decay).to(self.device)
        self.g_opt = optim.Adam(
            self.generator.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2)
        )
        self.d_opt = optim.Adam(
            self.discriminator.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2)
        )
        self.bce = nn.BCEWithLogitsLoss()

    def _modules(self) -> list:
        return [self.generator, self.discriminator]

    def _step(self, batch: torch.Tensor, i: int) -> dict:
        n = batch.shape[0]  # the full 2B batch
        real_labels = torch.ones(n, device=self.device)
        fake_labels = torch.zeros(n, device=self.device)

        # Discriminator: push real logits toward 1, (detached) fake logits toward 0.
        fake = self.generator(self.sample_z(n))
        d_loss = self.bce(self.discriminator(batch), real_labels) + self.bce(
            self.discriminator(fake.detach()), fake_labels
        )
        self.d_opt.zero_grad(set_to_none=True)
        d_loss.backward()
        self.d_opt.step()

        # Generator: non-saturating loss — maximize log D(G(z)) on fresh fakes.
        # Gradients also reach the discriminator here, but d_opt.zero_grad()
        # discards them before its next step (the textbook DCGAN loop).
        fake = self.generator(self.sample_z(n))
        g_loss = self.bce(self.discriminator(fake), real_labels)
        self.g_opt.zero_grad(set_to_none=True)
        g_loss.backward()
        self.g_opt.step()
        self.ema.update(self.generator)

        return {"d_loss": float(d_loss.detach()), "g_loss": float(g_loss.detach())}

    def _format_line(self, means: dict) -> str:
        return f"d_loss={means['d_loss']:.4f} g_loss={means['g_loss']:.4f}"

    # ---- checkpoint state ------------------------------------------------
    def _checkpoint_state(self) -> dict:
        return {
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "ema": self.ema.state_dict(),
            "g_opt": self.g_opt.state_dict(),
            "d_opt": self.d_opt.state_dict(),
        }

    def _load_checkpoint_state(self, ckpt: dict) -> None:
        self.generator.load_state_dict(ckpt["generator"])
        self.discriminator.load_state_dict(ckpt["discriminator"])
        self.ema.load_state_dict(ckpt["ema"])
        self.g_opt.load_state_dict(ckpt["g_opt"])
        self.d_opt.load_state_dict(ckpt["d_opt"])
