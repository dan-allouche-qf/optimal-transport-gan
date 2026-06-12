"""Exponential moving average of the generator weights.

Sampling from the EMA generator (instead of the raw one) is a cheap, standard win
for GAN sample stability and sharpness. The EMA is updated only on generator steps.
"""

import copy

import torch
import torch.nn as nn


class EMAGenerator:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for s, p in zip(self.shadow.parameters(), model.parameters(), strict=True):
            # s <- d * s + (1 - d) * p, in-place on the shadow parameter's data.
            s.detach().lerp_(p.detach(), 1.0 - d)
        for sb, pb in zip(self.shadow.buffers(), model.buffers(), strict=True):
            sb.copy_(pb)

    def to(self, device) -> "EMAGenerator":
        self.shadow.to(device)
        return self

    @property
    def model(self) -> nn.Module:
        return self.shadow

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, sd) -> None:
        self.shadow.load_state_dict(sd)
