"""Device resolution and reproducible seeding.

Replaces the original notebook's hard-coded ``.cuda()`` calls, which raised on any
machine without CUDA (e.g. an Apple-Silicon Mac, where MPS is the accelerator).
"""

import contextlib
import random

import numpy as np
import torch


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve a device string. ``"auto"`` prefers CUDA, then MPS, then CPU."""
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch (CPU + CUDA + MPS) for best-effort reproducibility.

    Bit-exact reproducibility is asserted only on CPU; MPS/CUDA are seed-level
    (kernel nondeterminism across backends is not fully controllable).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    mps = getattr(torch, "mps", None)
    if mps is not None and torch.backends.mps.is_available():
        with contextlib.suppress(Exception):
            torch.mps.manual_seed(seed)
