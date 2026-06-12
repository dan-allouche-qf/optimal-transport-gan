"""OT-GAN — Optimal Transport GAN (Salimans et al. 2018).

A correct, reproducible, device-portable and measurable re-implementation.

Import submodules directly, e.g. ``from otgan.trainer import OTGANTrainer``.
Convenience re-exports are attached lazily via ``__getattr__`` so that importing
a single submodule does not pull in heavy optional dependencies (torchmetrics).
"""

__version__ = "1.0.0"

_LAZY = {
    "Config": "otgan.config",
    "resolve_device": "otgan.device",
    "seed_everything": "otgan.device",
    "OTGANGenerator": "otgan.models",
    "OTGANCritic": "otgan.models",
    "build_dataloader": "otgan.data",
    "cost": "otgan.sinkhorn",
    "sinkhorn": "otgan.sinkhorn",
    "energy_distance": "otgan.energy",
    "EMAGenerator": "otgan.ema",
    "OTGANTrainer": "otgan.trainer",
}

__all__ = ["__version__", *_LAZY.keys()]


def __getattr__(name):
    import importlib

    if name in _LAZY:
        module = importlib.import_module(_LAZY[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
