"""Typed configuration. Replaces the original ``AttrDict`` with a validated
dataclass that round-trips to YAML.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


@dataclass
class Config:
    # Data
    dataset: str = "mnist"  # 'mnist' | 'cifar10'
    channels: int = 1
    image_size: int = 32
    data_root: str = "data"

    # Optimization
    batch_size: int = 64
    z_dim: int = 100
    learning_rate: float = 3e-4
    beta1: float = 0.5
    beta2: float = 0.999
    g2c_ratio: int = 3
    n_epochs: int = 10
    latent: str = "gaussian"  # 'gaussian' | 'uniform'

    # Optimal transport
    epsilon: float = 1.0
    sinkhorn_iters: int = 100
    # 'energy_distance' = Salimans et al. 2018 (independent-minibatch self-terms);
    # 'sinkhorn_divergence' = debiased same-batch self-terms (Genevay et al. 2018,
    # Feydy et al. 2019). See otgan/energy.py.
    loss: str = "energy_distance"

    # Model family: 'otgan' (default), 'dcgan' (baseline), 'cfm' (OT flow matching)
    model: str = "otgan"

    # Flow matching (only read when model == 'cfm')
    cfm_coupling: str = "sinkhorn"  # 'sinkhorn' | 'exact' (POT) | 'none' (I-CFM)
    cfm_eps: float = 0.05  # entropic eps for the minibatch coupling
    ode_steps: int = 100  # Euler steps when sampling from the flow

    # Quality levers
    ema_decay: float = 0.999
    critic_sign: bool = True  # False reproduces the original (buggy) descent

    # Evaluation / logging
    n_samples: int = 64
    log_step: int = 1
    fid_every: int = 5
    n_eval: int = 10000  # samples per side for FID/KID (MNIST test set caps real at 10k)
    kid_subset_size: int = 1000
    eval_cache_dir: str = "eval_cache"  # disk cache for real features / LeNet weights
    eval_dir: str = "OTGAN_eval"
    ckpt_dir: str = "weights"
    log_dir: str = "runs"

    # Reproducibility / misc
    seed: int = 11
    device: str = "auto"
    max_batches: int | None = None
    num_workers: int = 2

    def __post_init__(self):
        if self.dataset not in ("mnist", "cifar10"):
            raise ValueError(f"dataset must be 'mnist' or 'cifar10', got {self.dataset!r}")
        if self.latent not in ("gaussian", "uniform"):
            raise ValueError(f"latent must be 'gaussian' or 'uniform', got {self.latent!r}")
        if self.loss not in ("energy_distance", "sinkhorn_divergence"):
            raise ValueError(
                f"loss must be 'energy_distance' or 'sinkhorn_divergence', got {self.loss!r}"
            )
        if self.model not in ("otgan", "dcgan", "cfm"):
            raise ValueError(f"model must be 'otgan', 'dcgan' or 'cfm', got {self.model!r}")
        if self.cfm_coupling not in ("sinkhorn", "exact", "none"):
            raise ValueError(
                f"cfm_coupling must be 'sinkhorn', 'exact' or 'none', got {self.cfm_coupling!r}"
            )
        if self.channels not in (1, 3):
            raise ValueError(f"channels must be 1 or 3, got {self.channels}")
        for name in (
            "batch_size",
            "z_dim",
            "g2c_ratio",
            "n_epochs",
            "sinkhorn_iters",
            "n_samples",
            "kid_subset_size",
            "ode_steps",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon}")
        if self.cfm_eps <= 0:
            raise ValueError(f"cfm_eps must be > 0, got {self.cfm_eps}")
        if not (0.0 <= self.ema_decay < 1.0):
            raise ValueError(f"ema_decay must be in [0, 1), got {self.ema_decay}")
        if self.max_batches is not None and self.max_batches <= 0:
            raise ValueError(f"max_batches must be positive or null, got {self.max_batches}")

    # ---- serialization -------------------------------------------------
    @classmethod
    def from_yaml(cls, path) -> Config:
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
        return cls(**data)

    def to_yaml(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            yaml.safe_dump(asdict(self), fh, sort_keys=False)

    def to_dict(self) -> dict:
        return asdict(self)

    # ---- pretty print (prints EVERY field, including 0/False/None) -----
    def print_config(self) -> None:
        print("=" * 80)
        print("Configuration".center(80))
        print("=" * 80)
        for key, value in asdict(self).items():
            print(f"{key:>20}: {value!s:<30}")
        print("=" * 80)
