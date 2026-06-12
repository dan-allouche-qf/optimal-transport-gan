"""Typed configuration for the finance track, mirroring ``otgan/config.py``:
a validated dataclass that round-trips to YAML and rejects unknown keys.

One convention worth spelling out (it is also enforced in the reduce/trainer
modules and tested): when sinkhorn runs on *raw return paths* (squared
Euclidean cost) rather than on L2-normalized critic embeddings (cosine cost
in [0, 2]), the cost matrix is divided by its mean — ``C = C / C.mean()`` —
before the solver. This makes ``epsilon`` dimensionless, so ``epsilon=1.0``
means the same thing in both regimes and the image defaults transfer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


@dataclass
class FinanceConfig:
    # Data: which return process to learn
    target: str = "gjr_garch"  # 'gjr_garch' | 'gbm' | 'heston' | 'csv'
    csv_path: str | None = None  # required when target == 'csv'
    seq_len: int = 64
    n_train_paths: int = 8192
    n_eval_paths: int = 2048

    # GJR-GARCH(1,1) parameters (Glosten, Jagannathan & Runkle 1993)
    garch_omega: float = 5e-6
    garch_alpha: float = 0.05
    garch_beta: float = 0.90
    garch_gamma: float = 0.05

    # GBM / Heston parameters (annualized; dt = one trading day)
    mu: float = 0.05
    sigma: float = 0.2
    dt: float = 1.0 / 252.0
    heston_kappa: float = 2.0
    heston_theta: float = 0.04
    heston_xi: float = 0.3
    heston_rho: float = -0.7
    s0: float = 100.0

    # Optimization (step-based, unlike the epoch-based image trainer)
    batch_size: int = 64
    z_dim: int = 64
    learning_rate: float = 3e-4
    beta1: float = 0.5
    beta2: float = 0.999
    g2c_ratio: int = 3
    n_steps: int = 3000

    # Optimal transport (epsilon is dimensionless; see module docstring)
    epsilon: float = 1.0
    sinkhorn_iters: int = 50
    loss: str = "energy_distance"  # 'energy_distance' | 'sinkhorn_divergence'

    # Quality levers (same semantics as the image Config)
    ema_decay: float = 0.999
    critic_sign: bool = True

    # Evaluation / logging
    eval_every: int = 500
    n_samples: int = 256
    eval_dir: str = "finance/eval"
    ckpt_dir: str = "finance/ckpt"
    log_dir: str = "finance/logs"

    # Reproducibility / misc
    seed: int = 11
    device: str = "cpu"  # paths are tiny; CPU avoids MPS float64 quirks
    num_workers: int = 0

    def __post_init__(self):
        if self.target not in ("gjr_garch", "gbm", "heston", "csv"):
            raise ValueError(
                f"target must be 'gjr_garch', 'gbm', 'heston' or 'csv', got {self.target!r}"
            )
        if self.target == "csv" and not self.csv_path:
            raise ValueError("target='csv' requires csv_path to be set")
        if self.loss not in ("energy_distance", "sinkhorn_divergence"):
            raise ValueError(
                f"loss must be 'energy_distance' or 'sinkhorn_divergence', got {self.loss!r}"
            )
        for name in (
            "seq_len",
            "n_train_paths",
            "n_eval_paths",
            "batch_size",
            "z_dim",
            "g2c_ratio",
            "n_steps",
            "sinkhorn_iters",
            "eval_every",
            "n_samples",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        for name in (
            "garch_omega",
            "sigma",
            "dt",
            "heston_kappa",
            "heston_theta",
            "heston_xi",
            "s0",
            "learning_rate",
            "epsilon",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        for name in ("garch_alpha", "garch_beta", "garch_gamma"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        persistence = self.garch_alpha + self.garch_beta + self.garch_gamma / 2.0
        if persistence >= 1.0:
            raise ValueError(
                "GJR-GARCH stationarity requires garch_alpha + garch_beta + garch_gamma/2 < 1 "
                "(the unconditional variance garch_omega / (1 - garch_alpha - garch_beta - "
                f"garch_gamma/2) must be finite and positive); got {persistence:.4f}"
            )
        if not (-1.0 < self.heston_rho < 1.0):
            raise ValueError(f"heston_rho must be in (-1, 1), got {self.heston_rho}")
        if not (0.0 <= self.ema_decay < 1.0):
            raise ValueError(f"ema_decay must be in [0, 1), got {self.ema_decay}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")

    # ---- serialization -------------------------------------------------
    @classmethod
    def from_yaml(cls, path) -> FinanceConfig:
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
        print("Finance configuration".center(80))
        print("=" * 80)
        for key, value in asdict(self).items():
            print(f"{key:>20}: {value!s:<30}")
        print("=" * 80)
