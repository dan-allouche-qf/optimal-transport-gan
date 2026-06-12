"""Tests for the typed Config."""

import io
from contextlib import redirect_stdout

import pytest

from otgan.config import Config


def test_yaml_round_trip(tmp_path):
    cfg = Config(dataset="cifar10", channels=3, epsilon=0.5, max_batches=7)
    path = tmp_path / "c.yaml"
    cfg.to_yaml(path)
    assert Config.from_yaml(path) == cfg


def test_print_config_shows_falsy_values():
    """Regression for the original print_arguments truthy-filter bug."""
    cfg = Config(critic_sign=False, max_batches=None, fid_every=0)
    out = io.StringIO()
    with redirect_stdout(out):
        cfg.print_config()
    text = out.getvalue()
    assert "critic_sign" in text and "False" in text
    assert "max_batches" in text and "None" in text
    assert "fid_every" in text  # value 0 must appear


@pytest.mark.parametrize(
    "kwargs",
    [
        {"g2c_ratio": 0},
        {"epsilon": 0.0},
        {"epsilon": -1.0},
        {"n_epochs": -5},
        {"dataset": "imagenet"},
        {"latent": "poisson"},
        {"channels": 2},
        {"ema_decay": 1.0},
    ],
)
def test_validation_rejects_bad_values(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_from_yaml_rejects_unknown_keys(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("dataset: mnist\nbogus_key: 3\n")
    with pytest.raises(ValueError):
        Config.from_yaml(path)


def test_shipped_configs_load():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "configs"
    for name in ("mnist.yaml", "cifar10.yaml", "smoke.yaml"):
        Config.from_yaml(root / name)  # must not raise
