"""Tests for the ablation harness (offline: training and plotting are stubbed)."""

import pytest

from otgan import ablation
from otgan.ablation import AXES, _final, _load_base, run_ablation
from otgan.config import Config

BASE_YAML = """\
dataset: mnist
channels: 1
n_epochs: 2
epsilon: 2.0
num_workers: 0
device: cpu
"""


@pytest.fixture
def base_config_path(tmp_path):
    path = tmp_path / "base.yaml"
    path.write_text(BASE_YAML)
    return path


# ---- _final ----------------------------------------------------------------


def test_final_returns_last_non_none():
    history = [{"fid": 1.0}, {"fid": None}, {"fid": 3.0}]
    assert _final(history, "fid") == 3.0


def test_final_skips_trailing_none():
    history = [{"fid": 1.0}, {"fid": None}]
    assert _final(history, "fid") == 1.0


def test_final_returns_none_when_key_absent():
    assert _final([{"epoch": 0}, {"epoch": 1}], "fid") is None
    assert _final([], "fid") is None


# ---- _load_base ------------------------------------------------------------


def test_load_base_merges_overrides(base_config_path):
    cfg = _load_base(base_config_path, overrides={"epsilon": 0.5, "seed": 123})
    assert isinstance(cfg, Config)
    assert cfg.epsilon == 0.5  # override wins over the YAML value
    assert cfg.seed == 123  # override of a key absent from the YAML
    assert cfg.n_epochs == 2  # untouched YAML value survives


def test_load_base_without_overrides(base_config_path):
    cfg = _load_base(base_config_path)
    assert cfg.epsilon == 2.0 and cfg.dataset == "mnist"


# ---- run_ablation ----------------------------------------------------------


def test_run_ablation_rejects_unknown_axis(base_config_path):
    with pytest.raises(ValueError, match="axis must be one of"):
        run_ablation(base_config_path, axis="learning_rate")


def test_run_ablation_happy_path(tmp_path, monkeypatch, base_config_path):
    monkeypatch.setenv("OT_GAN_ROOT", str(tmp_path))

    histories = {
        "critic_sign=True": [
            {"epoch": 0, "fid": 300.0, "energy_distance": 1.5},
            {"epoch": 1, "is_mean": 2.5},  # fid/D^2 missing on this epoch
            {"epoch": 2, "fid": 120.5, "energy_distance": 0.25},
        ],
        "critic_sign=False": [{"epoch": 0}, {"epoch": 1, "fid": None}],  # no metrics at all
    }
    seen_cfgs: list[Config] = []

    def fake_run_experiment(cfg, tag):
        seen_cfgs.append(cfg)
        return {"tag": tag, "history": histories[tag], "cfg": cfg}

    plot_calls = []
    monkeypatch.setattr(ablation, "run_experiment", fake_run_experiment)
    monkeypatch.setattr(ablation, "plot_ablation", lambda *args: plot_calls.append(args))

    results = run_ablation(base_config_path, axis="critic_sign", overrides={"seed": 7})

    # One result per axis value, in AXES order.
    assert len(results) == len(AXES["critic_sign"])
    assert [r["tag"] for r in results] == ["critic_sign=True", "critic_sign=False"]

    # Per-value cfg overrides: axis value, run-scoped output dirs, and user overrides.
    for cfg, value in zip(seen_cfgs, AXES["critic_sign"], strict=True):
        assert cfg.critic_sign is value
        assert cfg.eval_dir == f"ablation/critic_sign/{value}/eval"
        assert cfg.ckpt_dir == f"ablation/critic_sign/{value}/ckpt"
        assert cfg.log_dir == f"ablation/critic_sign/{value}/logs"
        assert cfg.seed == 7  # user override survives the per-value replace

    # Plot stub called once with the results and a png path under OT_GAN_ROOT.
    assert plot_calls == [
        (results, "critic_sign", str(tmp_path / "ablation/critic_sign/fid_curves.png"))
    ]

    # final.md lands under OT_GAN_ROOT with formatted metrics and '-' for missing ones.
    table = (tmp_path / "ablation" / "critic_sign" / "final.md").read_text()
    assert "# Ablation: critic_sign" in table
    assert "| critic_sign=True | 120.50 | 2.50 | +0.2500 |" in table
    assert "| critic_sign=False | - | - | - |" in table
