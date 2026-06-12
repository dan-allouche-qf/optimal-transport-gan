"""Tests for the command-line interface (``otgan/cli.py``)."""

from pathlib import Path

import pytest

from otgan.cli import _coerce, _parse_overrides, build_parser, main

SMOKE_CONFIG = str(Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml")


# ---- _coerce -----------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        # bool fields
        ("critic_sign", "true", True),
        ("critic_sign", "1", True),
        ("critic_sign", "yes", True),
        ("critic_sign", "false", False),
        ("critic_sign", "0", False),
        # int / float / str fields
        ("batch_size", "16", 16),
        ("epsilon", "0.5", 0.5),
        ("dataset", "cifar10", "cifar10"),
        # 'none'/'null' map to None only for Optional fields; plain str fields
        # keep the literal string (e.g. cfm_coupling=none is a coupling MODE)
        ("cfm_coupling", "none", "none"),
        # Optional[int] (annotated ``int | None``) still coerces to int
        ("max_batches", "3", 3),
        ("max_batches", "none", None),
        ("max_batches", "null", None),
        ("max_batches", "Null", None),
    ],
)
def test_coerce(name, value, expected):
    out = _coerce(name, value)
    assert out == expected
    assert type(out) is type(expected)  # e.g. bool, not int, for critic_sign=1


def test_coerce_unknown_name_falls_back_to_str():
    assert _coerce("not_a_config_field", "42") == "42"


# ---- _parse_overrides --------------------------------------------------


def test_parse_overrides_builds_coerced_dict():
    out = _parse_overrides(["epsilon=0.5", "max_batches=none", "critic_sign=false"])
    assert out == {"epsilon": 0.5, "max_batches": None, "critic_sign": False}


def test_parse_overrides_splits_on_first_equals_only():
    assert _parse_overrides(["eval_dir=a=b"]) == {"eval_dir": "a=b"}


def test_parse_overrides_accepts_empty_and_none():
    assert _parse_overrides([]) == {}
    assert _parse_overrides(None) == {}


def test_parse_overrides_missing_equals_errors():
    with pytest.raises(SystemExit, match="key=value"):
        _parse_overrides(["epsilon0.5"])


def test_parse_overrides_unknown_key_errors():
    with pytest.raises(SystemExit, match="Unknown config key"):
        _parse_overrides(["bogus=1"])


# ---- parser ------------------------------------------------------------


@pytest.mark.parametrize("command", ["train", "sample", "eval", "ablate", "config"])
def test_parser_accepts_every_subcommand(command):
    args = build_parser().parse_args([command, "-c", "cfg.yaml"])
    assert args.command == command
    assert args.config == "cfg.yaml"
    assert args.override == []
    assert callable(args.func)


def test_parser_requires_a_subcommand():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_parser_requires_config():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train"])


def test_parser_sample_flags_and_defaults():
    args = build_parser().parse_args(["sample", "-c", "cfg.yaml"])
    assert (args.ckpt, args.n, args.nrow, args.out) == (None, 64, 8, "samples.png")
    args = build_parser().parse_args(
        ["sample", "-c", "cfg.yaml", "--ckpt", "w.pt", "-n", "4", "--nrow", "2", "-o", "x.png"]
    )
    assert (args.ckpt, args.n, args.nrow, args.out) == ("w.pt", 4, 2, "x.png")


def test_parser_ablate_axis_choices():
    assert build_parser().parse_args(["ablate", "-c", "c.yaml"]).axis == "critic_sign"
    assert build_parser().parse_args(["ablate", "-c", "c.yaml", "--axis", "epsilon"]).axis == (
        "epsilon"
    )
    with pytest.raises(SystemExit):
        build_parser().parse_args(["ablate", "-c", "c.yaml", "--axis", "nope"])


# ---- end-to-end via main() ----------------------------------------------


def test_main_config_prints_resolved_config(capsys):
    main(["config", "-c", SMOKE_CONFIG, "--override", "epsilon=0.5", "max_batches=none"])
    out = capsys.readouterr().out
    assert "Configuration" in out
    epsilon_line = next(line for line in out.splitlines() if "epsilon" in line)
    assert epsilon_line.split(":")[1].strip() == "0.5"  # override applied
    max_batches_line = next(line for line in out.splitlines() if "max_batches" in line)
    assert max_batches_line.split(":")[1].strip() == "None"


def test_main_sample_writes_png_from_checkpoint(tmp_path, monkeypatch, capsys, tiny_config):
    """Build a tiny checkpoint, then ``otgan sample`` must load it and write a grid PNG."""
    from otgan.trainer import OTGANTrainer

    monkeypatch.setenv("OT_GAN_ROOT", str(tmp_path))
    ckpt_path = OTGANTrainer(tiny_config).save_checkpoint(epoch=0)
    assert Path(ckpt_path).is_relative_to(tmp_path)  # honors OT_GAN_ROOT

    out_png = tmp_path / "cli_samples.png"
    main(
        [
            "sample",
            "-c",
            SMOKE_CONFIG,
            "--ckpt",
            ckpt_path,
            "-n",
            "4",
            "--nrow",
            "2",
            "-o",
            str(out_png),
            "--override",
            "device=cpu",
        ]
    )
    assert out_png.exists() and out_png.stat().st_size > 0
    assert f"Wrote 4 samples to {out_png}" in capsys.readouterr().out
