"""Parse-level and end-to-end tests for the finance CLI subcommands."""

import json

import pytest

from otgan.cli import _load_finance_config, _parse_overrides, build_parser, main
from otgan.finance.config import FinanceConfig


def test_parser_accepts_finance_subcommands():
    for command in ("finance-train", "finance-eval", "finance-reduce"):
        argv = [command, "-c", "cfg.yaml"]
        if command == "finance-eval":
            argv += ["--ckpt", "w.pt"]
        args = build_parser().parse_args(argv)
        assert args.command == command
        assert callable(args.func)


def test_finance_eval_requires_ckpt():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["finance-eval", "-c", "cfg.yaml"])


def test_finance_reduce_flags():
    args = build_parser().parse_args(
        ["finance-reduce", "-c", "c.yaml", "-K", "10", "--epsilon", "0.5", "--compare"]
    )
    assert (args.k, args.epsilon, args.compare) == (10, 0.5, True)


def test_overrides_coerce_against_finance_config():
    out = _parse_overrides(["garch_gamma=0.1", "n_steps=20", "critic_sign=false"], FinanceConfig)
    assert out == {"garch_gamma": 0.1, "n_steps": 20, "critic_sign": False}
    with pytest.raises(SystemExit, match="Unknown config key"):
        _parse_overrides(["fid_every=5"], FinanceConfig)  # image-only key


def test_load_finance_config_smoke_yaml():
    cfg = _load_finance_config("configs/finance_smoke.yaml", ["seed=3"])
    assert isinstance(cfg, FinanceConfig)
    assert cfg.seed == 3


def test_finance_reduce_end_to_end(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OT_GAN_ROOT", str(tmp_path))
    main(
        [
            "finance-reduce",
            "-c",
            "configs/finance_smoke.yaml",
            "-K",
            "4",
            "--epsilon",
            "0.1",
            "--compare",
            "--override",
            "n_train_paths=64",
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert set(report) == {"sinkhorn", "kmeans", "random"}
    assert all("cvar0.95_full" in r for r in report.values())
