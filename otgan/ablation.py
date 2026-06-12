"""Ablation harness.

Runs the same training across the values of one axis and produces an overlaid
FID curve + a final-FID table. The headline ablation is ``critic_sign`` ON vs OFF,
which quantifies that the audited sign fix actually lowers FID.
"""

from dataclasses import replace
from typing import Any

import yaml

from otgan.config import Config
from otgan.paths import resolve
from otgan.plotting import plot_ablation

AXES: "dict[str, list[Any]]" = {
    "critic_sign": [True, False],
    "epsilon": [0.1, 1.0, 10.0],
    "g2c_ratio": [1, 3, 5],
}


def _load_base(config_path, overrides=None) -> Config:
    with open(config_path) as fh:
        data = yaml.safe_load(fh) or {}
    data.update(overrides or {})
    return Config(**data)


def run_experiment(cfg: Config, tag: str) -> dict:
    from otgan.trainer import OTGANTrainer  # lazy (heavy)

    print(f"\n=== Ablation run: {tag} ===")
    trainer = OTGANTrainer(cfg)
    history = trainer.fit()
    return {"tag": tag, "history": history, "cfg": cfg}


def _final(history, key):
    vals = [r[key] for r in history if r.get(key) is not None]
    return vals[-1] if vals else None


def run_ablation(config_path, axis: str = "critic_sign", overrides=None):
    if axis not in AXES:
        raise ValueError(f"axis must be one of {list(AXES)}, got {axis!r}")
    base = _load_base(config_path, overrides)
    out_root = f"ablation/{axis}"

    results = []
    for value in AXES[axis]:
        run_overrides: dict[str, Any] = {
            axis: value,
            "eval_dir": f"{out_root}/{value}/eval",
            "ckpt_dir": f"{out_root}/{value}/ckpt",
            "log_dir": f"{out_root}/{value}/logs",
        }
        cfg = replace(base, **run_overrides)
        results.append(run_experiment(cfg, f"{axis}={value}"))

    out_dir = resolve(out_root, create=True)
    plot_ablation(results, axis, str(out_dir / "fid_curves.png"))
    _write_table(results, axis, out_dir / "final.md")
    return results


def _write_table(results, axis, path) -> None:
    lines = [
        f"# Ablation: {axis}",
        "",
        "| run | final FID | final IS | final D^2 |",
        "|-----|-----------|----------|-----------|",
    ]
    for r in results:
        fid = _final(r["history"], "fid")
        is_mean = _final(r["history"], "is_mean")
        d2 = _final(r["history"], "energy_distance")
        fid_s = f"{fid:.2f}" if fid is not None else "-"
        is_s = f"{is_mean:.2f}" if is_mean is not None else "-"
        d2_s = f"{d2:+.4f}" if d2 is not None else "-"
        lines.append(f"| {r['tag']} | {fid_s} | {is_s} | {d2_s} |")
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    print("\n" + text)
