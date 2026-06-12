"""Plotting helpers: training curves, fixed-noise grid evolution, ablation overlays.

Uses the non-interactive Agg backend so figures render headless (Colab/CI/CLI).
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def read_history(csv_path):
    rows = []
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append({k: (float(v) if v not in ("", None) else None) for k, v in row.items()})
    return rows


def _series(history, key):
    xs = [r["epoch"] for r in history if r.get(key) is not None]
    ys = [r[key] for r in history if r.get(key) is not None]
    return xs, ys


def plot_curves(history, out):
    """Energy distance (+ sub-terms), FID and IS over epochs."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for key in ("energy_distance", "cross", "real_real", "fake_fake"):
        xs, ys = _series(history, key)
        if xs:
            axes[0].plot(xs, ys, label=key)
    axes[0].set_title("Minibatch energy distance")
    axes[0].set_xlabel("epoch")
    axes[0].legend(fontsize=8)

    for ax, key, title in (
        (axes[1], "fid", "FID (lower is better)"),
        (axes[2], "is_mean", "Inception Score (higher is better)"),
    ):
        xs, ys = _series(history, key)
        if xs:
            ax.plot(xs, ys, marker="o")
        ax.set_title(title)
        ax.set_xlabel("epoch")

    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_sample_grid_evolution(eval_dir, out, max_cols=6):
    """Montage of the fixed-noise sample grids across epochs."""
    paths = sorted(Path(eval_dir).glob("sample_epoch*.png"))
    if not paths:
        return None
    if len(paths) > max_cols:
        step = len(paths) / max_cols
        paths = [paths[min(int(i * step), len(paths) - 1)] for i in range(max_cols)]
    n = len(paths)
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 2.6))
    if n == 1:
        axes = [axes]
    for ax, p in zip(axes, paths, strict=True):
        ax.imshow(mpimg.imread(p))
        ax.set_title(p.stem.replace("sample_", ""), fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_ablation(results, axis, out):
    """Overlaid FID-vs-epoch (falling back to energy distance if FID absent)."""
    metric = "fid"
    if not any(_series(r["history"], "fid")[0] for r in results):
        metric = "energy_distance"
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in results:
        xs, ys = _series(r["history"], metric)
        if xs:
            ax.plot(xs, ys, marker="o", label=r["tag"])
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.set_title(f"Ablation: {axis}")
    ax.legend()
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
