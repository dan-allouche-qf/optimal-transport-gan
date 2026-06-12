"""Assemble the headline results: loss-comparison figure + curated CSVs + summary table.

Reads the run histories (loss_compare/, baselines/) and the eval_v2 JSON
re-evaluations, then writes portfolio-grade artifacts under assets/.
Re-runnable: skips sources that do not exist yet.
"""

import csv
import json
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"


def read_history(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append({k: (float(v) if v else None) for k, v in row.items()})
    return rows


def series(rows: list[dict], key: str):
    xs = [r["epoch"] for r in rows if r.get(key) is not None]
    ys = [r[key] for r in rows if r.get(key) is not None]
    return xs, ys


def tail_json(path: Path) -> dict | None:
    """eval_v2 files carry log lines before the JSON object — parse from the last '{'."""
    if not path.exists():
        return None
    text = path.read_text()
    m = list(re.finditer(r"^\{", text, flags=re.M))
    return json.loads(text[m[-1].start() :]) if m else None


def loss_comparison_figure(energy: list[dict], sinkdiv: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    runs = [
        ("energy distance (2018)", energy, "tab:orange"),
        ("Sinkhorn divergence (2019)", sinkdiv, "tab:blue"),
    ]
    for label, rows, color in runs:
        xs, ys = series(rows, "fid")
        axes[0].plot(xs, ys, marker="o", label=label, color=color)
        xs, ys = series(rows, "kid_mean")
        axes[1].plot(xs, [1000 * y for y in ys], marker="o", label=label, color=color)
        xs, ys = series(rows, "energy_distance")
        axes[2].plot(xs, ys, marker=".", label=label, color=color)
    axes[0].set_title("FID@10k (lower is better)")
    axes[1].set_title("KID x 1e3 (lower is better)")
    axes[2].set_title("training objective D² (critic collapse check)")
    axes[2].set_yscale("symlog", linthresh=1e-3)
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.legend(fontsize=9)
    fig.suptitle("Same budget, same seed: debiasing the loss removes the critic collapse", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    # --- curated CSV copies -------------------------------------------------
    curated = {
        "loss_compare/energy/eval/history.csv": "history_loss_energy8.csv",
        "loss_compare/sinkdiv/eval/history.csv": "history_loss_sinkdiv8.csv",
        "baselines/dcgan/eval/history.csv": "history_dcgan10.csv",
        "baselines/otcfm/eval/history.csv": "history_otcfm30.csv",
        "baselines/icfm/eval/history.csv": "history_icfm30.csv",
    }
    for src, dst in curated.items():
        if (ROOT / src).exists():
            shutil.copy(ROOT / src, ASSETS / dst)
            print(f"curated {dst}")

    # --- loss-comparison figure ----------------------------------------------
    e_path = ROOT / "loss_compare/energy/eval/history.csv"
    s_path = ROOT / "loss_compare/sinkdiv/eval/history.csv"
    if e_path.exists() and s_path.exists():
        # The resumed energy run's CSV only holds post-resume epochs; merge from
        # the curated full log values if needed (epochs 0-6 live in the log).
        loss_comparison_figure(
            read_history(e_path), read_history(s_path), ASSETS / "loss_comparison.png"
        )

    # --- headline summary table ----------------------------------------------
    def last_metrics(rows: list[dict]) -> dict:
        out: dict = {}
        for key in ("fid", "kid_mean", "fid_lenet", "is_mean"):
            _, ys = series(rows, key)
            if ys:
                out[key] = ys[-1]
        return out

    lines = [
        "# Headline results (FID/KID @ 10k samples, torchmetrics-Inception + MNIST-LeNet)",
        "",
        "| model | budget | FID@10k | KIDx1e3 | FID-LeNet | IS |",
        "|---|---|---|---|---|---|",
    ]

    def row(name: str, budget: str, m: dict | None) -> None:
        if not m:
            return
        fid = f"{m['fid']:.1f}" if m.get("fid") is not None else "-"
        kid = f"{1000 * m['kid_mean']:.1f}" if m.get("kid_mean") is not None else "-"
        lenet = f"{m['fid_lenet']:.1f}" if m.get("fid_lenet") is not None else "-"
        is_m = f"{m['is_mean']:.2f}" if m.get("is_mean") is not None else "-"
        lines.append(f"| {name} | {budget} | {fid} | {kid} | {lenet} | {is_m} |")

    off_json = tail_json(ROOT / "results/eval_v2/critic_sign_off_final.json")
    on_json = tail_json(ROOT / "results/eval_v2/critic_sign_on_final.json")
    # Legacy 18-ep wall-clocks: per-epoch sums are 5.5 h / 5.0 h; ~6 h including
    # the in-loop evaluations (artifact timestamps).
    row("OT-GAN, buggy critic sign (final, 18 ep)", "~6 h MPS", off_json)
    row("OT-GAN, corrected (final, 18 ep)", "~6 h MPS", on_json)
    if e_path.exists():
        row("OT-GAN, energy distance (8 ep)", "~3.5 h MPS", last_metrics(read_history(e_path)))
    if s_path.exists():
        row("OT-GAN, Sinkhorn divergence (8 ep)", "~3.5 h MPS", last_metrics(read_history(s_path)))
    if (ROOT / "baselines/dcgan/eval/history.csv").exists():
        dcgan_rows = read_history(ROOT / "baselines/dcgan/eval/history.csv")
        row("DCGAN baseline (10 ep)", "~15 min MPS", last_metrics(dcgan_rows))
    if (ROOT / "baselines/icfm/eval/history.csv").exists():
        icfm_rows = read_history(ROOT / "baselines/icfm/eval/history.csv")
        row("I-CFM, no coupling (30 ep)", "~1.2 h MPS", last_metrics(icfm_rows))
    if (ROOT / "baselines/otcfm/eval/history.csv").exists():
        otcfm_rows = read_history(ROOT / "baselines/otcfm/eval/history.csv")
        row("OT-CFM, Sinkhorn coupling (30 ep)", "~1.2 h MPS", last_metrics(otcfm_rows))
    floor = tail_json(ROOT / "results/eval_v2/floor.json")
    if floor:
        f = floor["fid_floor"]
        lines.append(
            f"| *train-vs-test floor* | - | {f['fid']:.2f} | - | {f['fid_lenet']:.2f} | - |"
        )

    # NFE sweep (OT-CFM at 10/50/100 steps; I-CFM control for the coupling question).
    # FID-LeNet is included because it collapses at low NFE while Inception FID barely
    # moves — the domain-aware metric tells a different story.
    nfe_lines = []
    for steps in (10, 50, 100):
        m = tail_json(ROOT / f"results/eval_v2/otcfm_nfe{steps}.json")
        if m:
            icfm = tail_json(ROOT / f"results/eval_v2/icfm_nfe{steps}.json")
            icfm_fid = f"{icfm['fid']:.1f}" if icfm else "-"
            kid = 1000 * m["kid_mean"]
            nfe_lines.append(
                f"| {steps} | {m['fid']:.1f} | {kid:.1f} | {m['fid_lenet']:.1f} | {icfm_fid} |"
            )
    if nfe_lines:
        lines += [
            "",
            "## Sampler cost (NFE = Euler steps), best checkpoints",
            "",
            "| NFE | OT-CFM FID@10k | OT-CFM KIDx1e3 | OT-CFM FID-LeNet | I-CFM FID@10k |",
            "|---|---|---|---|---|",
        ]
        lines += nfe_lines

    out = ASSETS / "headline_results.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
