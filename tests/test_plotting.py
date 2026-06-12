"""Tests for plotting and the ablation table writer (offline, synthetic history)."""

from otgan.ablation import _write_table
from otgan.plotting import plot_ablation, plot_curves, read_history


def _history():
    return [
        {
            "epoch": 0,
            "energy_distance": 1.5,
            "cross": 2.0,
            "real_real": 0.1,
            "fake_fake": 0.1,
            "fid": 380.0,
            "is_mean": 1.1,
        },
        {
            "epoch": 1,
            "energy_distance": 1.2,
            "cross": 1.8,
            "real_real": 0.1,
            "fake_fake": 0.1,
            "fid": 250.0,
            "is_mean": 1.5,
        },
    ]


def test_plot_curves_writes_png(tmp_path):
    out = tmp_path / "curves.png"
    plot_curves(_history(), str(out))
    assert out.exists() and out.stat().st_size > 0


def test_plot_ablation_writes_png(tmp_path):
    results = [
        {"tag": "critic_sign=True", "history": _history()},
        {
            "tag": "critic_sign=False",
            "history": [{"epoch": 0, "fid": 390.0}, {"epoch": 1, "fid": 360.0}],
        },
    ]
    out = tmp_path / "ablation.png"
    plot_ablation(results, "critic_sign", str(out))
    assert out.exists() and out.stat().st_size > 0


def test_history_csv_round_trip(tmp_path):
    import csv

    path = tmp_path / "history.csv"
    rows = _history()
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    loaded = read_history(path)
    assert len(loaded) == 2 and loaded[1]["fid"] == 250.0


def test_ablation_table(tmp_path):
    results = [
        {"tag": "critic_sign=True", "history": _history()},
        {
            "tag": "critic_sign=False",
            "history": [{"epoch": 1, "fid": 360.0, "energy_distance": 0.9}],
        },
    ]
    path = tmp_path / "final.md"
    _write_table(results, "critic_sign", path)
    text = path.read_text()
    assert "critic_sign=True" in text and "250.00" in text
