#!/usr/bin/env python3
"""Compare centralized vs federated metrics and plot an overlay."""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib.pyplot as plt


def read_centralized(path: Path):
    vals = {}
    if not path.exists():
        raise FileNotFoundError(f"Centralized metrics not found: {path}")
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            vals[row["metric"]] = float(row["value"]) if row["metric"] != "F1" else float(row["value"])
    return vals


def read_fl_metrics(path: Path):
    rounds, aucs, precs, recs, f1s = [], [], [], [], []
    if not path.exists():
        raise FileNotFoundError(f"Federated metrics log not found: {path}")
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rounds.append(int(row["round"]))
            aucs.append(float(row["AUC"]))
            precs.append(float(row["Precision"]))
            recs.append(float(row["Recall"]))
            f1s.append(float(row["F1"]))
    return rounds, aucs, precs, recs, f1s


def compare_and_plot():
    cen_path = Path("outputs/metrics/centralized_metrics.csv")
    fl_path = Path("server/state/metrics_log.csv")
    cen = read_centralized(cen_path)
    rounds, aucs, _, _, f1s = read_fl_metrics(fl_path)

    # latest federated
    auc_last = aucs[-1] if aucs else 0
    f1_last = f1s[-1] if f1s else 0

    # write CSV
    out_dir = Path("outputs/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmp_path = out_dir / "baseline_vs_fl.csv"
    with cmp_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "centralized", "federated_last_round"]) 
        w.writerow(["AUC", f"{cen.get('AUC', 0):.6f}", f"{auc_last:.6f}"])
        w.writerow(["F1", f"{cen.get('F1', 0):.6f}", f"{f1_last:.6f}"])
    print(f"Saved comparison to {cmp_path}")

    # plot overlay
    out_plots = Path("outputs/plots")
    out_plots.mkdir(parents=True, exist_ok=True)
    out_path = out_plots / "centralized_vs_fl.png"

    plt.figure(figsize=(8, 5))
    if rounds:
        plt.plot(rounds, aucs, label="FL AUC")
        plt.plot(rounds, f1s, label="FL F1")
    plt.axhline(cen.get('AUC', 0), color='C0', linestyle='--', alpha=0.6, label="Centralized AUC")
    plt.axhline(cen.get('F1', 0), color='C1', linestyle='--', alpha=0.6, label="Centralized F1")
    plt.xlabel("Round")
    plt.ylabel("Metric")
    plt.title("Centralized vs Federated (server-side evaluation)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    compare_and_plot()
