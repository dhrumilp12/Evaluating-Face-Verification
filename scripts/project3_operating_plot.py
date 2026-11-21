#!/usr/bin/env python3
"""Plot the measured target-FMR policy tradeoff without fitting new models."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "results/project3/operating_points")
    parser.add_argument("--output", type=Path, default=ROOT / "results/project3/operating_points_analysis")
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest.get("status") != "completed":
        raise ValueError("A completed operating-point run is required")
    data_file = run / "aggregates.csv"
    data_hash = digest(data_file)
    if data_hash != manifest["output_sha256"]["aggregates.csv"]:
        raise ValueError("Aggregate results do not match the completed manifest")
    if output == run or output.is_relative_to(run):
        raise ValueError("Keep derived figures outside the measured run directory")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new or empty figure directory")
    with data_file.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = {(r["method"], r["policy"]): r for r in rows
                if r["regime"] == "mixed" and r["condition"] == "resolution_14"}
    methods = ("cosine_threshold", "qmagface_threshold", "logistic_interaction")
    policies = ("pooled", "condition_cap")
    if set(selected) != {(m, p) for m in methods for p in policies}:
        raise ValueError("Expected all six primary mixed-calibration comparisons")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.4))
    fig.subplots_adjust(left=.075, right=.98, top=.72, bottom=.21, wspace=.27)
    x = np.arange(len(methods))
    colors = ("#3E66A5", "#087F8C")
    names = ("Pooled calibration", "Per-condition calibration cap")
    for ax, metric, title in zip(axes, ("fmr", "fnmr"), ("False-match rate", "False-nonmatch rate")):
        all_values = []
        for j, policy in enumerate(policies):
            values = [100 * float(selected[m, policy][f"mean_{metric}"]) for m in methods]
            all_values.extend(values)
            bars = ax.bar(x + (j - .5) * .36, values, width=.34, color=colors[j], label=names[j], zorder=3)
            if metric == "fmr":
                ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], label_type="center",
                             color="white", fontsize=9)
            else:
                ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=4, fontsize=9)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=12)
        ax.set_xticks(x, ["Cosine", "QMagFace", "Learned score"])
        ax.set_ylabel("Held-out rate (%)")
        ax.grid(axis="y", color="#E0E6EB", zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(1.6, max(all_values) * 1.3) if metric == "fmr" else 100)
        if metric == "fmr":
            ax.axhline(1, color="#B44F24", linewidth=1.4, linestyle="--", zorder=2)
            ax.text(.98, .94, "Dashed line: 1% calibration target", transform=ax.transAxes,
                    ha="right", va="top", color="#9B421E", fontsize=9)
    fig.suptitle("A calibration target changes the verification tradeoff", x=.075, ha="left",
                 y=.98, fontsize=16, fontweight="bold", color="#18324A")
    fig.text(.075, .89, "Mixed conditions; 14-pixel held-out probes; target calibration FMR = 1%", fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.068, .86), ncol=2, frameon=False)
    fig.text(.075, .09, "Rates are means over ten pair folds. The target constrains calibration data; test FMR can exceed 1%.", fontsize=10)
    fig.text(.075, .045, "Each condition contains 3,000 held-out impostor and 2,999 eligible genuine pairs across the folds.", fontsize=10)
    output.mkdir(parents=True, exist_ok=True)
    figure = output / "target_fmr_tradeoff.png"
    fig.savefig(figure, dpi=170, facecolor="white")
    plt.close(fig)
    receipt = {"source_run_manifest_sha256": digest(run / "manifest.json"),
               "source_aggregates_sha256": data_hash,
               "script_sha256": digest(Path(__file__)),
               "scope": "Mixed regime, resolution_14, mean-fold FMR/FNMR; no model fitting",
               "output_sha256": {figure.name: digest(figure)}}
    (output / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"figure": str(figure), **receipt}, indent=2))


if __name__ == "__main__":
    main()
