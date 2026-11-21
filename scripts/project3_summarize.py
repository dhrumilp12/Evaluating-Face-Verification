#!/usr/bin/env python3
"""Build Project 3 scientific figures and traceable summaries from saved results.

This stage never fits or changes a model. All primary comparisons use the
predeclared C=1 run. Held-out quality support is descriptive only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEXT_FIELDS = {"regime", "method", "condition", "reference_method"}
SELECTION = (
    ("original_only", "qmagface_threshold", "Original calibration: QMagFace", "#64748b"),
    ("mixed", "cosine_threshold", "Mixed calibration: cosine", "#c36a28"),
    ("mixed", "qmagface_threshold", "Mixed calibration: QMagFace", "#255b9b"),
    ("mixed", "logistic_interaction", "Mixed calibration: learned quality", "#008678"),
    ("leave_resolution_out", "logistic_interaction", "Unseen resolution: learned quality", "#916296"),
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_table(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [{key: value if key in TEXT_FIELDS else None if value == "" else
             int(value) if re.fullmatch(r"-?\d+", value) else float(value)
             for key, value in row.items()} for row in rows]


def completed_run(run):
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest.get("status") != "completed":
        raise ValueError("Analysis requires a completed Project 3 run")
    for name, digest in manifest["output_sha256"].items():
        path = (run / name).resolve()
        if not path.is_relative_to(run) or not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Run output checksum mismatch: {name}")
    source = ROOT / "results/project2/robustness/pair_scores.jsonl"
    if sha256(source) != manifest["source"]["pair_scores_sha256"]:
        raise ValueError("Project 2 source scores differ from the calibration run")
    return manifest, source


def _save(fig, output, filename, caption):
    fig.text(.075, .018, caption, ha="left", va="bottom", fontsize=9, color="#475569")
    fig.savefig(output / filename, dpi=180, facecolor="white")
    plt.close(fig)


def resolution_figure(index, output):
    names = ("original", "resolution_56", "resolution_28", "resolution_14")
    fig, ax = plt.subplots(figsize=(10.8, 6.1))
    fig.subplots_adjust(left=.08, right=.98, top=.89, bottom=.30)
    for regime, method, label, color in SELECTION:
        values = [100 * index[(regime, method, name)]["mean_accuracy"] for name in names]
        ax.plot(range(4), values, marker="o", markersize=5, linewidth=2, color=color, label=label)
    ax.set_xticks(range(4), ("112\nOriginal", "56", "28", "14"))
    ax.set_xlabel("Probe resolution before restoration to 112 × 112 pixels")
    ax.set_ylabel("Mean fold accuracy (%)")
    ax.set_ylim(50, 101)
    ax.set_title("Calibration changes the response to resolution loss", loc="left", pad=16)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.26), ncols=2, fontsize=9, frameon=False)
    _save(fig, output, "resolution_accuracy.png",
          "Ten held-out pair folds, 5,999 scored pairs per condition. Mixed calibration sees these degradation types in other folds.")


def error_figure(index, output):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.8), sharey=True)
    fig.subplots_adjust(left=.255, right=.975, top=.80, bottom=.15, wspace=.28)
    labels = ("Original calibration\nQMagFace", "Mixed calibration\nCosine", "Mixed calibration\nQMagFace",
              "Mixed calibration\nLearned quality", "Unseen resolution\nLearned quality")
    positions = np.arange(len(SELECTION))[::-1]
    for ax, field, count, denom, title in zip(
            axes, ("pooled_fmr", "pooled_fnmr"), ("false_matches", "false_nonmatches"),
            ("impostor_pairs", "genuine_pairs"), ("False-match rate", "False-nonmatch rate")):
        values = [100 * index[(r, m, "resolution_14")][field] for r, m, _, _ in SELECTION]
        colors = [color for _, _, _, color in SELECTION]
        ax.barh(positions, values, color=colors, height=.57)
        maximum = max(values)
        ax.set_xlim(0, maximum * 1.48)
        for y, value, (regime, method, _, _) in zip(positions, values, SELECTION):
            row = index[(regime, method, "resolution_14")]
            ax.text(value + maximum * .035, y, f"{value:.2f}%\n{row[count]:,} / {row[denom]:,}",
                    va="center", ha="left", fontsize=9, color="#17202a")
        ax.set_title(title, loc="left", fontsize=13, pad=12)
        ax.set_xlabel("Pooled rate (%)")
        ax.set_yticks(positions, labels)
        ax.grid(axis="y", visible=False)
    fig.suptitle("14-pixel errors show the acceptance tradeoff", x=.075, y=.97, ha="left", fontsize=17)
    _save(fig, output, "resolution14_errors.png",
          "Pooled counts and rates across 5,999 scored pairs. Rates condition on successful preprocessing.")


def sensitivity_figure(index, output):
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    fig.subplots_adjust(left=.09, right=.96, top=.86, bottom=.23)
    variants = ((.1, "logistic_interaction_c0.1"), (1., "logistic_interaction"), (10., "logistic_interaction_c10"))
    for regime, label, color in (("mixed", "Mixed calibration", "#008678"),
                                 ("leave_resolution_out", "Unseen resolution", "#916296")):
        values = [100 * index[(regime, method, "resolution_14")]["mean_accuracy"] for _, method in variants]
        ax.plot([c for c, _ in variants], values, marker="o", linewidth=2, color=color, label=label)
        for (c, _), value in zip(variants, values):
            ax.annotate(f"{value:.2f}%", (c, value), xytext=(0, 9), textcoords="offset points",
                        ha="center", color=color, fontsize=11)
    ax.axvline(1, color="#cbd5e1", linestyle="--", zorder=0)
    ax.set_xscale("log")
    ax.set_xticks([.1, 1, 10], ["0.1", "1 (primary)", "10"])
    ax.set_xlabel("C: larger values reduce regularization")
    ax.set_ylabel("Mean fold accuracy at 14 pixels (%)")
    ax.set_ylim(58, 90)
    ax.set_title("Regularization sensitivity", loc="left", pad=14)
    ax.legend(loc="lower right", frameon=False)
    _save(fig, output, "regularization_sensitivity.png",
          "C = 1 was fixed before this experiment. All three settings are reported without choosing a winner on held-out results.")


def raw_parameters(model):
    weights = np.asarray(model["coefficients"]) / np.asarray(model["scaler_scale"])
    intercept = float(model["intercept"] - weights @ np.asarray(model["scaler_mean"]))
    return {name: float(value) for name, value in zip(model["feature_names"], weights)}, intercept


def boundary_figure(models, output):
    learned = next(m for m in models if (m["regime"], m["method"], m["fold"]) ==
                   ("mixed", "logistic_interaction", 0))
    cosine = next(m for m in models if (m["regime"], m["method"], m["fold"]) ==
                  ("mixed", "cosine_threshold", 0))
    w, intercept = raw_parameters(learned)
    q = np.linspace(*learned["quality_range_train"], 200)
    slopes = w["cosine_similarity"] + w["cosine_x_quality"] * q
    if np.any(slopes <= 0):
        raise ValueError("The selected illustrative model has a nonpositive cosine slope in training support")
    boundary = -(intercept + w["minimum_quality"] * q) / slopes
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    fig.subplots_adjust(left=.1, right=.97, top=.86, bottom=.23)
    ax.plot(q, boundary, color="#008678", linewidth=2.5, label="Learned quality boundary")
    ax.axhline(cosine["threshold"], color="#c36a28", linestyle="--", linewidth=2, label="Recalibrated cosine threshold")
    ax.set_xlabel("Minimum raw MagFace magnitude in the pair")
    ax.set_ylabel("Cosine similarity required for acceptance")
    ax.set_title("Illustrative decision boundaries from fold 0", loc="left", pad=14)
    ax.legend(frameon=False, loc="best")
    _save(fig, output, "decision_boundary.png",
          "Mixed calibration, C = 1. Only this fold's training quality range is shown. Both methods accept above their boundary.")
    return {"regime": "mixed", "method": "logistic_interaction", "fold": 0,
            "raw_weights": w, "raw_intercept": intercept,
            "quality_range_train": learned["quality_range_train"],
            "cosine_threshold": cosine["threshold"],
            "boundary_formula": "-(intercept + w_quality*q) / (w_cosine + w_interaction*q)",
            "minimum_cosine_slope_in_plotted_range": float(slopes.min())}


def quality_diagnostics(models, source_scores):
    """Describe held-out quality ranges without fitting or editing any model."""
    records = [json.loads(line) for line in source_scores.open(encoding="utf-8") if line.strip()]
    by_condition = {}
    for row in records:
        if row["status"] == "scored":
            by_condition.setdefault(row["condition"], []).append(
                (row["fold"], min(row["reference_quality"], row["probe_quality"])))
    arrays = {name: np.asarray(values, dtype=float) for name, values in by_condition.items()}
    diagnostics, combined = [], {}
    for model in models:
        if model["score_key"] is not None:
            continue
        weights, _ = raw_parameters(model)
        low, high = model["quality_range_train"]
        groups = []
        for condition in model["test_conditions"]:
            values = arrays[condition]
            q = values[values[:, 0] == model["fold"], 1]
            outside = (q < low) | (q > high)
            slopes = weights["cosine_similarity"] + weights.get("cosine_x_quality", 0) * q
            row = {"condition": condition, "scored_pair_variants": len(q),
                   "minimum_quality": float(q.min()), "maximum_quality": float(q.max()),
                   "outside_training_quality_range": int(outside.sum()),
                   "fraction_outside_training_quality_range": float(outside.mean()),
                   "minimum_effective_cosine_slope": float(slopes.min()),
                   "maximum_effective_cosine_slope": float(slopes.max()),
                   "nonpositive_cosine_slope_count": int(np.sum(slopes <= 0))}
            groups.append(row)
            key = (model["regime"], model["method"], condition)
            aggregate = combined.setdefault(key, {"regime": key[0], "method": key[1], "condition": condition,
                "scored_pair_variants": 0, "outside_training_quality_range": 0,
                "nonpositive_cosine_slope_count": 0,
                "minimum_effective_cosine_slope": float("inf"),
                "maximum_effective_cosine_slope": float("-inf")})
            for name in ("scored_pair_variants", "outside_training_quality_range", "nonpositive_cosine_slope_count"):
                aggregate[name] += row[name]
            aggregate["minimum_effective_cosine_slope"] = min(aggregate["minimum_effective_cosine_slope"], row["minimum_effective_cosine_slope"])
            aggregate["maximum_effective_cosine_slope"] = max(aggregate["maximum_effective_cosine_slope"], row["maximum_effective_cosine_slope"])
        diagnostics.append({"regime": model["regime"], "method": model["method"], "fold": model["fold"],
                            "training_quality_range": [low, high], "by_condition": groups})
    for row in combined.values():
        row["fraction_outside_training_quality_range"] = row["outside_training_quality_range"] / row["scored_pair_variants"]
    return {"scope": "Univariate minimum-quality support and local cosine slopes on held-out rows; no refitting or model selection",
            "models": diagnostics, "by_regime_method_condition": list(combined.values())}


def summarize(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output == run or output.is_relative_to(run):
        raise ValueError("Analysis output must be separate from the calibration run")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Choose a new or empty analysis output directory")
    manifest, source_scores = completed_run(run)
    aggregates = read_table(run / "aggregates.csv")
    paired = read_table(run / "paired_changes.csv")
    models = json.loads((run / "models.json").read_text())
    index = {(row["regime"], row["method"], row["condition"]): row for row in aggregates}
    if len(index) != len(aggregates):
        raise ValueError("Duplicate aggregate keys")
    if not manifest["settings"]["sensitivity"]:
        raise ValueError("The complete summary requires the predeclared sensitivity runs")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 17,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .22, "axes.axisbelow": True,
                         "legend.fontsize": 10, "savefig.bbox": "standard"})
    output.mkdir(parents=True, exist_ok=True)
    resolution_figure(index, output)
    error_figure(index, output)
    sensitivity_figure(index, output)
    boundary = boundary_figure(models, output)
    groups = {}
    for row in aggregates:
        groups.setdefault((row["regime"], row["method"]), []).append(row)
    macro = [{"regime": regime, "method": method,
              "conditions": [row["condition"] for row in rows], "n_conditions": len(rows),
              "macro_mean_fold_accuracy": float(np.mean([row["mean_accuracy"] for row in rows])),
              "scope": "Equal weighting of this regime's listed conditions; scopes differ across regimes"}
             for (regime, method), rows in groups.items()]
    comparisons = []
    comparison_specs = (
        ("mixed", "logistic_interaction", "original_only", "qmagface_threshold", "Different training exposure and scoring model"),
        ("mixed", "logistic_interaction", "mixed", "cosine_threshold", "Same calibration rows, different decision rule"),
        ("mixed", "logistic_interaction", "mixed", "qmagface_threshold", "Same calibration rows, different decision rule"),
        ("mixed", "logistic_interaction", "mixed", "logistic_score", "Quality-feature ablation with the same fitted model family"),
        ("mixed", "logistic_interaction", "mixed", "logistic_quality", "Interaction ablation with the same training rows"),
        ("leave_resolution_out", "logistic_interaction", "mixed", "logistic_interaction", "Resolution exposure ablation"),
    )
    for regime, method, reference_regime, reference_method, scope in comparison_specs:
        current = index[(regime, method, "resolution_14")]
        reference = index[(reference_regime, reference_method, "resolution_14")]
        comparisons.append({"condition": "resolution_14", "regime": regime, "method": method,
                            "reference_regime": reference_regime, "reference_method": reference_method,
                            "scope": scope,
                            "mean_accuracy_difference_percentage_points": 100 * (current["mean_accuracy"] - reference["mean_accuracy"]),
                            "net_correct_decisions": (current["true_accepts"] + current["true_rejects"] -
                                                      reference["true_accepts"] - reference["true_rejects"]),
                            "false_match_difference": current["false_matches"] - reference["false_matches"],
                            "false_nonmatch_difference": current["false_nonmatches"] - reference["false_nonmatches"]})
    summary = {
        "source": {"run_manifest_sha256": sha256(run / "manifest.json"),
                   "run_output_sha256": manifest["output_sha256"],
                   "pair_scores_sha256": manifest["source"]["pair_scores_sha256"],
                   "summary_script_sha256": sha256(__file__)},
        "metric_conventions": {"primary_accuracy": "unweighted mean of ten held-out pair-fold accuracies",
                               "error_plot_rates": "pooled confusion-count rates, with explicit denominators",
                               "auc_eer": "means of metrics computed within each held-out fold",
                               "sensitivity": "C=1 primary; C=0.1 and C=10 reported without held-out selection"},
        "aggregates": aggregates,
        "selected_resolution14": [index[(regime, method, "resolution_14")] for regime, method, _, _ in SELECTION],
        "sensitivity_resolution14": [index[(regime, method, "resolution_14")]
            for regime in ("mixed", "leave_resolution_out")
            for method in ("logistic_interaction_c0.1", "logistic_interaction", "logistic_interaction_c10")],
        "comparisons_resolution14": comparisons,
        "paired_resolution14": [row for row in paired if row["condition"] == "resolution_14"],
        "macro_accuracy_by_regime": macro,
        "illustrative_fold0_boundary": boundary,
        "quality_diagnostics": quality_diagnostics(models, source_scores),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    analysis_manifest = {"status": "completed", "source": summary["source"],
                         "output_sha256": {path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file()}}
    (output / "analysis_manifest.json").write_text(json.dumps(analysis_manifest, indent=2) + "\n", encoding="utf-8")
    return analysis_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "results/project3/calibration")
    parser.add_argument("--output", type=Path, default=ROOT / "results/project3/analysis")
    args = parser.parse_args()
    manifest = summarize(args.run, args.output)
    print(json.dumps({"status": manifest["status"], "output": str(args.output.resolve()),
                      "files": list(manifest["output_sha256"])}, indent=2))


if __name__ == "__main__":
    main()
