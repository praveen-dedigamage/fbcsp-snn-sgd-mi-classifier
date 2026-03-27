"""Cross-subject results analysis for the FBCSP-SNN pipeline.

Reads per-subject artifact directories produced by ``main.py train`` and
generates a summary table, per-subject accuracy bar chart, and an
aggregated confusion matrix across all subjects.

Usage
-----
    python analyze_results.py                       # Results/ with subjects 1-9
    python analyze_results.py --results-dir Results --subjects 1 2 3 4 5 6 7 8 9
    python analyze_results.py --results-dir Results --subjects 1 2 3 --n-classes 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-subject accuracy summary and confusion-matrix aggregation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results-dir", type=Path, default=Path("Results"))
    p.add_argument(
        "--subjects", type=int, nargs="+", default=list(range(1, 10)),
        metavar="N", help="Subject IDs to include",
    )
    p.add_argument("--n-folds", type=int, default=10)
    p.add_argument("--n-classes", type=int, default=4)
    p.add_argument(
        "--class-names", type=str, nargs="+",
        default=["Feet", "Left Hand", "Right Hand", "Tongue"],
        metavar="NAME",
    )
    return p.parse_args()


# ── Readers ───────────────────────────────────────────────────────────────────


def _read_subject(results_dir: Path, subject_id: int, n_folds: int, n_classes: int):
    """Return (fp32_accs, int8_accs, cm_fp32, cm_int8) for one subject.

    Reads per-fold JSON artifacts; falls back to the summary CSV when a fold
    JSON has no ``metrics`` key (e.g. artifacts from before this analysis
    script was added).
    """
    subj_dir = results_dir / f"Subject_{subject_id}"
    fp32_accs, int8_accs = [], []
    cm_fp32 = np.zeros((n_classes, n_classes), dtype=int)
    cm_int8 = np.zeros((n_classes, n_classes), dtype=int)
    missing = []

    for fold in range(1, n_folds + 1):
        json_path = subj_dir / f"pipeline_params_subject{subject_id}_fold{fold}.json"
        if not json_path.exists():
            missing.append(fold)
            continue
        with open(json_path) as fh:
            params = json.load(fh)
        m = params.get("metrics", {})
        if not m:
            missing.append(fold)
            continue
        fp32_accs.append(m["test_acc_fp32"])
        int8_accs.append(m["test_acc_int8"])
        cm_fp32 += np.array(m["cm_fp32"], dtype=int)
        cm_int8 += np.array(m["cm_int8"], dtype=int)

    if missing:
        print(f"  Subject {subject_id}: missing folds {missing}")

    return np.array(fp32_accs), np.array(int8_accs), cm_fp32, cm_int8


# ── Plots ─────────────────────────────────────────────────────────────────────


def _plot_per_subject_accuracy(
    subjects: list[int],
    fp32_means: list[float],
    fp32_stds: list[float],
    int8_means: list[float],
    int8_stds: list[float],
    out_path: Path,
) -> None:
    x = np.arange(len(subjects))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(subjects) * 1.2), 5))

    ax.bar(x - width / 2, fp32_means, width, yerr=fp32_stds, label="FP32",
           color="steelblue", capsize=4)
    ax.bar(x + width / 2, int8_means, width, yerr=int8_stds, label="INT8 (simulated)",
           color="coral", capsize=4)

    ax.set_xlabel("Subject")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Per-Subject 10-Fold CV Test Accuracy (FBCSP-SNN)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s}" for s in subjects])
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def _plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    title: str,
    out_path: Path,
) -> None:
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        vmin=0, vmax=1, ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = _parse_args()
    results_dir: Path = args.results_dir

    fp32_means, fp32_stds = [], []
    int8_means, int8_stds = [], []
    agg_cm_fp32 = np.zeros((args.n_classes, args.n_classes), dtype=int)
    agg_cm_int8 = np.zeros((args.n_classes, args.n_classes), dtype=int)
    rows = []

    print(f"\nReading results from {results_dir.resolve()}\n")
    print(f"{'Subject':>8}  {'FP32 mean':>10}  {'FP32 std':>9}  {'INT8 mean':>10}  {'INT8 std':>9}  {'Folds':>6}")
    print("-" * 62)

    valid_subjects = []
    for subj in args.subjects:
        fp32, int8, cm_fp32, cm_int8 = _read_subject(
            results_dir, subj, args.n_folds, args.n_classes
        )
        if len(fp32) == 0:
            print(f"  Subject {subj}: no artifacts found — skipping")
            continue

        valid_subjects.append(subj)
        m_fp32, s_fp32 = float(fp32.mean()), float(fp32.std())
        m_int8, s_int8 = float(int8.mean()), float(int8.std())

        fp32_means.append(m_fp32)
        fp32_stds.append(s_fp32)
        int8_means.append(m_int8)
        int8_stds.append(s_int8)
        agg_cm_fp32 += cm_fp32
        agg_cm_int8 += cm_int8

        print(f"  S{subj:>6}  {m_fp32:>10.4f}  {s_fp32:>9.4f}  {m_int8:>10.4f}  {s_int8:>9.4f}  {len(fp32):>6}")
        rows.append({
            "subject": subj,
            "fp32_mean": m_fp32, "fp32_std": s_fp32,
            "int8_mean": m_int8, "int8_std": s_int8,
            "n_folds": len(fp32),
        })

    if not valid_subjects:
        print("No subject data found. Run training first.")
        return

    overall_fp32 = float(np.mean(fp32_means))
    overall_int8 = float(np.mean(int8_means))
    print("-" * 62)
    print(f"  {'Overall':>6}  {overall_fp32:>10.4f}  {'':>9}  {overall_int8:>10.4f}")
    print(f"\n  FP32 overall: {overall_fp32:.2%}    INT8 overall: {overall_int8:.2%}\n")

    # ── Save cross-subject CSV ────────────────────────────────────────────────
    csv_path = results_dir / "cross_subject_summary.csv"
    with open(csv_path, "w") as fh:
        fh.write("subject,fp32_mean,fp32_std,int8_mean,int8_std,n_folds\n")
        for r in rows:
            fh.write(
                f"{r['subject']},{r['fp32_mean']:.6f},{r['fp32_std']:.6f},"
                f"{r['int8_mean']:.6f},{r['int8_std']:.6f},{r['n_folds']}\n"
            )
        fh.write(f"overall,{overall_fp32:.6f},,{overall_int8:.6f},,\n")
    print(f"Saved: {csv_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_per_subject_accuracy(
        valid_subjects, fp32_means, fp32_stds, int8_means, int8_stds,
        results_dir / "per_subject_accuracy.png",
    )
    _plot_confusion_matrix(
        agg_cm_fp32, args.class_names,
        f"Aggregated Confusion Matrix — FP32 ({len(valid_subjects)} subjects)",
        results_dir / "confusion_matrix_aggregated_FP32.png",
    )
    _plot_confusion_matrix(
        agg_cm_int8, args.class_names,
        f"Aggregated Confusion Matrix — INT8 ({len(valid_subjects)} subjects)",
        results_dir / "confusion_matrix_aggregated_INT8.png",
    )


if __name__ == "__main__":
    main()
