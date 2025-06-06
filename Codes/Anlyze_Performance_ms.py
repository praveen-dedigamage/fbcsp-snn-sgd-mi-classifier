#!/usr/bin/env python3
import os
import glob
import re
import sys
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import confusion_matrix

# -------------------- Settings --------------------
CLASS_NAMES = ["Left Hand", "Right Hand", "Feet", "Tongue"]
SELECTION_THRESHOLD = 0.60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUBJECT_ID = 1

plt.rcParams["figure.dpi"] = 100


# -------------------- Utility Functions --------------------

def find_pth_files(folder_path: str) -> list[str]:
    """
    Recursively find all .pth files under folder_path (including subject subdirs).
    """
    pattern = os.path.join(folder_path, "**", "*.pth")
    files = glob.glob(pattern, recursive=True)
    files.sort()
    return files


def load_checkpoint(file_path: str, device: torch.device) -> dict:
    """
    Load a checkpoint from disk, handling older pickle formats if necessary.
    """
    from numpy.core.multiarray import _reconstruct
    torch.serialization.add_safe_globals({_reconstruct})
    checkpoint = torch.load(file_path, map_location=device, weights_only=False)
    return checkpoint


def extract_metrics(checkpoint: dict) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    float, float, float
]:
    """
    Extract exactly the keys your training script saved:
      - train_losses
      - val_losses
      - train_accuracies
      - val_accuracies
      - train_incorrect_ratios
      - val_incorrect_ratios
      - train_cm
      - val_cm
      - test_cm
      - train_acc
      - val_acc
      - test_acc

    Returns, in order:
      train_losses, val_losses,
      train_accuracies, val_accuracies,
      train_incorrect_ratios, val_incorrect_ratios,
      train_cm, val_cm, test_cm,
      train_acc, val_acc, test_acc
    """
    required = [
        "train_losses", "val_losses",
        "train_accuracies", "val_accuracies",
        "train_incorrect_ratios", "val_incorrect_ratios",
        "train_cm", "val_cm", "test_cm",
        "train_acc", "val_acc", "test_acc"
    ]
    for key in required:
        if key not in checkpoint:
            raise KeyError(f"Checkpoint missing required key: '{key}'")

    train_losses            = np.array(checkpoint["train_losses"])
    val_losses              = np.array(checkpoint["val_losses"])
    train_accuracies        = np.array(checkpoint["train_accuracies"])
    val_accuracies          = np.array(checkpoint["val_accuracies"])
    train_incorrect_ratios  = np.array(checkpoint["train_incorrect_ratios"])
    val_incorrect_ratios    = np.array(checkpoint["val_incorrect_ratios"])
    train_cm                = np.array(checkpoint["train_cm"])
    val_cm                  = np.array(checkpoint["val_cm"])
    test_cm                 = np.array(checkpoint["test_cm"])
    train_acc               = float(checkpoint["train_acc"])
    val_acc                 = float(checkpoint["val_acc"])
    test_acc                = float(checkpoint["test_acc"])

    return (
        train_losses,
        val_losses,
        train_accuracies,
        val_accuracies,
        train_incorrect_ratios,
        val_incorrect_ratios,
        train_cm,
        val_cm,
        test_cm,
        train_acc,
        val_acc,
        test_acc,
    )


def parse_hyperparams_from_filename(filename: str) -> dict[str, str]:
    """
    Parse hyperparameters from a filename of the form:
      Multiple_Subject_model_LR{lr}_FB{fb}_SP{sp}_BT{bt}_AI{ai}_D{d}_HN{hn}_NPC{npc}_TestSub{sub}.pth

    Returns a dict or {} if the pattern doesn’t match.
    """
    base = os.path.basename(filename)
    pattern = (
        r"Multiple_Subject_model_"
        r"LR(?P<LR>[0-9.]+)_"
        r"FB\[(?P<FB>[^\]]+)\]_"
        r"SP(?P<SP>[0-9.]+)_"
        r"BT(?P<BT>[0-9.]+)_"
        r"AI(?P<AI>[0-9.]+)_"
        r"D(?P<D>[0-9.]+)_"
        r"HN(?P<HN>\d+)_"
        r"NPC(?P<NPC>\d+)_"
        r"TestSub(?P<TestSub>\d+)\.pth$"
    )
    m = re.match(pattern, base)
    if not m:
        return {}
    return m.groupdict()


# -------------------- Plotting Functions --------------------

def plot_loss_and_accuracy(
    train_losses: np.ndarray,
    val_losses: np.ndarray,
    train_accuracies: np.ndarray,
    val_accuracies: np.ndarray,
    train_incorrect_ratios: np.ndarray,
    val_incorrect_ratios: np.ndarray,
    title_prefix: str = "",
) -> None:
    """
    Plot three subplots side by side:
      1) Loss (train vs val)
      2) Incorrect‐spike Ratio (train vs val)
      3) Accuracy (train vs val)
    """
    epochs = np.arange(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # — Loss Curve —
    axes[0].plot(epochs, train_losses, label="Train Loss")
    axes[0].plot(epochs, val_losses,   label="Val Loss")
    axes[0].set_title(f"{title_prefix} Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # — Incorrect‐Spike Ratio Curve —
    axes[1].plot(epochs, train_incorrect_ratios, label="Train Incorrect Ratio")
    axes[1].plot(epochs, val_incorrect_ratios,   label="Val Incorrect Ratio")
    axes[1].set_title(f"{title_prefix} Incorrect Spike Ratios")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Ratio")
    axes[1].legend()
    axes[1].grid(True)

    # — Accuracy Curve —
    axes[2].plot(epochs, train_accuracies, label="Train Acc")
    axes[2].plot(epochs, val_accuracies,   label="Val Acc")
    axes[2].set_title(f"{title_prefix} Accuracy Curve")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(
    train_cm: np.ndarray,
    val_cm: np.ndarray,
    test_cm: np.ndarray,
    class_names: list[str],
    title_suffix: str = "",
) -> None:
    """
    Plot train vs. val vs. test confusion matrices side by side.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    sns.heatmap(
        train_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0],
    )
    axes[0].set_title(f"Train CM {title_suffix}")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(
        val_cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1],
    )
    axes[1].set_title(f"Val CM {title_suffix}")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    sns.heatmap(
        test_cm,
        annot=True,
        fmt="d",
        cmap="Oranges",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[2],
    )
    axes[2].set_title(f"Test CM {title_suffix}")
    axes[2].set_xlabel("Predicted")
    axes[2].set_ylabel("True")

    plt.tight_layout()
    plt.show()


# -------------------- Main Workflow --------------------

if __name__ == "__main__":
    # Determine the base folder to scan
    if SUBJECT_ID is None:
        # No subject filter → search all under loso_models/
        base_folder = "loso_models"
    else:
        base_folder = os.path.join("loso_models", f"subject_{SUBJECT_ID}")

    if not os.path.isdir(base_folder):
        print(f"❌ Directory not found: {base_folder}")
        sys.exit(1)

    # 1) Find all .pth files under base_folder
    pth_files = find_pth_files(base_folder)
    if not pth_files:
        print(f"❌ No .pth files found in '{base_folder}'")
        sys.exit(1)

    # 2) Load each checkpoint → extract metrics → filter by test_acc
    selected_models: list[tuple[str, float]] = []
    all_records = []

    for file_path in pth_files:
        print(f"\n📂 Found: {file_path}")

        # Parse hyperparams from filename (optional)
        hparams = parse_hyperparams_from_filename(file_path)
        if hparams:
            lr_str = hparams["LR"]
            fb_str = hparams["FB"]
            print(f"   • Parsed LR = {lr_str}, FB = [{fb_str}]")

        # Load checkpoint
        checkpoint = load_checkpoint(file_path, DEVICE)

        # Extract metrics
        try:
            (
                train_losses,
                val_losses,
                train_accs,
                val_accs,
                train_spike,
                val_spike,
                train_cm,
                val_cm,
                test_cm,
                train_acc,
                val_acc,
                test_acc,
            ) = extract_metrics(checkpoint)
        except KeyError as e:
            print(f"   ⚠️  Skipping '{file_path}'—missing key: {e}")
            continue

        print(f"   ✅ Train  Acc: {train_acc * 100:.2f}%")
        print(f"   ✅ Val    Acc: {val_acc   * 100:.2f}%")
        print(f"   ✅ Test   Acc: {test_acc  * 100:.2f}%")

        # Collect record for CSV
        record = {
            "filepath":             file_path,
            "train_acc":            train_acc,
            "val_acc":              val_acc,
            "test_acc":             test_acc,
            **{f"param_{k}": v for k, v in hparams.items()},
        }
        all_records.append(record)

        # Keep if test_acc > threshold
        if test_acc > SELECTION_THRESHOLD:
            selected_models.append((file_path, test_acc))

    # Save a summary of all records to CSV
    df_all = pd.DataFrame(all_records)
    df_all.to_csv("all_model_records.csv", index=False)
    print("\nℹ️  Wrote all checkpoint summaries to 'all_model_records.csv'.")

    if not selected_models:
        print(f"\n❌ No models exceeded the {SELECTION_THRESHOLD*100:.0f}% threshold.")
        sys.exit(0)

    # 3) Rank the selected models by descending test accuracy
    ranked = sorted(selected_models, key=lambda x: x[1], reverse=True)
    print("\n🏆 Ranked Models (by Test Accuracy):")
    for rank, (path, acc) in enumerate(ranked, start=1):
        print(f"   [Rank {rank:>2}]  Acc = {acc*100:>6.2f}%   |   {path}")

    # 4) For each ranked model, reload and plot metrics + confusion matrices
    for rank, (path, acc) in enumerate(ranked, start=1):
        print(f"\n📈 Rank {rank:>2} | Acc = {acc*100:.2f}% | Path: {path}")
        checkpoint = load_checkpoint(path, DEVICE)

        # Extract again to plot
        (
            tr_losses,
            vl_losses,
            tr_accs,
            vl_accs,
            tr_spike,
            vl_spike,
            tr_cm,
            vl_cm,
            te_cm,
            ta,
            va,
            ta_out,
        ) = extract_metrics(checkpoint)

        title_pref = f"Rank {rank}"
        plot_loss_and_accuracy(
            tr_losses,
            vl_losses,
            tr_accs,
            vl_accs,
            tr_spike,
            vl_spike,
            title_prefix=title_pref,
        )
        plot_confusion_matrices(tr_cm, vl_cm, te_cm, CLASS_NAMES, title_suffix=f"(Rank {rank})")

        input("Press Enter to continue to the next ranked model…")

    print("\n✅ Done. All selected models processed.")
