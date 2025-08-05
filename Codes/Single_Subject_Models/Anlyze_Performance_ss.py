import os
import glob
import re

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- Settings --------------------
FOLDER_PATH = "results_2"
CLASS_NAMES = ["Left Hand", "Right Hand", "Feet", "Tongue"]
SELECTION_THRESHOLD = 0.60  # Only keep models with test_acc > 60%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rcParams["figure.dpi"] = 100  # Higher‐resolution plots

# -------------------- “Key‐Alias” Definitions --------------------
# For each metric, list all possible checkpoint‐dict keys (in priority order).
#
# If your first checkpoint format uses e.g.
#   "train_losses", "val_losses", "train_accuracies", etc.
# and your second format uses e.g.
#   "loss_train",     "loss_val",    "acc_train",        etc.
#
# then just add those alternate names into the lists below.
#
# When you get your new checkpoint format, simply insert its key‐names into
# the appropriate lists. The helper function `pick_key(...)` will look through
# them in order and pick the first one it finds in the loaded checkpoint.

METRIC_KEY_ALIASES = {
    "train_losses":                 ["train_losses"],
    "val_losses":                   ["val_losses", "test_losses"],
    "train_accuracies":             ["train_accuracies"],
    "val_accuracies":               ["val_accuracies", "test_accuracies"],
    "train_incorrect_spike_ratios": ["train_incorrect_spike_ratios", "train_incorrect_ratios"],
    "val_incorrect_spike_ratios":   ["val_incorrect_spike_ratios", "val_incorrect_ratios",
                                     "test_incorrect_spike_ratios", "test_incorrect_ratios"],
    "train_cm":                     ["train_cm"],
    "test_cm":                      ["test_cm", "val_cm"],
    "train_acc":                    ["train_acc", "trian_acc"],
    "test_acc":                     ["test_acc", "val_acc"],
}



# -------------------- Utility Functions --------------------

def find_pth_files(folder_path: str) -> list[str]:
    """
    Return a sorted list of all .pth files in the given folder,
    even if their names contain brackets, commas, etc.
    """
    pattern = os.path.join(folder_path, "*.pth")
    files = glob.glob(pattern)
    files.sort()
    return files


def load_checkpoint(file_path: str, device: torch.device) -> dict:
    """
    Load a checkpoint from disk, taking care of older pickle formats.
    """
    from numpy.core.multiarray import _reconstruct
    torch.serialization.add_safe_globals({_reconstruct})
    checkpoint = torch.load(file_path, map_location=device, weights_only=False)
    return checkpoint


def pick_key(checkpoint: dict, possible_keys: list[str]) -> str:
    """
    Given a checkpoint dict and a list of possible key‐names (in preferred order),
    return the first key that actually exists in checkpoint.
    Raise KeyError if none of them are found.
    """
    for k in possible_keys:
        if k in checkpoint:
            return k
    raise KeyError(f"None of keys {possible_keys} found in checkpoint.")


def extract_metrics(checkpoint: dict) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    float, float
]:
    """
    Try to extract, from a single checkpoint dict, all of:
      train_losses, val_losses,
      train_accuracies, val_accuracies,
      train_incorrect_spike_ratios, val_incorrect_spike_ratios,
      train_cm, test_cm,
      train_acc, test_acc

    Uses METRIC_KEY_ALIASES to know which keys to look for. If a particular alias
    list is missing entirely, KeyError will be raised (so be sure to add your new
    checkpoint's key‐names into METRIC_KEY_ALIASES).

    Returns everything as numpy arrays (where appropriate) or floats for accuracies.
    """
    # 1) For each “semantic” metric, pick whichever actual key is present:
    k_train_losses = pick_key(checkpoint, METRIC_KEY_ALIASES["train_losses"])
    k_val_losses   = pick_key(checkpoint, METRIC_KEY_ALIASES["val_losses"])
    k_train_accs   = pick_key(checkpoint, METRIC_KEY_ALIASES["train_accuracies"])
    k_val_accs     = pick_key(checkpoint, METRIC_KEY_ALIASES["val_accuracies"])
    k_train_spike  = pick_key(checkpoint, METRIC_KEY_ALIASES["train_incorrect_spike_ratios"])
    k_val_spike    = pick_key(checkpoint, METRIC_KEY_ALIASES["val_incorrect_spike_ratios"])
    k_train_cm     = pick_key(checkpoint, METRIC_KEY_ALIASES["train_cm"])
    k_test_cm      = pick_key(checkpoint, METRIC_KEY_ALIASES["test_cm"])

    # 2) Train‐acc might be directly stored as “train_acc” or a typo “trian_acc”:
    try:
        k_train_acc_final = pick_key(checkpoint, METRIC_KEY_ALIASES["train_acc"])
    except KeyError:
        # if “train_acc” none found, try exactly “trian_acc”
        k_train_acc_final = pick_key(checkpoint, METRIC_KEY_ALIASES["trian_acc"])

    # 3) Test‐acc:
    k_test_acc_final = pick_key(checkpoint, METRIC_KEY_ALIASES["test_acc"])

    # Now pull them out:
    train_losses = checkpoint[k_train_losses]
    val_losses   = checkpoint[k_val_losses]
    train_accs   = checkpoint[k_train_accs]
    val_accs     = checkpoint[k_val_accs]
    train_spike  = checkpoint[k_train_spike]
    val_spike    = checkpoint[k_val_spike]
    train_cm     = checkpoint[k_train_cm]
    test_cm      = checkpoint[k_test_cm]
    train_acc    = checkpoint[k_train_acc_final]
    test_acc     = checkpoint[k_test_acc_final]

    return (
        np.array(train_losses),
        np.array(val_losses),
        np.array(train_accs),
        np.array(val_accs),
        np.array(train_spike),
        np.array(val_spike),
        np.array(train_cm),
        np.array(test_cm),
        float(train_acc),
        float(test_acc),
    )


def parse_hyperparams_from_filename(filename: str) -> dict[str, str]:
    """
    Attempt to parse something of the form:
      model_LR<lr>_FB[<something>]_SP<sp>_BT<bt>_AI<ai>_D<d>_HN<hn>_NPC<npc>_Sub<sub>.pth

    Example:
      model_LR0.0001_FB[(4, 10), (10, 14), (14, 30)]_SP0.7_BT0.001_AI0.6_D0.95_HN64_NPC20_Sub1.pth

    Returns a dict, e.g.:
      {
        "LR": "0.0001",
        "FB": "[(4, 10), (10, 14), (14, 30)]",
        "SP": "0.7",
        "BT": "0.001",
        "AI": "0.6",
        "D": "0.95",
        "HN": "64",
        "NPC": "20",
        "Sub": "1"
      }
    If it doesn’t match exactly, returns an empty dict.
    """
    base = os.path.basename(filename)
    pattern = (
        r"model_"
        r"LR(?P<LR>[0-9.]+)_"
        r"FB\[(?P<FB>[^\]]+)\]_"
        r"SP(?P<SP>[0-9.]+)_"
        r"BT(?P<BT>[0-9.]+)_"
        r"AI(?P<AI>[0-9.]+)_"
        r"D(?P<D>[0-9.]+)_"
        r"HN(?P<HN>\d+)_"
        r"NPC(?P<NPC>\d+)_"
        r"Sub(?P<Sub>\d+)\.pth$"
    )
    m = re.match(pattern, base)
    if not m:
        return {}
    return m.groupdict()


def plot_loss_and_accuracy(
    train_losses: np.ndarray,
    val_losses: np.ndarray,
    train_accuracies: np.ndarray,
    val_accuracies: np.ndarray,
    train_incorrect_spike_ratios: np.ndarray,
    val_incorrect_spike_ratios: np.ndarray,
    title_prefix: str = "",
) -> None:
    """
    Plot (1) Loss curve, (2) Spike ratio curve, (3) Accuracy curve side by side.
    """
    epochs_loss = np.arange(1, len(train_losses) + 1)
    epochs_spike = np.arange(1, len(train_incorrect_spike_ratios) + 1)
    epochs_acc = np.arange(1, len(train_accuracies) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # — Loss Curve —
    axes[0].plot(epochs_loss, train_losses, label="Train Loss")
    axes[0].plot(epochs_loss, val_losses, label="Val Loss")
    axes[0].set_title(f"{title_prefix} Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # — Spike Ratios (Incorrect Spike) Curve —
    axes[1].plot(epochs_spike, train_incorrect_spike_ratios, label="Train Spike Ratio")
    axes[1].plot(epochs_spike, val_incorrect_spike_ratios, label="Val Spike Ratio")
    axes[1].set_title(f"{title_prefix} Spike Ratios")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Ratio")
    axes[1].legend()
    axes[1].grid(True)

    # — Accuracy Curve —
    axes[2].plot(epochs_acc, train_accuracies, label="Train Acc")
    axes[2].plot(epochs_acc, val_accuracies, label="Val Acc")
    axes[2].set_title(f"{title_prefix} Accuracy Curve")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(
    train_cm: np.ndarray,
    test_cm: np.ndarray,
    class_names: list[str],
    title_suffix: str = "",
) -> None:
    """
    Plot train vs. test confusion matrices side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

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
        test_cm,
        annot=True,
        fmt="d",
        cmap="Oranges",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1],
    )
    axes[1].set_title(f"Test CM {title_suffix}")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.show()


# -------------------- Main Workflow --------------------

if __name__ == "__main__":
    # 1) Gather all .pth files
    pth_files = find_pth_files(FOLDER_PATH)
    if not pth_files:
        print(f"❌ No .pth files found in '{FOLDER_PATH}'")

    # 2) Loop: load each .pth → extract metrics (handling both structures)
    selected_models: list[tuple[str, float]] = []
    for file_path in pth_files:
        print(f"\n📂 Found: {file_path}")

        # (Optional) Try to parse hyperparameters out of the filename:
        hparams = parse_hyperparams_from_filename(file_path)
        if hparams:
            # Example: print LR and FB
            lr_str = hparams["LR"]
            fb_str = hparams["FB"]
            print(f"   • Parsed LR = {lr_str}, FB = [{fb_str}]")
            # (You can print the rest if you want)

        # 2a) Load checkpoint, then extract metrics via our “alias‐aware” routine:
        checkpoint = load_checkpoint(file_path, DEVICE)
        try:
            (
                train_losses,
                val_losses,
                train_accuracies,
                val_accuracies,
                train_spike,
                val_spike,
                train_cm,
                test_cm,
                train_acc,
                test_acc,
            ) = extract_metrics(checkpoint)
        except KeyError as e:
            # If any required key isn’t found (in both alias lists), alert and skip:
            print(f"   ⚠️  Skipping '{file_path}'—missing key: {e}")
            continue

        print(f"   🖥️  Device: {DEVICE}")
        print(f"   ✅ Train Acc: {train_acc * 100:.2f}%")
        print(f"   ✅ Test  Acc: {test_acc * 100:.2f}%")

        if test_acc > SELECTION_THRESHOLD:
            selected_models.append((file_path, test_acc))

    if not selected_models:
        print(f"\n❌ No models exceeded the {SELECTION_THRESHOLD*100:.0f}% threshold.")

    # 3) Rank the selected models by descending test accuracy
    ranked = sorted(selected_models, key=lambda x: x[1], reverse=True)
    print("\n🏆 Ranked Models (by Test Accuracy):")
    for rank, (path, acc) in enumerate(ranked, start=1):
        print(f"   [Rank {rank:>2}]  Acc = {acc*100:>6.2f}%   |   {path}")

    # 4) For each ranked model, re‐load and plot curves + confusion matrices
    for rank, (path, acc) in enumerate(ranked, start=1):
        print(f"\n📈 Rank {rank:>2} | Acc = {acc*100:.2f}% | Path: {path}")
        checkpoint = load_checkpoint(path, DEVICE)

        # Extract again (so we can plot)
        (
            tr_losses,
            vl_losses,
            tr_accs,
            vl_accs,
            tr_spike,
            vl_spike,
            tr_cm,
            te_cm,
            ta,
            tta,
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
        plot_confusion_matrices(tr_cm, te_cm, CLASS_NAMES, title_suffix=f"(Rank {rank})")

        input("Press Enter to continue…")
