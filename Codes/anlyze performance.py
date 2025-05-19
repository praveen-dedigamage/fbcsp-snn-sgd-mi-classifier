import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import glob
import os
from numpy.core.multiarray import _reconstruct
import torch.serialization

# Safe load patch (needed for older pickle formats)
torch.serialization.add_safe_globals({_reconstruct})

# ----------- Settings -----------
folder_path = "model_hist5"
class_names = ["Left Hand", "Right Hand", "Feet", "Tongue"]

# ----------- Find .pth Files -----------
pth_files = glob.glob(os.path.join(folder_path, "*.pth"))
pth_files.sort()

if not pth_files:
    print("❌ No .pth files found in the folder!")
    exit()

# ----------- Plot Confusion Matrices Side-by-Side -----------
def plot_combined_confusion_matrices(train_cm, test_cm, class_names, rank):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(train_cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title(f"Train Confusion Matrix (Rank {rank})")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(test_cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title("Test Confusion Matrix")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.show()

# ----------- Step 1: Select Models Interactively -----------
selected_models_with_acc = []

for pth_file_path in pth_files:
    print(f"\n📂 Loading file: {pth_file_path}")

    checkpoint = torch.load(pth_file_path, weights_only=False)

    train_losses = checkpoint["train_losses"]
    val_losses = checkpoint["val_losses"]
    train_accuracies = checkpoint["train_accuracies"]
    val_accuracies = checkpoint["val_accuracies"]
    train_cm = checkpoint["train_cm"]
    test_cm = checkpoint["test_cm"]
    train_acc = checkpoint.get("train_acc", checkpoint.get("trian_acc"))
    test_acc = checkpoint["test_acc"]

    """
    # Plot Loss and Accuracy Curves
    epochs = np.arange(1, len(train_losses) + 1)
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    axs[0].plot(epochs, train_losses, label="Train Loss")
    axs[0].plot(epochs, val_losses, label="Validation Loss")
    axs[0].set_title("Loss Curve")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(epochs, train_accuracies, label="Train Accuracy")
    axs[1].plot(epochs, val_accuracies, label="Validation Accuracy")
    axs[1].set_title("Accuracy Curve")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

    # Plot Confusion Matrices
    plot_combined_confusion_matrices(train_cm, test_cm, class_names, rank="N/A")
    """

    print(f"✅ Final Train Accuracy: {train_acc * 100:.2f}%")
    print(f"✅ Final Test Accuracy: {test_acc * 100:.2f}%")

    #selected = input("⏩ Enter 'y' if this model should be selected: ")
    #if selected.lower() == 'y':
    #    selected_models_with_acc.append((pth_file_path, test_acc))
    
    #if test_acc*100 > 60.00:
    selected_models_with_acc.append((pth_file_path, test_acc))
    

# ----------- Step 2: Rank Selected Models by Accuracy -----------
indexed_models = [(i, path, acc) for i, (path, acc) in enumerate(selected_models_with_acc)]
ranked_models = sorted(indexed_models, key=lambda x: x[2], reverse=True)
ranked_indices = [idx for idx, _, _ in ranked_models]

print("\n🏆 Ranked Selected Models by Test Accuracy:\n")
for rank, (idx, path, acc) in enumerate(ranked_models, start=1):
    print(f"[Rank {rank}] Index {idx} | Accuracy: {acc * 100:.2f}% | Path: {path}")
    
    if rank == 480:
        input("Press Enter")

# ----------- Step 3: Plot Curves + Confusion Matrices in Ranked Order -----------
for rank, idx in enumerate(ranked_indices, start=1):
    path, test_acc = selected_models_with_acc[idx]
    print(f"\n📈 Rank {rank} | Index {idx} | Accuracy: {test_acc * 100:.2f}%")
    print(f"📂 Path: {path}")

    checkpoint = torch.load(path, weights_only=False)

    train_losses = checkpoint["train_losses"]
    val_losses = checkpoint["val_losses"]
    train_accuracies = checkpoint["train_accuracies"]
    val_accuracies = checkpoint["val_accuracies"]
    train_cm = checkpoint["train_cm"]
    test_cm = checkpoint["test_cm"]

    # Plot Curves
    epochs = np.arange(1, len(train_losses) + 1)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    axs[0].plot(epochs, train_losses, label="Train Loss")
    axs[0].plot(epochs, val_losses, label="Validation Loss")
    axs[0].set_title("Loss Curve")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(epochs, train_accuracies, label="Train Accuracy")
    axs[1].plot(epochs, val_accuracies, label="Validation Accuracy")
    axs[1].set_title("Accuracy Curve")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()
    axs[1].grid(True)

    plt.suptitle(f"Model Rank {rank} | Final Test Accuracy: {test_acc * 100:.2f}%", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # Plot Confusion Matrices
    plot_combined_confusion_matrices(train_cm, test_cm, class_names, rank=rank)
    
    button = input("Press Enter to Continue")
