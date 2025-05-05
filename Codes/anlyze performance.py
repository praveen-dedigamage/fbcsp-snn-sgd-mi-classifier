import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy.core.multiarray import _reconstruct
import torch.serialization
import glob
import os

# Add safe globals for torch.load if needed
torch.serialization.add_safe_globals({_reconstruct})

# ----------- Automatically Find .pth Files -----------
folder_path = "model_hist"  # Change to your directory if needed
pth_files = glob.glob(os.path.join(folder_path, "*.pth"))
pth_files.sort()  # Optional: sort alphabetically

if not pth_files:
    print("❌ No .pth files found in the folder!")
    exit()

# ----------- Class Names for Confusion Matrix -----------
class_names = ["Left Hand", "Right Hand", "Feet", "Tongue"]

# ----------- Plot Combined Confusion Matrices -----------
def plot_combined_confusion_matrices(train_cm, test_cm, class_names):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(train_cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title("Train Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(test_cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title("Test Confusion Matrix")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.show()
    
selected_models = []

# ----------- Loop Over Checkpoints -----------
for pth_file_path in pth_files:
    print(f"\n📂 Loading file: {pth_file_path}")

    # Load the checkpoint
    checkpoint = torch.load(pth_file_path, weights_only=False)

    # Extract data
    train_losses = checkpoint["train_losses"]
    val_losses = checkpoint["val_losses"]
    train_accuracies = checkpoint["train_accuracies"]
    val_accuracies = checkpoint["val_accuracies"]
    train_cm = checkpoint["train_cm"]
    test_cm = checkpoint["test_cm"]
    train_acc = checkpoint.get("train_acc", checkpoint.get("trian_acc"))  # fallback for typo
    test_acc = checkpoint["test_acc"]

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

    # Print Final Accuracies
    print(f"✅ Final Train Accuracy: {train_acc * 100:.2f}%")
    print(f"✅ Final Test Accuracy: {test_acc * 100:.2f}%")

    # Plot Combined Confusion Matrices
    plot_combined_confusion_matrices(train_cm, test_cm, class_names)

    # Wait for user to continue
    selected = input("\n⏩ Enter 'y' if this can be selected \n")
    
    if selected == 'y':
        
        selected_models.append(pth_file_path)
        
