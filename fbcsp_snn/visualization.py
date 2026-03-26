"""Visualization utilities for spikes, weights, and performance metrics."""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from snntorch import spikeplot

from fbcsp_snn import setup_logger
from fbcsp_snn.model import SNNClassifier

logger = setup_logger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved plot: %s", path)


# ── Spike raster ──────────────────────────────────────────────────────────────


def plot_spike_propagation(
    input_spikes: torch.Tensor,
    hidden_spikes: torch.Tensor,
    output_spikes: torch.Tensor,
    true_label: int,
    pred_label: int,
    subject_id: int,
    fold: int,
    population_per_class: int,
    save_dir: Path,
    filename_suffix: str = "",
) -> None:
    """Save a three-panel raster plot showing spike propagation through the network."""
    inp = input_spikes.squeeze().cpu()
    hid = hidden_spikes.squeeze().cpu()
    out = output_spikes.squeeze().cpu()

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, dpi=150)
    fig.suptitle(
        f"Spike Propagation — Subject {subject_id}, Fold {fold}\n"
        f"True: {true_label}  Predicted: {pred_label}",
        fontsize=16,
    )

    spikeplot.raster(inp, axes[0], s=2, c="black")
    axes[0].set_title("Input Layer")
    axes[0].set_ylabel("Neuron")

    spikeplot.raster(hid, axes[1], s=2, c="steelblue")
    axes[1].set_title(f"Hidden Layer ({hid.shape[1]} neurons)")
    axes[1].set_ylabel("Neuron")

    spikeplot.raster(out, axes[2], s=2, c="tomato")
    axes[2].set_title(f"Output Layer ({out.shape[1]} neurons)")
    axes[2].set_xlabel("Time step")
    axes[2].set_ylabel("Neuron")

    num_classes = out.shape[1] // population_per_class
    for i in range(1, num_classes):
        axes[2].axhline(i * population_per_class - 0.5, color="gray", ls="--", lw=1)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save(
        fig,
        save_dir / f"spike_propagation_subject{subject_id}_fold{fold}{filename_suffix}.png",
    )


# ── Membrane potential traces ─────────────────────────────────────────────────


def plot_neuron_traces(
    mem_hidden: torch.Tensor,
    mem_output: torch.Tensor,
    spk_hidden: torch.Tensor,
    spk_output: torch.Tensor,
    subject_id: int,
    fold: int,
    save_dir: Path,
    filename_suffix: str = "",
) -> None:
    """Save a two-panel membrane-potential trace plot."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, dpi=150)
    fig.suptitle(f"Membrane Potential Traces — Subject {subject_id}, Fold {fold}", fontsize=16)

    plt.sca(axes[0])
    spikeplot.traces(mem_hidden.squeeze().cpu(), spk=spk_hidden.squeeze().cpu())
    axes[0].set_title("Hidden Layer")
    axes[0].set_ylabel("Membrane Potential")

    plt.sca(axes[1])
    spikeplot.traces(mem_output.squeeze().cpu(), spk=spk_output.squeeze().cpu())
    axes[1].set_title("Output Layer")
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("Membrane Potential")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save(
        fig,
        save_dir / f"neuron_traces_subject{subject_id}_fold{fold}{filename_suffix}.png",
    )


# ── Weight histograms ─────────────────────────────────────────────────────────


def plot_weight_histograms(
    fp32_model: SNNClassifier,
    int8_model: SNNClassifier,
    subject_id: int,
    fold: int,
    save_dir: Path,
) -> None:
    """Save side-by-side weight distribution histograms for FP32 vs INT8 models."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=150)
    fig.suptitle(f"Weight Distributions — Subject {subject_id}, Fold {fold}", fontsize=16)

    for ax, model, title, colors in zip(
        axes,
        [fp32_model, int8_model],
        ["FP32", "Simulated INT8"],
        [("skyblue", "royalblue"), ("salmon", "firebrick")],
    ):
        sns.histplot(model.fc1.weight.data.cpu().flatten(), ax=ax, color=colors[0], label="FC1", kde=True)
        sns.histplot(model.fc2.weight.data.cpu().flatten(), ax=ax, color=colors[1], label="FC2", kde=True)
        ax.set_title(f"{title} Weight Distribution")
        ax.set_xlabel("Weight value")
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, save_dir / f"weight_distribution_subject{subject_id}_fold{fold}.png")


# ── Confusion matrix heatmap ──────────────────────────────────────────────────


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    subject_id: int,
    precision: str,
    save_dir: Path,
) -> None:
    """Save a labelled confusion-matrix heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix ({precision}) — Subject {subject_id}", fontsize=16)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    _save(fig, save_dir / f"confusion_matrix_{precision}_subject{subject_id}.png")


# ── Spike probability heatmap ─────────────────────────────────────────────────


def plot_spike_probability_heatmap(
    spikes: torch.Tensor,
    labels: torch.Tensor,
    subject_id: int,
    fold: int,
    save_dir: Path,
    tag: str = "test",
) -> None:
    """Save per-class average spike-probability heatmaps (one panel per class)."""
    spikes_cpu = spikes.detach().cpu()
    labels_np = labels.detach().cpu().numpy()

    n_time, _, n_neurons = spikes_cpu.shape
    unique_classes = np.unique(labels_np)
    n_classes = len(unique_classes)

    prob_maps = []
    for cls in unique_classes:
        idx = np.where(labels_np == cls)[0]
        if len(idx) == 0:
            prob_maps.append(torch.zeros(n_time, n_neurons))
        else:
            prob_maps.append(spikes_cpu[:, idx, :].mean(dim=1))

    vmax = max(p.max().item() for p in prob_maps) if prob_maps else 0.0

    fig, axes = plt.subplots(n_classes, 1, figsize=(20, 7 * n_classes), sharex=True)
    if n_classes == 1:
        axes = [axes]

    for i, (cls, prob_map) in enumerate(zip(unique_classes, prob_maps)):
        n_trials = int((labels_np == cls).sum())
        im = axes[i].imshow(
            prob_map.T.numpy(), aspect="auto", cmap="inferno", vmin=0, vmax=vmax
        )
        axes[i].set_title(f"Class {int(cls)}  (n={n_trials})", fontsize=14, loc="left")
        axes[i].set_ylabel(f"Neuron ({n_neurons})")
        fig.colorbar(im, ax=axes[i], pad=0.01).set_label("Spike probability")

    axes[-1].set_xlabel(f"Time step (T={n_time})")
    fig.suptitle(
        f"Average Spike Probability by Class [{tag}] — Subject {subject_id}, Fold {fold}",
        fontsize=18,
        y=1.0,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    _save(fig, save_dir / f"spike_prob_{tag}_subject{subject_id}_fold{fold}.png")


# ── Spike count difference from mean ─────────────────────────────────────────


def plot_spike_count_difference(
    spikes: torch.Tensor,
    labels: torch.Tensor,
    subject_id: int,
    fold: int,
    save_dir: Path,
    tag: str = "test",
) -> None:
    """Save per-neuron spike-count difference from the all-class mean."""
    spikes_cpu = spikes.detach().cpu()
    labels_np = labels.detach().cpu().numpy()
    unique_classes = np.unique(labels_np)
    n_neurons = spikes_cpu.shape[2]

    per_class = []
    for cls in unique_classes:
        idx = np.where(labels_np == cls)[0]
        if len(idx) == 0:
            per_class.append(torch.zeros(n_neurons))
        else:
            per_class.append(spikes_cpu[:, idx, :].sum(dim=(0, 1)) / len(idx))

    if not per_class:
        logger.warning("No data for spike count difference plot (%s)", tag)
        return

    baseline = torch.stack(per_class).mean(dim=0)
    cmap = plt.get_cmap("tab10", len(unique_classes))
    neuron_idx = np.arange(n_neurons)

    fig, ax = plt.subplots(figsize=(18, 7), dpi=120)
    for i, (cls, counts) in enumerate(zip(unique_classes, per_class)):
        ax.plot(neuron_idx, (counts - baseline).numpy(), color=cmap(i),
                label=f"Class {int(cls)}", lw=1.5, alpha=0.8)
    ax.axhline(0, color="black", ls="--", lw=1, label="Overall mean")
    ax.set_title(f"Spike Count vs. Mean [{tag}] — Subject {subject_id}, Fold {fold}", fontsize=16)
    ax.set_xlabel(f"Neuron index ({n_neurons})")
    ax.set_ylabel("Spike count – class mean")
    ax.set_xlim(0, n_neurons)
    ax.legend()
    ax.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    _save(fig, save_dir / f"spike_diff_{tag}_subject{subject_id}_fold{fold}.png")


# ── Discriminative importance ─────────────────────────────────────────────────


def plot_discriminative_importance(
    spikes: torch.Tensor,
    labels: torch.Tensor,
    subject_id: int,
    fold: int,
    save_dir: Path,
    tag: str = "test",
    percentile: float = 75.0,
) -> None:
    """Save a bar chart of per-neuron discriminative importance."""
    spikes_cpu = spikes.detach().cpu()
    labels_np = labels.detach().cpu().numpy()
    unique_classes = np.unique(labels_np)
    n_neurons = spikes_cpu.shape[2]

    per_class = []
    for cls in unique_classes:
        idx = np.where(labels_np == cls)[0]
        if len(idx) == 0:
            per_class.append(torch.zeros(n_neurons))
        else:
            per_class.append(spikes_cpu[:, idx, :].sum(dim=(0, 1)) / len(idx))

    if not per_class:
        logger.warning("No data for discriminative importance plot (%s)", tag)
        return

    stacked = torch.stack(per_class)
    importance = (stacked - stacked.mean(dim=0)).std(dim=0).numpy()
    threshold = np.percentile(importance, percentile)

    fig, ax = plt.subplots(figsize=(18, 7), dpi=120)
    ax.bar(np.arange(n_neurons), importance, width=1.0, color="steelblue")
    ax.axhline(threshold, color="red", ls="--", lw=1.5,
               label=f"{percentile:.0f}th percentile ({threshold:.2f})")
    ax.set_title(f"Discriminative Importance [{tag}] — Subject {subject_id}, Fold {fold}", fontsize=16)
    ax.set_xlabel(f"Neuron index ({n_neurons})")
    ax.set_ylabel("Importance (std of class averages)")
    ax.set_xlim(0, n_neurons)
    ax.legend()
    ax.grid(True, ls="--", axis="y", alpha=0.7)
    plt.tight_layout()
    _save(fig, save_dir / f"feature_importance_{tag}_subject{subject_id}_fold{fold}.png")
