"""Model evaluation and feature-importance utilities."""

from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix

from fbcsp_snn import DEVICE
from fbcsp_snn.model import SNNClassifier


def evaluate(
    model: SNNClassifier,
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device = DEVICE,
) -> Tuple[float, np.ndarray]:
    """Evaluate classification accuracy and confusion matrix on a dataset.

    Parameters
    ----------
    model:
        Trained :class:`~fbcsp_snn.model.SNNClassifier`.
    X : Tensor, shape ``(T, batch, channels)``
        Pre-encoded spike tensor.
    y : LongTensor, shape ``(batch,)``
        Zero-indexed ground-truth labels.
    device:
        Device to run inference on.

    Returns
    -------
    accuracy : float in ``[0, 1]``
    confusion_matrix : ndarray, shape ``(n_classes, n_classes)``
    """
    model.eval()
    X = X.to(device)
    y = y.to(device)

    population_per_class = model.population_per_class
    num_classes = model.total_outputs // population_per_class

    with torch.no_grad():
        out_spk, _, _, _ = model(X)
        batch_size = out_spk.size(1)
        summed = out_spk.sum(dim=0)  # (batch, total_outputs)
        class_scores = summed.view(batch_size, num_classes, population_per_class).sum(dim=2)
        preds = class_scores.argmax(dim=1)

    labels = list(range(num_classes))
    acc = accuracy_score(y.cpu(), preds.cpu())
    cm = confusion_matrix(y.cpu(), preds.cpu(), labels=labels)
    return acc, cm


def calculate_feature_importance(
    spikes: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute per-neuron discriminative importance scores.

    Importance is the standard deviation across classes of each neuron's
    mean-subtracted average spike count.  A higher score means the neuron's
    activity varies more across classes, making it more discriminative.

    Parameters
    ----------
    spikes : Tensor, shape ``(T, batch, n_neurons)``
    labels : LongTensor, shape ``(batch,)``

    Returns
    -------
    importance : Tensor, shape ``(n_neurons,)`` on the same device as *spikes*.
    """
    spikes_cpu = spikes.detach().cpu()
    labels_np = labels.detach().cpu().numpy()
    unique_classes = np.unique(labels_np)

    per_class_means: list[torch.Tensor] = []
    for cls in unique_classes:
        idx = np.where(labels_np == cls)[0]
        if len(idx) == 0:
            per_class_means.append(torch.zeros(spikes_cpu.shape[2]))
            continue
        class_spikes = spikes_cpu[:, idx, :]
        # Average total spikes per neuron across time and trials
        per_class_means.append(class_spikes.sum(dim=(0, 1)) / len(idx))

    if not per_class_means:
        return torch.zeros(spikes.shape[2], device=spikes.device)

    stacked = torch.stack(per_class_means)           # (n_classes, n_neurons)
    global_mean = stacked.mean(dim=0)                # (n_neurons,)
    deviations = stacked - global_mean               # (n_classes, n_neurons)
    importance = deviations.std(dim=0)               # (n_neurons,)

    return importance.to(spikes.device)
