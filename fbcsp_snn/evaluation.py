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
    X = X.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    population_per_class = model.population_per_class
    num_classes = model.total_outputs // population_per_class
    use_amp = device.type == "cuda"

    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=use_amp):
        out_spk, _, _, _ = model(X)
        batch_size = out_spk.size(1)
        summed = out_spk.float().sum(dim=0)  # float32 for argmax stability
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
    mean-subtracted average spike count.  Uses fully vectorised GPU tensor
    operations — no CPU data transfer or Python loops over classes.

    Parameters
    ----------
    spikes : Tensor, shape ``(T, batch, n_neurons)``
    labels : LongTensor, shape ``(batch,)``

    Returns
    -------
    importance : Tensor, shape ``(n_neurons,)`` on the same device as *spikes*.
    """
    T, B, N = spikes.shape
    num_classes = int(labels.max().item()) + 1

    # Sum spikes over time for each sample: (B, N)
    spike_sums = spikes.float().sum(dim=0)

    # One-hot class membership: (B, C)
    one_hot = torch.zeros(B, num_classes, device=spikes.device)
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

    # Count of samples per class: (C,)
    class_counts = one_hot.sum(dim=0).clamp(min=1.0)

    # Per-class mean spike count: (C, N)
    # one_hot.T @ spike_sums = (C, B) @ (B, N) → (C, N)
    per_class_means = (one_hot.T @ spike_sums) / class_counts.unsqueeze(1)

    # Discriminative importance = std of per-class means across classes
    importance = per_class_means.std(dim=0)  # (N,)

    return importance
