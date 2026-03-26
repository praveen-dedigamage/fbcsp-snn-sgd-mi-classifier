"""Training loop, early stopping, and per-batch metrics for the SNN classifier."""

import copy
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from fbcsp_snn import setup_logger
from fbcsp_snn.losses import create_target_spikes, van_rossum_loss
from fbcsp_snn.model import SNNClassifier

logger = setup_logger(__name__)


# ── Early stopping ─────────────────────────────────────────────────────────────


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Parameters
    ----------
    patience:
        Number of calls without improvement before ``early_stop`` is set.
    min_delta:
        Minimum change that counts as an improvement.
    mode:
        ``'max'`` if higher is better (e.g. accuracy), ``'min'`` otherwise.
    """

    def __init__(
        self,
        patience: int = 25,
        min_delta: float = 1e-4,
        mode: str = "max",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score: Optional[float] = None
        self.counter: int = 0
        self.early_stop: bool = False

    def __call__(self, score: float) -> None:
        if self.best_score is None:
            self.best_score = score
            return

        improved = (
            score >= self.best_score + self.min_delta
            if self.mode == "max"
            else score <= self.best_score - self.min_delta
        )

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# ── Per-step metrics ────────────────────────────────────────────────────────────


def compute_metrics(
    output_spikes: torch.Tensor,
    y_true: torch.Tensor,
    num_classes: int,
    population_per_class: int,
) -> tuple[float, float]:
    """Return accuracy and the fraction of spikes fired by non-target neurons.

    Parameters
    ----------
    output_spikes : Tensor, shape ``(T, batch, total_outputs)``
    y_true : LongTensor, shape ``(batch,)``

    Returns
    -------
    accuracy : float
    incorrect_spike_ratio : float
    """
    batch_size = output_spikes.size(1)
    summed = output_spikes.sum(dim=0)  # (batch, total_outputs)

    class_scores = summed.view(batch_size, num_classes, population_per_class).sum(dim=2)
    preds = class_scores.argmax(dim=1)
    accuracy = float((preds == y_true).sum().item() / batch_size)

    total_spikes = summed.sum().item()
    incorrect_spikes = 0.0
    for i in range(batch_size):
        start = int(y_true[i].item()) * population_per_class
        end = start + population_per_class
        non_target = summed[i].clone()
        non_target[start:end] = 0.0
        incorrect_spikes += non_target.sum().item()

    incorrect_ratio = incorrect_spikes / (total_spikes + 1e-6)
    return accuracy, incorrect_ratio


# ── Training history ────────────────────────────────────────────────────────────


@dataclass
class TrainingHistory:
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    train_accuracies: List[float] = field(default_factory=list)
    val_accuracies: List[float] = field(default_factory=list)
    train_incorrect_ratios: List[float] = field(default_factory=list)
    val_incorrect_ratios: List[float] = field(default_factory=list)


# ── Main training function ──────────────────────────────────────────────────────


def train(
    model: SNNClassifier,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    lr: float = 1e-3,
    epochs: int = 1000,
    weight_decay: float = 1e-2,
    spike_prob: float = 0.7,
    early_stopping_patience: int = 100,
    early_stopping_warmup: int = 100,
    log_every: int = 50,
) -> tuple[SNNClassifier, TrainingHistory]:
    """Train the SNN with Van Rossum supervised loss and return the best model.

    Parameters
    ----------
    model:
        Uninitialised :class:`~fbcsp_snn.model.SNNClassifier` on the correct device.
    X_train, X_val : Tensor, shape ``(T, batch, channels)``
        Pre-encoded spike tensors.
    y_train, y_val : LongTensor, shape ``(batch,)``
        Zero-indexed class labels.
    lr:
        AdamW learning rate.
    epochs:
        Maximum number of training epochs.
    weight_decay:
        AdamW weight-decay (L2 regularisation).
    spike_prob:
        Target neuron firing probability for ideal spike generation.
    early_stopping_patience:
        Epochs without val-accuracy improvement before stopping.
    early_stopping_warmup:
        Epochs to complete before early stopping is active.
    log_every:
        Log training progress every this many epochs.

    Returns
    -------
    best_model : SNNClassifier with weights from the epoch of highest val accuracy.
    history : :class:`TrainingHistory`
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    num_classes = int(y_train.max().item()) + 1
    population_per_class = model.population_per_class
    num_steps_train = X_train.size(0)
    num_steps_val = X_val.size(0)

    train_targets = create_target_spikes(
        y_train, num_classes, population_per_class, num_steps_train, spike_prob
    )
    val_targets = create_target_spikes(
        y_val, num_classes, population_per_class, num_steps_val, spike_prob
    )

    stopper = EarlyStopping(patience=early_stopping_patience, mode="max")
    history = TrainingHistory()
    best_val_acc = 0.0
    best_state: Optional[Dict] = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── Training step ──────────────────────────────────────────────────
        model.train()
        optimizer.zero_grad()
        out_spk, _, _, _ = model(X_train)
        loss = van_rossum_loss(out_spk, train_targets)
        loss.backward()
        optimizer.step()

        train_acc, train_inc = compute_metrics(
            out_spk.detach(), y_train, num_classes, population_per_class
        )

        # ── Validation step ────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_spk, _, _, _ = model(X_val)
            val_loss = van_rossum_loss(val_spk, val_targets)
            val_acc, val_inc = compute_metrics(
                val_spk, y_val, num_classes, population_per_class
            )

        history.train_losses.append(loss.item())
        history.val_losses.append(val_loss.item())
        history.train_accuracies.append(train_acc)
        history.val_accuracies.append(val_acc)
        history.train_incorrect_ratios.append(train_inc)
        history.val_incorrect_ratios.append(val_inc)

        if epoch % log_every == 0:
            logger.info(
                "Epoch %d/%d  train_loss=%.4f  train_acc=%.2f%%  "
                "val_loss=%.4f  val_acc=%.2f%%  (%.2fs)",
                epoch, epochs,
                loss.item(), train_acc * 100,
                val_loss.item(), val_acc * 100,
                time.time() - t0,
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            logger.info("New best val_acc=%.4f at epoch %d", val_acc, epoch)

        if epoch > early_stopping_warmup:
            stopper(val_acc)
            if stopper.early_stop:
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history
