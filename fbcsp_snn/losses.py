"""Loss functions and target spike generation for SNN training."""

import torch


def create_target_spikes(
    y: torch.Tensor,
    num_classes: int,
    population_per_class: int,
    num_steps: int,
    spike_prob: float = 0.7,
) -> torch.Tensor:
    """Generate stochastic population-coded target spike trains.

    Each sample's target class neurons fire independently at *spike_prob*;
    all other neurons remain silent.

    Parameters
    ----------
    y : LongTensor, shape ``(batch_size,)``
        Zero-indexed class labels.
    num_classes:
        Total number of classes.
    population_per_class:
        Output neurons per class.
    num_steps:
        Number of simulation time steps.
    spike_prob:
        Firing probability for target neurons at each time step.

    Returns
    -------
    Tensor of shape ``(num_steps, batch_size, num_classes * population_per_class)``
    with binary spike values.
    """
    batch_size = y.size(0)
    total_outputs = num_classes * population_per_class
    targets = torch.zeros(num_steps, batch_size, total_outputs, device=y.device)

    for i in range(batch_size):
        start = y[i].item() * population_per_class
        end = start + population_per_class
        targets[:, i, start:end] = (
            torch.rand(num_steps, population_per_class, device=y.device) < spike_prob
        ).float()

    return targets


def _van_rossum_filter(spikes: torch.Tensor, tau: float, dt: float = 1.0) -> torch.Tensor:
    """Apply a causal exponential (Van Rossum) kernel to spike trains.

    Parameters
    ----------
    spikes : Tensor, shape ``(T, batch, neurons)``
    tau : float
        Time constant in time-step units.
    dt : float
        Integration step size.
    """
    alpha = dt / tau
    filtered = torch.empty_like(spikes)
    filtered[0] = spikes[0]
    for t in range(1, spikes.size(0)):
        filtered[t] = (1.0 - alpha) * filtered[t - 1] + alpha * spikes[t]
    return filtered


def van_rossum_loss(
    output_spikes: torch.Tensor,
    target_spikes: torch.Tensor,
    tau: float = 20.0,
    dt: float = 1.0,
) -> torch.Tensor:
    """Mean squared error between Van Rossum-filtered output and target spikes.

    Parameters
    ----------
    output_spikes, target_spikes : Tensor, shape ``(T, batch, neurons)``
    tau : float
        Kernel time constant in time-step units.
    dt : float
        Integration step size.
    """
    f_pred = _van_rossum_filter(output_spikes, tau, dt)
    f_target = _van_rossum_filter(target_spikes, tau, dt)
    return torch.mean((f_pred - f_target) ** 2)
