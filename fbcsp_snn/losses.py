"""Loss functions and target spike generation for SNN training."""

import torch
import torch.nn.functional as F


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

    # Build target column indices: (batch_size, population_per_class)
    pop_range = torch.arange(population_per_class, device=y.device)
    target_cols = (y * population_per_class).unsqueeze(1) + pop_range  # (B, pop)

    # Random firing: (num_steps, batch_size, population_per_class)
    fires = (torch.rand(num_steps, batch_size, population_per_class, device=y.device) < spike_prob).float()

    # Scatter into the correct column positions — no Python loop over batch
    targets.scatter_(2, target_cols.unsqueeze(0).expand(num_steps, -1, -1), fires)

    return targets


def _van_rossum_filter(spikes: torch.Tensor, tau: float, dt: float = 1.0) -> torch.Tensor:
    """Apply a causal exponential (Van Rossum) kernel via FFT-based convolution.

    Replaces the O(T) sequential IIR loop with an O(T log T) parallel FFT
    convolution.  The causal kernel ``h[k] = α·(1−α)^k`` is computed once and
    convolved with the spike trains in the frequency domain.

    Parameters
    ----------
    spikes : Tensor, shape ``(T, batch, neurons)``
        Input spike trains.  Can be float16 (AMP); internally upcast to float32
        for numerical stability.
    tau : float
        Time constant in time-step units.
    dt : float
        Integration step size.

    Returns
    -------
    Tensor of the same shape and dtype as *spikes*.
    """
    T, B, N = spikes.shape
    alpha = dt / tau

    # Upcast to float32 for numerical precision in the filter kernel
    spikes_f32 = spikes.float()

    # Causal IIR kernel: h[k] = alpha * beta^k
    k = torch.arange(T, device=spikes.device, dtype=torch.float32)
    kernel = alpha * (1.0 - alpha) ** k  # (T,)

    # Reshape to (B*N, T) for batched FFT
    x = spikes_f32.permute(1, 2, 0).reshape(B * N, T)

    # Linear (non-circular) convolution via zero-padding to length 2T
    n_fft = 2 * T
    Y = torch.fft.rfft(x, n=n_fft) * torch.fft.rfft(kernel, n=n_fft)
    y = torch.fft.irfft(Y, n=n_fft)[:, :T]  # keep only causal part

    return y.view(B, N, T).permute(2, 0, 1).to(spikes.dtype)


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
