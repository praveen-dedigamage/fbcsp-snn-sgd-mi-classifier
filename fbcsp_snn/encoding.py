"""Adaptive threshold spike encoding for projected EEG signals."""

from typing import Dict, Optional, Tuple, Union

import torch

from fbcsp_snn import DEVICE


def encode_to_spikes(
    projected_data: Dict[Union[Tuple[int, int], int], "np.ndarray"],  # noqa: F821
    base_thresh: float = 0.02,
    adapt_inc: float = 0.04,
    decay: float = 0.95,
    device: torch.device = DEVICE,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Convert projected EEG signals into spike trains via adaptive thresholding.

    For each time step the absolute frame-to-frame delta is compared against a
    per-channel adaptive threshold.  When the delta exceeds the threshold a
    spike is emitted and the threshold is bumped up; otherwise the threshold
    decays exponentially.

    Parameters
    ----------
    projected_data:
        Dict mapping keys (class-pair tuples or class ints) to arrays of shape
        ``(n_trials, n_components, n_time_steps)``.  Keys are sorted before
        concatenation to guarantee a deterministic channel order.
    base_thresh:
        Initial threshold value for all channels.
    adapt_inc:
        Amount added to the threshold after each spike.
    decay:
        Multiplicative decay applied to thresholds every time step.
    device:
        Target PyTorch device.
    seed:
        Optional manual seed for reproducibility.

    Returns
    -------
    Tensor of shape ``(n_time_steps, n_trials, total_channels)`` with binary
    spike values (0 or 1).
    """
    if seed is not None:
        torch.manual_seed(seed)

    spike_segments: list[torch.Tensor] = []

    for key in sorted(projected_data.keys()):
        data = projected_data[key]  # (trials, components, time)
        n_trials, n_components, n_time = data.shape

        # Permute to (time, trials, components) for time-step iteration
        signal = torch.tensor(data, dtype=torch.float32, device=device).permute(2, 0, 1)

        spikes = torch.zeros_like(signal)
        thresholds = torch.full((n_trials, n_components), base_thresh, device=device)

        for t in range(1, n_time):
            delta = (signal[t] - signal[t - 1]).abs()
            fired = (delta > thresholds).float()
            spikes[t] = fired
            thresholds = thresholds * decay + fired * adapt_inc

        spike_segments.append(spikes)

    return torch.cat(spike_segments, dim=2)
