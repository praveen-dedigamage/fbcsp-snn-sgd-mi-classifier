"""Adaptive threshold spike encoding for projected EEG signals."""

from typing import Dict, Optional, Tuple, Union

import torch

from fbcsp_snn import DEVICE


@torch.jit.script
def _encode_segment(
    signal: torch.Tensor,
    base_thresh: float,
    adapt_inc: float,
    decay: float,
) -> torch.Tensor:
    """JIT-compiled adaptive-threshold encoding for a single segment.

    Compiles the time-step loop to TorchScript, eliminating CPython interpreter
    overhead for each of the T iterations.  All batch and channel operations
    inside the loop remain parallelised on the GPU.

    Parameters
    ----------
    signal : float32 Tensor, shape ``(T, n_trials, n_components)``
        Pre-permuted projected signal.
    base_thresh, adapt_inc, decay:
        Encoding hyperparameters (see :func:`encode_to_spikes`).

    Returns
    -------
    spikes : Tensor of same shape as *signal* with binary values.
    """
    T: int = signal.shape[0]
    n_trials: int = signal.shape[1]
    n_components: int = signal.shape[2]

    spikes = torch.zeros_like(signal)
    thresholds = torch.full(
        (n_trials, n_components), base_thresh,
        dtype=signal.dtype, device=signal.device,
    )

    for t in range(1, T):
        delta = (signal[t] - signal[t - 1]).abs()
        fired = (delta > thresholds).to(signal.dtype)
        spikes[t] = fired
        thresholds = thresholds * decay + fired * adapt_inc

    return spikes


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

    The inner time-step loop is compiled with :func:`torch.jit.script` to
    eliminate Python interpreter overhead while all batch/channel operations
    remain fully parallel on the GPU.

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
        # Transfer numpy array to GPU as a contiguous float32 tensor
        signal = torch.as_tensor(
            projected_data[key], dtype=torch.float32, device=device
        ).permute(2, 0, 1).contiguous()  # (T, trials, components)

        spike_segments.append(_encode_segment(signal, base_thresh, adapt_inc, decay))

    return torch.cat(spike_segments, dim=2)
