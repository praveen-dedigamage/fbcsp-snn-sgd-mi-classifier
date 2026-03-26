"""Simulated symmetric INT8 quantization for weights and CSP filters."""

import copy

import numpy as np
import torch

from fbcsp_snn.model import SNNClassifier
from fbcsp_snn.preprocessing import PairwiseCSP

_INT8_MAX = 127.0


def quantize_tensor(weight: torch.Tensor) -> torch.Tensor:
    """Symmetric per-tensor INT8 quantization, dequantized back to float32.

    Computes a scale factor from the absolute maximum weight, maps weights to
    the ``[-127, 127]`` integer range, then maps back to float32.  This
    simulates the precision loss of INT8 inference without changing the dtype.

    Parameters
    ----------
    weight : Tensor
        Any-shape float32 weight tensor.

    Returns
    -------
    Tensor of the same shape and dtype as *weight*.
    """
    abs_max = weight.abs().max()
    if abs_max == 0:
        return weight.clone()
    scale = abs_max / _INT8_MAX
    return weight.div(scale).round().mul(scale)


def quantize_array(weight: np.ndarray) -> np.ndarray:
    """Symmetric per-array INT8 quantization for NumPy arrays.

    Parameters
    ----------
    weight : ndarray
        Any-shape float array.

    Returns
    -------
    ndarray of the same shape and dtype as *weight*.
    """
    abs_max = np.abs(weight).max()
    if abs_max == 0:
        return weight.copy()
    scale = abs_max / _INT8_MAX
    return np.round(weight / scale) * scale


def quantize_model(model: SNNClassifier) -> SNNClassifier:
    """Return a deep copy of *model* with INT8-simulated FC layer weights.

    Parameters
    ----------
    model : SNNClassifier
        Trained model (must be on the target device already).

    Returns
    -------
    SNNClassifier with quantized ``fc1`` and ``fc2`` weights (float32 values,
    INT8 precision).
    """
    q_model = copy.deepcopy(model)
    with torch.no_grad():
        q_model.fc1.weight.data = quantize_tensor(q_model.fc1.weight.data)
        q_model.fc2.weight.data = quantize_tensor(q_model.fc2.weight.data)
    return q_model


def quantize_csp(csp: PairwiseCSP) -> PairwiseCSP:
    """Return a deep copy of *csp* with INT8-simulated spatial filters.

    Parameters
    ----------
    csp : PairwiseCSP
        Fitted CSP object.

    Returns
    -------
    PairwiseCSP with quantized ``pairwise_filters``.
    """
    q_csp = copy.deepcopy(csp)
    q_csp.pairwise_filters = {
        pair: quantize_array(W) for pair, W in q_csp.pairwise_filters.items()
    }
    return q_csp
