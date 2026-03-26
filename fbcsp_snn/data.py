"""EEG data loading utilities."""

from pathlib import Path
from typing import Tuple

import h5py
import numpy as np

from fbcsp_snn import setup_logger

logger = setup_logger(__name__)

# Expected filename template inside data_dir
_FILENAME_TEMPLATE = "EEG_restMI_split_3s_A0{subject_id}{session}.mat"


def load_data(
    data_dir: Path,
    subject_id: int,
    session: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load EEG trials and labels from an HDF5 .mat file.

    Parameters
    ----------
    data_dir:
        Directory that contains the dataset files.
    subject_id:
        Subject number (e.g. 1–9).
    session:
        ``'T'`` for training or ``'E'`` for evaluation.

    Returns
    -------
    X : ndarray, shape ``(n_trials, n_channels, n_samples)``
    y : ndarray, shape ``(n_trials,)``
    """
    filename = _FILENAME_TEMPLATE.format(subject_id=subject_id, session=session)
    path = data_dir / filename

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    logger.info("Loading %s", path)
    with h5py.File(path, "r") as f:
        X = np.transpose(f["X"][:], (2, 0, 1))  # → (trials, channels, samples)
        y = f["y"][:].flatten()

    logger.info("Loaded X=%s  y=%s", X.shape, y.shape)
    return X, y
