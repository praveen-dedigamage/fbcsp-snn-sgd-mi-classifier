"""MOABB-based loaders for standard MI-EEG benchmark datasets.

Supported datasets
------------------
BNCI2014_001  — BCI Competition IV 2a. 9 subjects, 4 classes
                (left hand / right hand / feet / tongue), 22 EEG channels,
                250 Hz, 2 sessions (train + eval).  *Recommended default.*

PhysionetMI   — PhysioNet EEG Motor Movement/Imagery. 109 subjects, 4 classes
                (left hand / right hand / feet / hands), 64 EEG channels,
                160 Hz resampled to 250 Hz, 1 session (80/20 split used).

Cho2017       — Cho et al. 2017. 52 subjects, 2 classes
                (left hand / right hand), 64 EEG channels, 512 Hz resampled
                to 250 Hz, 1 session (80/20 split used).

BNCI2015_001  — BNCI Horizon 2020 dataset 1. 12 subjects, 2 classes
                (right hand / feet), 13 EEG channels, 512 Hz resampled to
                250 Hz, 2 sessions.

All datasets are downloaded automatically on first use via MOABB and cached
in ``~/mne_data/``.
"""

from typing import Dict, List, Tuple

import numpy as np

from fbcsp_snn import setup_logger

logger = setup_logger(__name__)

# ── Registry ──────────────────────────────────────────────────────────────────

#: Maps public dataset names → (MOABB class path, n_sessions, default n_classes)
_REGISTRY: Dict[str, dict] = {
    "BNCI2014_001": {
        "module": "moabb.datasets",
        "cls": "BNCI2014_001",
        "n_sessions": 2,
        "default_classes": 4,
        "fs": 250,
        "description": "BCI-IV-2a: 9 subjects, 4-class MI, 22ch, 250Hz",
    },
    "PhysionetMI": {
        "module": "moabb.datasets",
        "cls": "PhysionetMI",
        "n_sessions": 1,
        "default_classes": 4,
        "fs": 160,
        "description": "PhysioNet MI: 109 subjects, 4-class MI, 64ch, 160Hz",
    },
    "Cho2017": {
        "module": "moabb.datasets",
        "cls": "Cho2017",
        "n_sessions": 1,
        "default_classes": 2,
        "fs": 512,
        "description": "Cho2017: 52 subjects, 2-class MI, 64ch, 512Hz",
    },
    "BNCI2015_001": {
        "module": "moabb.datasets",
        "cls": "BNCI2015_001",
        "n_sessions": 2,
        "default_classes": 2,
        "fs": 512,
        "description": "BNCI2015-001: 12 subjects, 2-class MI, 13ch, 512Hz",
    },
}


def list_datasets() -> List[str]:
    """Return names of all supported MOABB datasets."""
    return list(_REGISTRY.keys())


def dataset_info(name: str) -> str:
    """Return a one-line description of a dataset."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list_datasets()}")
    return _REGISTRY[name]["description"]


def _import_moabb_dataset(name: str):
    """Dynamically import and instantiate a MOABB dataset class."""
    try:
        import importlib
        info = _REGISTRY[name]
        mod = importlib.import_module(info["module"])
        return getattr(mod, info["cls"])()
    except ImportError:
        raise ImportError(
            "MOABB is required for dataset integration.\n"
            "Install with:  pip install moabb"
        )


# ── Label normalisation ────────────────────────────────────────────────────────


def _normalise_labels(y_str: np.ndarray, n_classes: int) -> np.ndarray:
    """Map MOABB string labels to consecutive integers starting at 1.

    Classes are assigned in alphabetical order of their string name, so the
    mapping is deterministic across runs and subjects.

    Parameters
    ----------
    y_str : ndarray of str, shape ``(n_trials,)``
    n_classes : int
        Expected number of classes.  Raises :exc:`ValueError` if the actual
        number of unique labels in the data does not match.

    Returns
    -------
    y_int : int64 ndarray, shape ``(n_trials,)``  with values in ``[1, n_classes]``
    """
    unique = sorted(np.unique(y_str))
    if len(unique) != n_classes:
        raise ValueError(
            f"Dataset has {len(unique)} classes {unique} but --n-classes {n_classes} "
            f"was requested.  Pass --n-classes {len(unique)} or omit --n-classes to "
            f"auto-detect from the dataset registry."
        )
    label_map = {lbl: i + 1 for i, lbl in enumerate(unique)}
    logger.info("Label map: %s", label_map)
    return np.array([label_map[lbl] for lbl in y_str], dtype=np.int64)


# ── Public loader ──────────────────────────────────────────────────────────────


def load_moabb_subject(
    dataset_name: str,
    subject_id: int,
    tmin: float = 0.5,
    tmax: float = 3.5,
    n_classes: int = 4,
    resample_hz: float = 250.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Download (if needed) and load one subject from a MOABB dataset.

    Data are downloaded automatically to ``~/mne_data/`` on first call and
    served from the cache on subsequent calls.

    A minimal 1–100 Hz passband is applied by MOABB so that our pipeline's
    bandpass filter (``freq_bands``) is the effective feature-engineering
    filter.

    Parameters
    ----------
    dataset_name : str
        One of :func:`list_datasets`.
    subject_id : int
        Subject number (1-based, must be in the dataset's subject list).
    tmin, tmax : float
        Epoch window in seconds relative to the event cue.
        Default ``(0.5, 3.5)`` gives a 3-second window at 250 Hz → 750 samples,
        matching the ``EEG_restMI_split_3s`` file format used in the original
        pipeline.
    n_classes : int
        Number of MI classes to retain.  MOABB selects the *n_classes* most
        frequent classes automatically.
    resample_hz : float
        Target sampling frequency.  All datasets are resampled to this rate so
        the rest of the pipeline can assume a fixed ``fs = 250 Hz``.

    Returns
    -------
    X_train : ndarray, shape ``(n_train, n_channels, n_samples)``
    y_train : int64 ndarray, shape ``(n_train,)``   labels in ``[1, n_classes]``
    X_eval  : ndarray, shape ``(n_eval,  n_channels, n_samples)``
    y_eval  : int64 ndarray, shape ``(n_eval,)``    labels in ``[1, n_classes]``
    split_info : dict
        Serialisable description of the train/eval split so callers can save
        it alongside model artifacts for reproducibility.  For session-based
        splits: ``{"type": "session", "train_session": ..., "eval_sessions": [...]}``.
        For random splits: ``{"type": "random", "random_state": 42,
        "test_size": 0.2, "eval_indices": [...]}``.

    Notes
    -----
    * Datasets with **2 sessions**: first session → train, second → eval.
    * Datasets with **1 session** : random 80 / 20 stratified split by trial.
    """
    if dataset_name not in _REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list_datasets()}"
        )

    try:
        from moabb.paradigms import MotorImagery
    except ImportError:
        raise ImportError("Install MOABB:  pip install moabb")

    dataset = _import_moabb_dataset(dataset_name)

    # Validate subject
    if subject_id not in dataset.subject_list:
        raise ValueError(
            f"Subject {subject_id} not in {dataset_name}. "
            f"Valid subjects: {dataset.subject_list}"
        )

    # Use a wide passband so our pipeline's bandpass filter is effective
    paradigm = MotorImagery(
        n_classes=n_classes,
        fmin=1.0,
        fmax=100.0,
        tmin=tmin,
        tmax=tmax,
        resample=resample_hz,
    )

    logger.info(
        "Loading %s — subject %d  (tmin=%.1f, tmax=%.1f, n_classes=%d, resample=%.0f Hz)",
        dataset_name, subject_id, tmin, tmax, n_classes, resample_hz,
    )

    X, y_str, meta = paradigm.get_data(dataset, subjects=[subject_id])
    y = _normalise_labels(y_str, n_classes)

    logger.info("Loaded  X=%s  classes=%s", X.shape, np.unique(y))

    # ── Train / eval split ────────────────────────────────────────────────────
    n_sessions = _REGISTRY[dataset_name]["n_sessions"]

    if n_sessions >= 2:
        # Use the alphabetically first session for training, the rest for eval
        sessions = sorted(meta["session"].unique())
        train_mask = meta["session"].values == sessions[0]
        eval_mask = ~train_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_eval, y_eval = X[eval_mask], y[eval_mask]

        split_info: dict = {
            "type": "session",
            "train_session": sessions[0],
            "eval_sessions": sessions[1:],
        }
        logger.info(
            "Session split: train='%s' (%d trials)  eval=%s (%d trials)",
            sessions[0], train_mask.sum(),
            sessions[1:], eval_mask.sum(),
        )
    else:
        # Single session: stratified 80/20 split
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, eval_idx = next(sss.split(X, y))

        X_train, y_train = X[train_idx], y[train_idx]
        X_eval, y_eval = X[eval_idx], y[eval_idx]

        split_info = {
            "type": "random",
            "random_state": 42,
            "test_size": 0.2,
            "eval_indices": eval_idx.tolist(),
        }
        logger.info(
            "Single-session 80/20 split: train=%d trials  eval=%d trials",
            len(train_idx), len(eval_idx),
        )

    return X_train, y_train, X_eval, y_eval, split_info
