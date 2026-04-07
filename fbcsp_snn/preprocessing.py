"""EEG preprocessing: bandpass filtering and Common Spatial Pattern (CSP) variants."""

from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, filtfilt

from fbcsp_snn import setup_logger

logger = setup_logger(__name__)


# ── Euclidean Alignment ───────────────────────────────────────────────────────


def euclidean_alignment(X: np.ndarray) -> np.ndarray:
    """Align trial covariances to the identity matrix (Euclidean Alignment).

    Computes the mean covariance **R** across all trials in *X*, then multiplies
    each trial by **R**^{-1/2}, so the mean covariance becomes **I**.  This
    removes inter-session and inter-subject covariance drift, which is the
    primary cause of poor generalisation for "hard" BCI subjects.

    Apply **separately** to training and evaluation sets so that each set is
    aligned to its own mean — do NOT fit on train and apply to eval, as that
    would leak session-level information.

    Parameters
    ----------
    X : ndarray, shape ``(n_trials, n_channels, n_samples)``

    Returns
    -------
    ndarray, same shape as *X*
    """
    covs = np.array([np.cov(trial) for trial in X])   # (n_trials, C, C)
    R = covs.mean(axis=0)                              # mean covariance (C, C)

    # R^{-1/2} via eigendecomposition (eigh guarantees real, symmetric output)
    eigvals, eigvecs = np.linalg.eigh(R)
    inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1e-10))) @ eigvecs.T

    # Apply: X_aligned[i] = R^{-1/2} @ X[i]
    aligned = np.einsum("ij,tjk->tik", inv_sqrt, X)
    logger.debug("Euclidean alignment applied: mean cov condition %.2f → 1.00",
                 float(np.linalg.cond(R)))
    return aligned


# ── Outlier trial rejection ───────────────────────────────────────────────────


def reject_outlier_trials(
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove trials whose peak-to-peak amplitude is a statistical outlier.

    Computes the maximum peak-to-peak amplitude across channels for every
    trial, then discards trials exceeding ``mean + threshold * std``.  This
    removes epochs contaminated by EMG bursts, electrode pops, or large eye
    movements before CSP covariance estimation.

    Parameters
    ----------
    X : ndarray, shape ``(n_trials, n_channels, n_samples)``
    y : ndarray, shape ``(n_trials,)``
    threshold : float
        Number of standard deviations above the mean beyond which a trial is
        rejected.  Typical values: 2.5–4.0.  Set to 0 to disable.

    Returns
    -------
    X_clean, y_clean : ndarrays with rejected trials removed
    """
    if threshold <= 0:
        return X, y

    # Peak-to-peak per trial: max over channels of (max_t - min_t)
    ptp = (X.max(axis=2) - X.min(axis=2)).max(axis=1)   # (n_trials,)
    mu, sigma = ptp.mean(), ptp.std()
    keep = ptp < (mu + threshold * sigma)
    n_removed = int((~keep).sum())

    if n_removed:
        logger.info(
            "Artifact rejection: removed %d/%d trials (threshold=%.1f σ, "
            "ptp_mean=%.2e, ptp_std=%.2e)",
            n_removed, len(X), threshold, mu, sigma,
        )
    else:
        logger.debug("Artifact rejection: no trials removed (threshold=%.1f σ)", threshold)

    return X[keep], y[keep]


# ── Bandpass filter ───────────────────────────────────────────────────────────


def bandpass_filter(
    data: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float = 250.0,
    order: int = 5,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter to EEG trials.

    All trials and channels are filtered in a single :func:`scipy.signal.filtfilt`
    call by reshaping ``(n_trials, n_channels, n_samples)`` to
    ``(n_trials * n_channels, n_samples)``.  This eliminates the nested Python
    loop and lets SciPy/BLAS process every signal in one vectorised pass.

    Parameters
    ----------
    data:
        Array of shape ``(n_trials, n_channels, n_samples)``.
    lowcut, highcut:
        Passband edge frequencies in Hz.
    fs:
        Sampling frequency in Hz.
    order:
        Filter order.

    Returns
    -------
    ndarray of the same shape as *data*.
    """
    nyquist = 0.5 * fs
    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype="band")

    n_trials, n_channels, n_samples = data.shape
    # filtfilt operates along axis=-1 by default; flatten the leading two dims
    # so every trial-channel pair is filtered in one vectorised call.
    flat = data.reshape(n_trials * n_channels, n_samples)
    return filtfilt(b, a, flat).reshape(n_trials, n_channels, n_samples)


# ── Helpers shared by both CSP classes ───────────────────────────────────────


def _normalised_cov(X: np.ndarray, use_ledoit_wolf: bool = False) -> np.ndarray:
    """Return trace-normalised covariance matrices for a set of trials.

    Parameters
    ----------
    X : ndarray, shape ``(n_trials, n_channels, n_samples)``
    use_ledoit_wolf : bool
        When *True*, use the Ledoit-Wolf shrinkage estimator instead of the
        sample covariance.  More robust for low-SNR data or when
        ``n_samples ≈ n_channels``.
    """
    if use_ledoit_wolf:
        from sklearn.covariance import ledoit_wolf
        # ledoit_wolf expects (n_samples, n_features), so transpose each trial
        covs = np.array([ledoit_wolf(trial.T)[0] for trial in X])
    else:
        covs = np.array([np.cov(trial) for trial in X])
    traces = covs.trace(axis1=1, axis2=2)[:, np.newaxis, np.newaxis]
    return covs / traces


def _regularised(cov: np.ndarray, reg_lambda: float) -> np.ndarray:
    return (1 - reg_lambda) * cov + reg_lambda * np.eye(cov.shape[0])


def _csp_filters(cov_a: np.ndarray, cov_b: np.ndarray, n_components: int) -> np.ndarray:
    """Return spatial filters from the generalised eigenvalue problem cov_a W = λ (cov_a + cov_b) W."""
    eigvals, eigvecs = eigh(cov_a, cov_a + cov_b)
    W = eigvecs[:, np.argsort(eigvals)[::-1]]
    return W[:, :n_components]


# ── Pairwise CSP ─────────────────────────────────────────────────────────────


class PairwiseCSP:
    """Fit one set of CSP spatial filters per pair of classes.

    Parameters
    ----------
    n_components:
        Number of spatial filters to retain per class pair.
    selected_classes:
        Subset of class labels to consider; defaults to all unique labels in ``y``.
    reg_lambda:
        Tikhonov regularisation coefficient applied to each covariance matrix.
    use_ledoit_wolf:
        Use Ledoit-Wolf shrinkage covariance instead of the sample covariance.
        Recommended for hard subjects with low SNR.
    """

    def __init__(
        self,
        n_components: int = 2,
        selected_classes: Optional[List[int]] = None,
        reg_lambda: float = 0.01,
        use_ledoit_wolf: bool = False,
    ) -> None:
        self.n_components = n_components
        self.selected_classes = selected_classes
        self.reg_lambda = reg_lambda
        self.use_ledoit_wolf = use_ledoit_wolf
        self.pairwise_filters: Dict[Tuple[int, int], np.ndarray] = {}
        self.class_pairs: List[Tuple[int, int]] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PairwiseCSP":
        """Fit CSP filters for every pair among ``selected_classes``.

        Parameters
        ----------
        X : ndarray, shape ``(n_trials, n_channels, n_samples)``
        y : ndarray, shape ``(n_trials,)``
        """
        classes = (
            np.array(self.selected_classes)
            if self.selected_classes is not None
            else np.unique(y)
        )
        self.class_pairs = list(combinations(classes, 2))

        for cl1, cl2 in self.class_pairs:
            mask = (y == cl1) | (y == cl2)
            X_pair, y_pair = X[mask], y[mask]

            covs = _normalised_cov(X_pair, use_ledoit_wolf=self.use_ledoit_wolf)
            cov1 = _regularised(covs[y_pair == cl1].mean(axis=0), self.reg_lambda)
            cov2 = _regularised(covs[y_pair == cl2].mean(axis=0), self.reg_lambda)

            self.pairwise_filters[(cl1, cl2)] = _csp_filters(cov1, cov2, self.n_components)
            logger.info("Computed CSP for classes %s vs %s", cl1, cl2)

        return self

    def transform(self, X: np.ndarray) -> Dict[Tuple[int, int], np.ndarray]:
        """Project trials onto each pair's CSP subspace.

        Returns
        -------
        dict mapping ``(cl1, cl2)`` → ndarray of shape ``(n_trials, n_components, n_samples)``
        """
        return {
            pair: np.einsum("ij,tjk->tik", W.T, X)
            for pair, W in self.pairwise_filters.items()
        }


# ── Rest-vs-one CSP ───────────────────────────────────────────────────────────


class RestVsOneCSP:
    """Fit one CSP filter set per MI class versus the rest class (label 0).

    Parameters
    ----------
    n_components:
        Number of spatial filters to retain per class.
    reg_lambda:
        Tikhonov regularisation coefficient.
    use_ledoit_wolf:
        Use Ledoit-Wolf shrinkage covariance instead of the sample covariance.
    """

    def __init__(
        self,
        n_components: int = 2,
        reg_lambda: float = 0.01,
        use_ledoit_wolf: bool = False,
    ) -> None:
        self.n_components = n_components
        self.reg_lambda = reg_lambda
        self.use_ledoit_wolf = use_ledoit_wolf
        self.filters: Dict[int, np.ndarray] = {}
        self.classes: List[int] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RestVsOneCSP":
        self.classes = [int(c) for c in np.unique(y) if c != 0]

        for mi_class in self.classes:
            mask = (y == 0) | (y == mi_class)
            X_pair, y_pair = X[mask], y[mask]

            covs = _normalised_cov(X_pair, use_ledoit_wolf=self.use_ledoit_wolf)
            cov_rest = _regularised(covs[y_pair == 0].mean(axis=0), self.reg_lambda)
            cov_mi = _regularised(covs[y_pair == mi_class].mean(axis=0), self.reg_lambda)

            # Maximise MI-class variance; eigenvalue problem: cov_mi W = λ (cov_rest + cov_mi) W
            self.filters[mi_class] = _csp_filters(cov_mi, cov_rest, self.n_components)

        return self

    def transform(self, X: np.ndarray) -> Dict[int, np.ndarray]:
        return {
            mi_class: np.einsum("ij,tjk->tik", W.T, X)
            for mi_class, W in self.filters.items()
        }
