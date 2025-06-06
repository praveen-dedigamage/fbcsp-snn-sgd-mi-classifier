import h5py
from scipy.signal import butter, filtfilt
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.linalg import eigh
import logging

class PairwiseCSP:
    def __init__(self, n_components=22, selected_classes=None, verbose=True):
        self.n_components = n_components
        self.selected_classes = selected_classes
        self.pairwise_filters = {}
        self.class_pairs = []
        self.selected_indices = {}

        self.logger = logging.getLogger(f"PairwiseCSP")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
            self.logger.addHandler(handler)

        self.logger.propagate = False
        self.verbose = verbose
        if not self.verbose:
            self.logger.disabled = True

    def _validate_filter(self, W, cl1, cl2):
        if np.isnan(W).any():
            self.logger.warning(f"CSP filter for class pair ({cl1}, {cl2}) contains NaN values.")
        elif np.isinf(W).any():
            self.logger.warning(f"CSP filter for class pair ({cl1}, {cl2}) contains Inf values.")
        elif np.allclose(W, 0):
            self.logger.warning(f"CSP filter for class pair ({cl1}, {cl2}) is all zeros.")
        else:
            self.logger.info(f"CSP filter for class pair ({cl1}, {cl2}) validated successfully.")

    def fit(self, X, y):
        self.class_pairs = list(combinations(
            np.unique(y) if self.selected_classes is None else self.selected_classes, 2
        ))

        for (cl1, cl2) in self.class_pairs:
            idx = np.where((y == cl1) | (y == cl2))[0]
            X_pair = X[idx]
            y_pair = y[idx]
            self.selected_indices[(cl1, cl2)] = idx

            covs = []
            for trial in X_pair:
                cov = np.cov(trial)
                cov /= np.trace(cov)
                covs.append(cov)
            covs = np.stack(covs)

            cov1 = np.mean(covs[y_pair == cl1], axis=0)
            cov2 = np.mean(covs[y_pair == cl2], axis=0)
            cov1 += 1e-10 * np.eye(cov1.shape[0])
            cov2 += 1e-10 * np.eye(cov2.shape[0])

            eigvals, eigvecs = eigh(cov1, cov1 + cov2)
            ix = np.argsort(eigvals)[::-1]
            W = eigvecs[:, ix]

            self._validate_filter(W[:, :self.n_components], cl1, cl2)
            self.pairwise_filters[(cl1, cl2)] = W[:, :self.n_components]

        return self

    def transform(self, X):
        projected = {}
        for (cl1, cl2), W in self.pairwise_filters.items():
            idx = self.selected_indices[(cl1, cl2)]
            X_pair = X[idx]
            X_proj = np.array([W.T @ trial for trial in X_pair])
            projected[(cl1, cl2)] = X_proj
        return projected

def apply_csp_to_test(X_test, y_test, trained_filters, selected_indices):
    X_test_t = np.transpose(X_test, (2, 0, 1))
    projections = {}
    for pair, W in trained_filters.items():
        idx = selected_indices[pair]
        X_selected = X_test_t[idx]
        X_proj = np.array([W.T @ trial for trial in X_selected])
        projections[pair] = X_proj
    return projections

def plot_csp_eeg_style(projected, y, trial_idx=0, class_pair=(1, 2), fs=250, spacing=100):
    proj = projected[class_pair]
    n_components, n_timepoints = proj.shape[1], proj.shape[2]
    time = np.arange(n_timepoints) / fs

    plt.figure(figsize=(14, 0.5 * n_components + 4))
    for i in range(n_components):
        signal = proj[trial_idx, i] + i * spacing
        plt.plot(time, signal, label=f'CSP {i+1}')

    plt.title(f'EEG-Style CSP Projection (Trial {trial_idx}, Label {y[trial_idx]}, Class Pair {class_pair})')
    plt.xlabel('Time (s)')
    plt.yticks(np.arange(0, spacing * n_components, spacing), [f'CSP {i+1}' for i in range(n_components)])
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_eeg_style_raw(signal_data, trial_idx=0, fs=250, spacing=20, channel_names=None):
    n_channels = signal_data.shape[0]
    time = np.arange(signal_data.shape[1]) / fs

    plt.figure(figsize=(14, 0.5 * n_channels + 4))
    for ch in range(n_channels):
        signal = signal_data[ch, :, trial_idx] + ch * spacing
        plt.plot(time, signal, label=channel_names[ch] if channel_names else f'Ch {ch+1}')

    plt.title(f'EEG-Style Plot of Raw/Bandpassed Signals (Trial {trial_idx})')
    plt.xlabel('Time (s)')
    plt.yticks(np.arange(0, spacing * n_channels, spacing), 
               channel_names if channel_names else [f'Ch {i+1}' for i in range(n_channels)])
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def plot_classwise_csp_variance(projected, y, class_pair, title_suffix=''):
    """
    Plots bar chart of mean variance per CSP component, separated by class (within a pair).

    Parameters:
    - projected: dict from PairwiseCSP.transform()
    - y: array of labels
    - class_pair: tuple, e.g., (1, 2)
    - title_suffix: str, optional
    """
    X_proj = projected[class_pair]  # shape: (n_trials, n_components, time)
    
    # Get indices for each class
    cl1, cl2 = class_pair
    idx_cl1 = np.where(y == cl1)[0]
    idx_cl2 = np.where(y == cl2)[0]

    # Variance over time, per trial
    var_proj = np.var(X_proj, axis=2)  # shape: (n_trials, n_components)

    # Mean variance per class
    mean_var_cl1 = np.mean(var_proj[idx_cl1], axis=0)
    mean_var_cl2 = np.mean(var_proj[idx_cl2], axis=0)

    x = np.arange(len(mean_var_cl1))
    width = 0.35

    plt.figure(figsize=(12, 5))
    plt.bar(x - width/2, mean_var_cl1, width, label=f'Class {cl1}', alpha=0.7)
    plt.bar(x + width/2, mean_var_cl2, width, label=f'Class {cl2}', alpha=0.7)

    plt.xlabel("CSP Component Index")
    plt.ylabel("Mean Variance")
    plt.title(f"CSP Component Variance by Class – Pair {class_pair} {title_suffix}")
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
    
def plot_test_csp_variances(projected, y_test, class_pair, title_suffix=''):
    """
    Plot CSP component variances for each test trial, color-coded by class label.

    Parameters:
    - projected: dict of CSP projections
    - y_test: true class labels (shape: n_trials,)
    - class_pair: tuple (class1, class2)
    """
    X_proj = projected[class_pair]  # shape: (n_trials, n_components, time)
    variances = np.var(X_proj, axis=2)  # (n_trials, n_components)
    
    cl1, cl2 = class_pair
    colors = np.array(['C0' if label == cl1 else 'C1' for label in y_test])

    plt.figure(figsize=(12, 6))
    for comp in range(X_proj.shape[1]):
        plt.subplot(1, X_proj.shape[1], comp + 1)
        plt.scatter(np.arange(len(variances)), variances[:, comp], c=colors, alpha=0.7)
        plt.title(f'CSP {comp+1}')
        plt.xlabel('Trial')
        plt.ylabel('Variance')
        plt.grid(True)
    plt.suptitle(f'Test CSP Component Variances – Pair {class_pair} {title_suffix}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def bandpass_filter(data, lowcut, highcut, fs=250, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered = np.zeros_like(data)
    for ch in range(data.shape[0]):
        for trial in range(data.shape[2]):
            filtered[ch, :, trial] = filtfilt(b, a, data[ch, :, trial])
    return filtered

if __name__ == "__main__":
    base_directory = r'C:/Users/USER/Desktop/fbcsp-snn-mi-classifier/fbcsp-snn-mi-classifier'
    relative_directory = r'Dataset'

    with h5py.File(fr'{base_directory}/{relative_directory}/EEG_python_ready_A01T.mat', 'r') as file:
        X_train = file['X'][:]
        y_train = file['y'][:].flatten()

    with h5py.File(fr'{base_directory}/{relative_directory}/EEG_python_ready_A01E.mat', 'r') as file:
        X_test = file['X'][:]
        y_test = file['y'][:].flatten()

    freq_bands = [(4, 30)]
    train_filtered = {band: bandpass_filter(X_train, band[0], band[1]) for band in freq_bands}
    test_filtered = {band: bandpass_filter(X_test, band[0], band[1]) for band in freq_bands}

    X = np.transpose(train_filtered[(4, 30)], (2, 0, 1))
    y = y_train

    csp = PairwiseCSP(n_components=22, selected_classes=[1, 2, 3, 4])
    csp.fit(X, y)
    projected_signals = csp.transform(X)

    X_test_proj = apply_csp_to_test(test_filtered[(4, 30)], y_test, csp.pairwise_filters, csp.selected_indices)

    plot_eeg_style_raw(train_filtered[(4, 30)], trial_idx=0)
    plot_csp_eeg_style(projected_signals, y, trial_idx=0, class_pair=(1, 2))
    plot_classwise_csp_variance(projected_signals, y, class_pair=(1, 2))
    plot_test_csp_variances(X_test_proj, y_test, class_pair=(1, 2))
