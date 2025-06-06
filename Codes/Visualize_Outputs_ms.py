import os
import re
import sys
import time
import ast
import h5py
import csv
import pickle
import random
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.serialization

import snntorch as snn
from snntorch import surrogate, spikegen

import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.linalg import eigh
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from itertools import combinations

# Allow safe unpickling of numpy arrays if loading trusted checkpoints
torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])


class SNNClassifier(nn.Module):
    """
    A simple two-layer spiking neural network with Leaky Integrate-and-Fire neurons.
    The second layer has `output_size * population_per_class` output neurons.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int, population_per_class: int = 5):
        super().__init__()
        self.population_per_class = population_per_class
        self.total_outputs = output_size * population_per_class
        beta = 0.95

        # First linear layer + LIF
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        # Second linear layer + LIF
        self.fc2 = nn.Linear(hidden_size, self.total_outputs)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the SNN. Assumes `x` has shape [time_steps, batch_size, input_features].
        Returns stacked output spikes of shape [time_steps, batch_size, total_outputs].
        """
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec = []

        for t in range(x.size(0)):
            h1 = self.fc1(x[t])
            spk1, mem1 = self.lif1(h1, mem1)

            h2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(h2, mem2)

            spk2_rec.append(spk2)

        return torch.stack(spk2_rec)


def parse_parameters_from_filename(filename: str) -> dict:
    """
    Extract training parameters from a checkpoint filename using regex.
    Expected format:
      LR<lr>_FB[<bandlist>]_SP<sp>_BT<base_thresh>_AI<adapt_inc>_D<decay>_HN<hidden_neurons>_NPC<pop_per_class>_Sub<subject_id>.pth
    Example:
      model_LR0.0001_FB[(4, 10), (10, 14), (14, 30)]_SP0.7_BT0.001_AI0.6_D0.95_HN64_NPC20_Sub1.pth
    """
    pattern = (
        r'LR([\de\.\-]+)_FB(\[.*?\])_SP([\de\.\-]+)_'
        r'(?:BaseThres|BT)([\de\.\-]+)_'
        r'(?:AdaptInc|AI)([\de\.\-]+)_'
        r'(?:Decay|D)([\de\.\-]+)_'
        r'HN(\d+)_'
        r'(?:NPPC|NPC)(\d+)_'
        r'(?:SubID|Sub)(\d+)'
    )
    match = re.search(pattern, filename)
    if not match:
        raise ValueError(f"Filename does not match expected pattern: {filename}")

    lr = float(match.group(1))
    freq_bands = eval(match.group(2))
    sp = float(match.group(3))
    base_thresh = float(match.group(4))
    adapt_inc = float(match.group(5))
    decay = float(match.group(6))
    hidden_neurons = int(match.group(7))
    population_per_class = int(match.group(8))
    subject_id = int(match.group(9))

    return {
        'lr': lr,
        'freq_bands': freq_bands,
        'sp': sp,
        'base_thresh': base_thresh,
        'adapt_inc': adapt_inc,
        'decay': decay,
        'hidden_neurons': hidden_neurons,
        'population_per_class': population_per_class,
        'subject_id': subject_id
    }


def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float = 250.0, order: int = 5) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter to multi-trial, multi-channel EEG data.
    - data: shape (n_trials, n_channels, n_timepoints)
    - lowcut, highcut: cutoff frequencies in Hz
    - fs: sampling frequency (default 250 Hz)
    - order: filter order (default 5)
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered = np.zeros_like(data)

    for trial in range(data.shape[0]):
        for ch in range(data.shape[1]):
            signal = data[trial, ch, :]
            filtered[trial, ch, :] = filtfilt(b, a, signal)

    return filtered


class PairwiseCSP:
    """
    Implements Pairwise Common Spatial Patterns (CSP) for multi-class EEG data.
    - n_components: number of spatial filters to keep per pair (default 2)
    - selected_classes: optional list/array of class labels to process; if None, uses all in y
    """
    def __init__(self, n_components: int = 2, selected_classes: np.ndarray = None):
        self.n_components = n_components
        self.selected_classes = selected_classes
        self.pairwise_filters = {}   # dict mapping (class_i, class_j) -> projection matrix W
        self.class_pairs = []        # list of class-pair tuples

    def fit(self, X: np.ndarray, y: np.ndarray, reg_lambda: float = 0.01):
        """
        Fit CSP filters for each pair of classes in `selected_classes` (or all unique classes).
        - X: shape (n_trials, n_channels, n_timepoints)
        - y: shape (n_trials,)
        - reg_lambda: regularization parameter to shrink covariance matrices
        """
        classes = np.unique(y) if self.selected_classes is None else self.selected_classes
        self.class_pairs = list(combinations(classes, 2))

        for cl1, cl2 in self.class_pairs:
            idx = np.where((y == cl1) | (y == cl2))[0]
            X_pair = X[idx]                  # (n_pair_trials, n_channels, n_timepoints)
            y_pair = y[idx]

            # Compute normalized trial covariances
            covs = []
            for trial in X_pair:
                cov = np.cov(trial) / np.trace(np.cov(trial))
                covs.append(cov)
            covs = np.stack(covs)            # (n_pair_trials, n_channels, n_channels)

            cov1 = np.mean(covs[y_pair == cl1], axis=0)
            cov2 = np.mean(covs[y_pair == cl2], axis=0)

            I = np.eye(cov1.shape[0])
            cov1 = (1 - reg_lambda) * cov1 + reg_lambda * I
            cov2 = (1 - reg_lambda) * cov2 + reg_lambda * I

            eigvals, eigvecs = eigh(cov1, cov1 + cov2)
            # Sort eigenvectors by descending eigenvalues
            idxs = np.argsort(eigvals)[::-1]
            W = eigvecs[:, idxs[: self.n_components]]
            self.pairwise_filters[(cl1, cl2)] = W

        return self

    def transform(self, X: np.ndarray) -> dict:
        """
        Project each trial in X onto each pairwise filter set.
        - X: shape (n_trials, n_channels, n_timepoints)
        Returns a dict mapping (cl1, cl2) -> projected_data of shape (n_trials, n_components, n_timepoints).
        """
        projected = {}
        for pair, W in self.pairwise_filters.items():
            # For each trial, compute W^T @ trial_data
            X_proj = np.array([W.T @ trial for trial in X])  # (n_trials, n_components, n_timepoints)
            projected[pair] = X_proj
        return projected

    def visualize_weight_matrices(self, channel_names: list = None, cmap: str = "RdBu_r"):
        """
        Plot the CSP weight matrix W for each class-pair.
        If `n_components` == n_channels, W is square; otherwise, the plot may crop/extend.
        - channel_names: list of length n_channels for axis labels; defaults to indices if None.
        """
        if not self.pairwise_filters:
            raise RuntimeError("Call fit(...) before visualizing.")

        for (cl1, cl2), W_full in self.pairwise_filters.items():
            n_chans, _ = W_full.shape
            labels = (
                channel_names
                if (channel_names is not None and len(channel_names) == n_chans)
                else [str(i) for i in range(n_chans)]
            )

            plt.figure(figsize=(8, 6))
            im = plt.imshow(W_full, aspect="auto", cmap=cmap)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xticks(np.arange(n_chans), labels, rotation=90, fontsize=6)
            plt.yticks(np.arange(n_chans), labels, rotation=0, fontsize=6)
            plt.title(f"CSP Weight Matrix for Pair ({cl1} vs {cl2})")
            plt.xlabel("Component index")
            plt.ylabel("Channel index")
            plt.tight_layout()
            plt.show()

    def visualize_as_network(
        self,
        class_pair: tuple,
        threshold: float = 0.0,
        max_linewidth: float = 2.0,
        figsize: tuple = (8, 8),
    ):
        """
        Draw a bipartite graph representation of the CSP weight matrix for one class-pair.
        - class_pair: tuple of (cl1, cl2) key in self.pairwise_filters
        - threshold: only plot edges where |weight| >= threshold
        - max_linewidth: scale the thickest line to this value
        """
        if class_pair not in self.pairwise_filters:
            raise ValueError(f"No filters for {class_pair}. Did you call fit(...) first?")

        W = self.pairwise_filters[class_pair]
        n_chans = W.shape[0]
        if W.shape[1] != n_chans:
            raise ValueError(f"Expected square W of shape ({n_chans},{n_chans}), got {W.shape}.")

        mask = np.abs(W) >= threshold
        if not np.any(mask):
            raise ValueError(f"No weights >= {threshold} in W for pair {class_pair}.")

        max_w = np.max(np.abs(W[mask]))
        y_input = np.linspace(0, 1, n_chans)
        y_output = np.linspace(0, 1, n_chans)

        plt.figure(figsize=figsize)
        plt.scatter(np.zeros(n_chans), y_input, color="k", s=66, label="Inputs")
        plt.scatter(np.ones(n_chans), y_output, color="k", s=66, label="Outputs")

        for i in range(n_chans):
            for j in range(n_chans):
                w = W[i, j]
                if abs(w) < threshold:
                    continue

                linewidth = (abs(w) / max_w) * max_linewidth
                linestyle = "solid"
                color = "blue" if w < 0 else "red"

                plt.plot([0, 1], [y_input[i], y_output[j]],
                         linewidth=linewidth, linestyle=linestyle,
                         color=color, alpha=0.6)

        plt.axis("off")
        plt.title(f"CSP Network for Pair {class_pair} (threshold={threshold})", fontsize=12)
        plt.tight_layout()
        plt.show()


def encode_projected_signals_to_spikes(
    projected_data: dict,
    base_thresh: float = 0.02,
    adapt_inc: float = 0.04,
    decay: float = 0.95,
    seed: int = None
) -> torch.Tensor:
    """
    Convert CSP-projected continuous signals into spike trains using adaptive thresholding.
    - projected_data: dict mapping (cl1, cl2) -> np.ndarray of shape (n_trials, n_components, n_timepoints)
    Returns a concatenated spike tensor of shape [time_steps, n_trials, total_features].
    """
    if seed is not None:
        torch.manual_seed(seed)

    encoded_list = []

    for pair in sorted(projected_data.keys()):
        data = projected_data[pair]  # (n_trials, n_components, n_timepoints)
        tensor_data = torch.tensor(data, dtype=torch.float32)
        # Rearrange to [T, batch_size, n_components]
        tensor_data = tensor_data.permute(2, 0, 1)  # [time_steps, n_trials, n_components]

        time_steps, batch_size, num_chans = tensor_data.shape
        spikes = torch.zeros_like(tensor_data)
        thresholds = torch.full((batch_size, num_chans), base_thresh)

        for t in range(1, time_steps):
            delta = (tensor_data[t] - tensor_data[t - 1]).abs()
            spike_t = (delta > thresholds).float()
            spikes[t] = spike_t
            thresholds = thresholds * decay + spike_t * adapt_inc

        encoded_list.append(spikes)

    # Concatenate along feature dimension
    return torch.cat(encoded_list, dim=2)  # [time_steps, n_trials, total_features]


def load_eeg_data(subject_id: int, base_dir: str, session: str) -> tuple:
    """
    Load EEG data for a given subject and session.
    - subject_id: integer subject number
    - base_dir: base directory containing 'Dataset'
    - session: 'T' for training or 'E' for evaluation
    Returns:
      X: np.ndarray of shape (n_trials, n_channels, n_timepoints)
      y: np.ndarray of shape (n_trials,)
    """
    filename = f'EEG_python_ready_without_ICA_A0{subject_id}{session}.mat'
    data_path = os.path.join(base_dir, 'Dataset', filename)
    with h5py.File(data_path, 'r') as f:
        X = f['X'][:]  # shape in file: (channels, time, trials)
        y = f['y'][:].flatten()
    # Transpose to (n_trials, n_channels, n_timepoints)
    X = np.transpose(X, (2, 0, 1))
    return X, y


def preprocess_eeg_data(
    X: np.ndarray,
    freq_bands: list
) -> np.ndarray:
    """
    Bandpass-filter the EEG data for each frequency band, then concatenate along channel axis.
    - X: np.ndarray of shape (n_trials, n_channels, n_timepoints)
    - freq_bands: list of (low, high) tuples
    Returns:
      X_filtered: np.ndarray of shape (n_trials, n_channels * len(freq_bands), n_timepoints)
    """
    filtered_bands = [bandpass_filter(X, low, high) for (low, high) in freq_bands]
    return np.concatenate(filtered_bands, axis=1)  # concatenate along channel dimension


def fit_and_visualize_csp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    freq_bands: list,
    reg_lambda: float,
    n_components: int
) -> tuple:
    """
    Fit PairwiseCSP on training data, visualize weight matrices and networks,
    then transform validation data.
    Returns:
      csp: fitted PairwiseCSP instance
      projected_val: dict mapping class-pair -> projecting array of shape (n_val_trials, n_components, n_timepoints)
    """
    # Concatenate filtered channels across all bands
    X_train_filtered = preprocess_eeg_data(X_train, freq_bands)
    X_val_filtered = preprocess_eeg_data(X_val, freq_bands)

    unique_classes = np.unique(np.concatenate([y_train, y_val]))
    csp = PairwiseCSP(n_components=n_components, selected_classes=unique_classes)
    csp.fit(X_train_filtered, y_train, reg_lambda=reg_lambda)

    # Visualize the learned filters
    csp.visualize_weight_matrices()
    for pair in csp.class_pairs:
        csp.visualize_as_network(pair, threshold=20, figsize=(6, 6), max_linewidth=2.5)

    # Project validation data
    projected_val = csp.transform(X_val_filtered)
    return csp, projected_val


def run_trial_visualizations(
    spike_train_val: torch.Tensor,
    y_val: np.ndarray,
    model: SNNClassifier,
    params: dict
):
    """
    For each trial in `spike_train_val`, run it through the model and plot:
     - Input, Hidden, Output rasters
     - Input, Hidden, Output spike-count bar charts
    An interactive prompt is given to continue or break.
    - spike_train_val: tensor [T, n_trials, n_features]
    - y_val: array of true labels (n_trials,)
    - model: trained SNNClassifier
    - params: dict of model/training parameters for title annotation
    """
    n_trials = spike_train_val.shape[1]

    for trial_idx in range(n_trials):
        x = spike_train_val[:, trial_idx : trial_idx + 1, :]  # [T, 1, features]
        mem1 = model.lif1.init_leaky()
        mem2 = model.lif2.init_leaky()
        spk1_list, spk2_list = [], []

        # Run through the network to record hidden and output spikes
        for t in range(x.size(0)):
            h1 = model.fc1(x[t])
            spk1, mem1 = model.lif1(h1, mem1)
            h2 = model.fc2(spk1)
            spk2, mem2 = model.lif2(h2, mem2)
            spk1_list.append(spk1.cpu())
            spk2_list.append(spk2.cpu())

        spk1_tensor = torch.stack(spk1_list)  # [T, 1, hidden_size]
        spk2_tensor = torch.stack(spk2_list)  # [T, 1, total_outputs]

        # Compute spike-counts
        spike_count_input = x.sum(dim=0).squeeze(0).numpy()
        spike_count_hidden = spk1_tensor.sum(dim=0).squeeze(0).detach().numpy()
        spike_count_output = spk2_tensor.sum(dim=0).squeeze(0).detach().numpy()

        # Prepare rasters
        spk_input = x.squeeze(1).cpu().numpy()       # [T, features]
        spk1_raster = spk1_tensor.squeeze(1).detach().cpu().numpy()  # [T, hidden]
        spk2_raster = spk2_tensor.squeeze(1).detach().cpu().numpy()  # [T, outputs]

        # Plotting
        fig = plt.figure(figsize=(16, 12))
        plt.suptitle(f"Trial {trial_idx} - True Label: {y_val[trial_idx]}\nParameters: {params}", fontsize=14)

        # Raster Plots
        ax1 = plt.subplot2grid((3, 2), (0, 0))
        ax1.imshow(spk_input.T, aspect='auto', cmap='Greys', origin='lower')
        ax1.set_title('Input Raster')
        ax1.set_ylabel('Input Feature')

        ax2 = plt.subplot2grid((3, 2), (1, 0))
        ax2.imshow(spk1_raster.T, aspect='auto', cmap='Greys', origin='lower')
        ax2.set_title('Hidden Raster')
        ax2.set_ylabel('Hidden Neuron')

        ax3 = plt.subplot2grid((3, 2), (2, 0))
        ax3.imshow(spk2_raster.T, aspect='auto', cmap='Greys', origin='lower')
        ax3.set_title('Output Raster')
        ax3.set_ylabel('Output Neuron')
        ax3.set_xlabel('Timestep')

        # Spike Count Bar Plots
        ax4 = plt.subplot2grid((3, 2), (0, 1))
        ax4.barh(np.arange(len(spike_count_input)), spike_count_input, color='b')
        ax4.set_title('Input Spike Count')
        ax4.set_ylabel('Feature')
        ax4.set_xlabel('Total Spikes')

        ax5 = plt.subplot2grid((3, 2), (1, 1))
        ax5.barh(np.arange(len(spike_count_hidden)), spike_count_hidden, color='g')
        ax5.set_title('Hidden Spike Count')
        ax5.set_ylabel('Hidden Neuron')
        ax5.set_xlabel('Total Spikes')

        ax6 = plt.subplot2grid((3, 2), (2, 1))
        ax6.barh(np.arange(len(spike_count_output)), spike_count_output, color='r')
        ax6.set_title('Output Spike Count')
        ax6.set_ylabel('Output Neuron')
        ax6.set_xlabel('Total Spikes')

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.show()

        to_continue = input("Press Enter to continue, or 'q' to quit: ")
        if to_continue.lower() == 'q':
            break


def plot_average_spikes_by_label(
    spike_train_val: torch.Tensor,
    y_val: np.ndarray,
    model: SNNClassifier,
    params: dict
):
    """
    For each unique label, compute the average spike activity over all trials with that label:
      - Average input spikes per timestep-feature
      - Average hidden and output spikes via forwarding each trial
    Plot rasters (thresholded) and average spike-count bar charts.
    """
    labels = np.unique(y_val)
    n_trials = spike_train_val.shape[1]

    for label in labels:
        indices = np.where(y_val == label)[0]
        avg_input_spikes = spike_train_val[:, indices, :].mean(dim=1).cpu().numpy()  # [T, features]

        hidden_spikes_accum = []
        output_spikes_accum = []

        # Collect hidden/output spike trains for each trial of this label
        for idx in indices:
            x = spike_train_val[:, idx : idx + 1, :]  # [T, 1, features]
            mem1 = model.lif1.init_leaky()
            mem2 = model.lif2.init_leaky()
            spk1_list, spk2_list = [], []

            for t in range(x.size(0)):
                h1 = model.fc1(x[t])
                spk1, mem1 = model.lif1(h1, mem1)
                h2 = model.fc2(spk1)
                spk2, mem2 = model.lif2(h2, mem2)
                spk1_list.append(spk1.cpu().detach())
                spk2_list.append(spk2.cpu().detach())

            hidden_spikes_accum.append(torch.stack(spk1_list).cpu().numpy())  # [T, 1, hidden]
            output_spikes_accum.append(torch.stack(spk2_list).cpu().numpy())  # [T, 1, outputs]

        # Compute averages across trials for this label
        avg_hidden_spikes = np.mean(np.concatenate(hidden_spikes_accum, axis=1), axis=1)  # [T, hidden]
        avg_output_spikes = np.mean(np.concatenate(output_spikes_accum, axis=1), axis=1)  # [T, outputs]

        # Plotting
        fig = plt.figure(figsize=(16, 12))
        plt.suptitle(f"Average Spike Activity - Label {label}\nParameters: {params}", fontsize=14)

        ax1 = plt.subplot2grid((3, 2), (0, 0))
        ax1.imshow((avg_input_spikes > 0.5).T, aspect='auto', cmap='Greys', origin='lower')
        ax1.set_title('Avg Input Raster (threshold=0.5)')
        ax1.set_ylabel('Feature')

        ax2 = plt.subplot2grid((3, 2), (1, 0))
        ax2.imshow((avg_hidden_spikes > 0.5).T, aspect='auto', cmap='Greys', origin='lower')
        ax2.set_title('Avg Hidden Raster (threshold=0.5)')
        ax2.set_ylabel('Hidden Neuron')

        ax3 = plt.subplot2grid((3, 2), (2, 0))
        ax3.imshow((avg_output_spikes > 0.25).T, aspect='auto', cmap='Greys', origin='lower')
        ax3.set_title('Avg Output Raster (threshold=0.25)')
        ax3.set_ylabel('Output Neuron')
        ax3.set_xlabel('Timestep')

        ax4 = plt.subplot2grid((3, 2), (0, 1))
        ax4.barh(np.arange(avg_input_spikes.shape[1]), avg_input_spikes.sum(axis=0), color='b')
        ax4.set_title('Avg Input Spike Count')
        ax4.set_ylabel('Feature')
        ax4.set_xlabel('Avg Total Spikes')

        ax5 = plt.subplot2grid((3, 2), (1, 1))
        ax5.barh(np.arange(avg_hidden_spikes.shape[1]), avg_hidden_spikes.sum(axis=0), color='g')
        ax5.set_title('Avg Hidden Spike Count')
        ax5.set_ylabel('Hidden Neuron')
        ax5.set_xlabel('Avg Total Spikes')

        ax6 = plt.subplot2grid((3, 2), (2, 1))
        ax6.barh(np.arange(avg_output_spikes.shape[1]), avg_output_spikes.sum(axis=0), color='r')
        ax6.set_title('Avg Output Spike Count')
        ax6.set_ylabel('Output Neuron')
        ax6.set_xlabel('Avg Total Spikes')

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.show()


def plot_grand_average_spikes(
    spike_train_val: torch.Tensor,
    model: SNNClassifier,
    params: dict
):
    """
    Compute and plot the grand average (across all validation trials) spike activity for:
      - Input spikes: directly from spike_train_val
      - Hidden and output spikes: via forwarding each trial
    """
    # Average over all trials for input
    avg_input_spikes = spike_train_val.mean(dim=1).cpu().numpy()  # [T, features]

    n_trials = spike_train_val.shape[1]
    hidden_spikes_accum = []
    output_spikes_accum = []

    for idx in range(n_trials):
        x = spike_train_val[:, idx : idx + 1, :]  # [T, 1, features]
        mem1 = model.lif1.init_leaky()
        mem2 = model.lif2.init_leaky()
        spk1_list, spk2_list = [], []

        for t in range(x.size(0)):
            h1 = model.fc1(x[t])
            spk1, mem1 = model.lif1(h1, mem1)
            h2 = model.fc2(spk1)
            spk2, mem2 = model.lif2(h2, mem2)
            spk1_list.append(spk1.cpu().detach())
            spk2_list.append(spk2.cpu().detach())

        hidden_spikes_accum.append(torch.stack(spk1_list).cpu().numpy())  # [T, 1, hidden]
        output_spikes_accum.append(torch.stack(spk2_list).cpu().numpy())  # [T, 1, outputs]

    avg_hidden_spikes = np.mean(np.concatenate(hidden_spikes_accum, axis=1), axis=1)  # [T, hidden]
    avg_output_spikes = np.mean(np.concatenate(output_spikes_accum, axis=1), axis=1)  # [T, outputs]

    # Plotting
    fig = plt.figure(figsize=(16, 12))
    plt.suptitle(f"Grand Average Spike Activity\nParameters: {params}", fontsize=14)

    ax1 = plt.subplot2grid((3, 2), (0, 0))
    ax1.imshow((avg_input_spikes > 0.5).T, aspect='auto', cmap='Greys', origin='lower')
    ax1.set_title('Grand Avg Input Raster (threshold=0.5)')
    ax1.set_ylabel('Feature')

    ax2 = plt.subplot2grid((3, 2), (1, 0))
    ax2.imshow((avg_hidden_spikes > 0.5).T, aspect='auto', cmap='Greys', origin='lower')
    ax2.set_title('Grand Avg Hidden Raster (threshold=0.5)')
    ax2.set_ylabel('Hidden Neuron')

    ax3 = plt.subplot2grid((3, 2), (2, 0))
    ax3.imshow((avg_output_spikes > 0.15).T, aspect='auto', cmap='Greys', origin='lower')
    ax3.set_title('Grand Avg Output Raster (threshold=0.2)')
    ax3.set_ylabel('Output Neuron')
    ax3.set_xlabel('Timestep')

    ax4 = plt.subplot2grid((3, 2), (0, 1))
    ax4.barh(np.arange(avg_input_spikes.shape[1]), avg_input_spikes.sum(axis=0), color='b')
    ax4.set_title('Grand Avg Input Spike Count')
    ax4.set_ylabel('Feature')
    ax4.set_xlabel('Avg Total Spikes')

    ax5 = plt.subplot2grid((3, 2), (1, 1))
    ax5.barh(np.arange(avg_hidden_spikes.shape[1]), avg_hidden_spikes.sum(axis=0), color='g')
    ax5.set_title('Grand Avg Hidden Spike Count')
    ax5.set_ylabel('Hidden Neuron')
    ax5.set_xlabel('Avg Total Spikes')

    ax6 = plt.subplot2grid((3, 2), (2, 1))
    ax6.barh(np.arange(avg_output_spikes.shape[1]), avg_output_spikes.sum(axis=0), color='r')
    ax6.set_title('Grand Avg Output Spike Count')
    ax6.set_ylabel('Output Neuron')
    ax6.set_xlabel('Avg Total Spikes')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()
    
def get_sorted_model_paths(codes_dir: str, subject_id: int):
    """
    Returns a list of (checkpoint_path, val_acc) tuples for all .pth files
    in the `Codes/results_<subject_id>` folder, sorted by descending validation accuracy.
    """
    torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
    results_dir = os.path.join(codes_dir, f"loso_models/subject_{subject_id}")
    if not os.path.isdir(results_dir):
        raise RuntimeError(f"No such folder: {results_dir!r}")

    # Gather all .pth files matching *_Sub<subject_id>.pth
    candidates = [
        fn for fn in os.listdir(results_dir)
        if fn.endswith(f"_Sub{subject_id}.pth")
        or fn.endswith(f"_SubID{subject_id}.pth")
        or fn.endswith(f"_TestSub{subject_id}.pth")
    ]
    if not candidates:
        raise RuntimeError(f"No checkpoints ending with _Sub{subject_id}.pth in {results_dir!r}")

    ckpt_acc_list = []
    for fname in candidates:
        full_path = os.path.join(results_dir, fname)
        ckpt = torch.load(full_path, map_location="cpu", weights_only=False)
        
        print(f"\nCheckpoint `{fname}` contains keys:\n", list(ckpt.keys()))

        # Extract validation accuracy
        if "val_acc" in ckpt:
            val_acc = float(ckpt["val_acc"])
        elif "test_acc" in ckpt:
            val_acc = float(ckpt["test_acc"])
        elif "val_cm" in ckpt:
            cm = ckpt["val_cm"]
            correct = float(cm.trace())
            total = float(cm.sum())
            val_acc = correct / total
        elif "test_cm" in ckpt:
            cm = ckpt["test_cm"]
            correct = float(cm.trace())
            total = float(cm.sum())
            val_acc = correct / total
        else:
            raise KeyError(f"Checkpoint {fname!r} has neither 'val_acc' nor 'val_cm'.")

        ckpt_acc_list.append((full_path, val_acc))

    # Sort by accuracy (descending)
    ckpt_acc_list.sort(key=lambda x: x[1], reverse=True)
    return ckpt_acc_list


def main():

    # 1. Determine directories
    script_dir = os.path.dirname(os.path.abspath(__file__))  # .../fbcsp-snn-sgd-mi-classifier/Codes
    repo_root = os.path.dirname(script_dir)                  # .../fbcsp-snn-sgd-mi-classifier
    codes_dir = script_dir                                    # “Codes” is where this script lives

    # 2. Choose which subject’s results folder to scan
    subject_id = 9

    # 3. Find and sort all checkpoints for this subject by val_acc
    sorted_ckpts = get_sorted_model_paths(codes_dir, subject_id)
    for idx, (_, acc) in enumerate(sorted_ckpts):
        print(f"{idx} → val_acc = {acc:.4f}")

    sorted_paths = [p for (p, _) in sorted_ckpts]
    best_model_path = sorted_paths[0]
    print("→ Best checkpoint:", best_model_path)

    # 4. Parse parameters from the chosen filename
    best_filename = os.path.basename(best_model_path)
    params = parse_parameters_from_filename(best_filename)
    freq_bands = params['freq_bands']

    # 5. Load checkpoint & initialize model
    checkpoint = torch.load(best_model_path, weights_only=False, map_location='cpu')
    input_size = checkpoint['model_state_dict']['fc1.weight'].shape[1]
    hidden_size = checkpoint['model_state_dict']['fc1.weight'].shape[0]

    if 'val_cm' in checkpoint:
        cm = checkpoint['val_cm']
    elif 'test_cm' in checkpoint:
        cm = checkpoint['test_cm']
    else:
        raise KeyError("Checkpoint has neither 'val_cm' nor 'test_cm'.")

    output_size = int(cm.shape[0])
    population_per_class = int(
        checkpoint['model_state_dict']['fc2.weight'].shape[0] // output_size
    )

    model = SNNClassifier(input_size, hidden_size, output_size, population_per_class)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 6. Load & preprocess training and validation data
    X_train, y_train = load_eeg_data(subject_id, repo_root, session='T')
    X_val, y_val     = load_eeg_data(subject_id, repo_root, session='E')

    # 7. Fit CSP, visualize, and transform validation data
    reg_lambda  = params['lr']
    n_components = X_train.shape[1] * len(freq_bands)  # total filtered channels
    csp, projected_val = fit_and_visualize_csp(
        X_train, y_train, X_val, y_val,
        freq_bands=freq_bands,
        reg_lambda=reg_lambda,
        n_components=n_components
    )

    # 8. Encode projected validation signals to spikes
    spike_train_val = encode_projected_signals_to_spikes(
        projected_val,
        base_thresh=params['base_thresh'],
        adapt_inc=params['adapt_inc'],
        decay=params['decay']
    )

    # 9. Per-trial visualizations (interactive)
    run_trial_visualizations(spike_train_val, y_val, model, params)

    # 10. Plot average spikes by label
    plot_average_spikes_by_label(spike_train_val, y_val, model, params)

    # 11. Plot grand average spikes across all trials
    plot_grand_average_spikes(spike_train_val, model, params)


if __name__ == '__main__':
    main()
