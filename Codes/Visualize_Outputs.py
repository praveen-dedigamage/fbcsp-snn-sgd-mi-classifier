import h5py
import numpy as np
import torch
import torch.nn as nn
import pickle
import snntorch as snn
import logging
import random
import os
import csv
import sys
import time
import ast
import re
import matplotlib.pyplot as plt

from snntorch import surrogate, spikegen
from sklearn.metrics import accuracy_score, confusion_matrix
from itertools import combinations
from scipy.signal import butter, filtfilt
from scipy.linalg import eigh

from sklearn.model_selection import StratifiedKFold

# If you trust the checkpoint (e.g., your own file), you can add the needed global for unpickling
import torch.serialization

torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])


# 1. SNNClassifier
class SNNClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, population_per_class=5):
        super().__init__()
        self.population_per_class = population_per_class
        self.total_outputs = output_size * population_per_class
        beta = 0.95
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.fc2 = torch.nn.Linear(hidden_size, self.total_outputs)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec = []
        for step in range(x.size(0)):
            spk1, mem1 = self.lif1(self.fc1(x[step]), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            spk2_rec.append(spk2)
        return torch.stack(spk2_rec)

# 2. Helper: Parse Parameters

def parse_parameters_from_filename(filename):
    pattern = r'LR([\de.-]+)_FB(\[.*?\])_SP([\de.]+)_BaseThres([\de.]+)_AdaptInc([\de.]+)_Decay([\de.]+)_HN(\d+)_NPPC(\d+)_SubID(\d+)'
    match = re.search(pattern, filename)
    if match:
        lr = float(match.group(1))
        fb = eval(match.group(2))
        sp = float(match.group(3))
        base_thresh = float(match.group(4))
        adapt_inc = float(match.group(5))
        decay = float(match.group(6))
        hn = int(match.group(7))
        nppc = int(match.group(8))
        subid = int(match.group(9))
        return dict(
            lr=lr, freq_bands=fb, sp=sp, base_thresh=base_thresh, adapt_inc=adapt_inc,
            decay=decay, hidden_neurons=hn, population_per_class=nppc, subject_id=subid)
    else:
        raise ValueError('Filename does not match expected pattern.')

# 3. Preprocessing Utilities

def bandpass_filter(data, lowcut, highcut, fs=250.0, order=5):
    b, a = butter(order, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype='band')
    filtered = np.zeros_like(data)
    for trial in range(data.shape[0]):
        for ch in range(data.shape[1]):
            signal = data[trial, ch, :]
            filtered[trial, ch, :] = filtfilt(b, a, signal)
    return filtered

class PairwiseCSP:
    def __init__(self, n_components=2, selected_classes=None):
        self.n_components = n_components
        self.selected_classes = selected_classes
        self.pairwise_filters = {}
        self.class_pairs = []

    def fit(self, X, y, reg_lambda=0.01):
        self.class_pairs = list(combinations(np.unique(y) if self.selected_classes is None else self.selected_classes, 2))
        for (cl1, cl2) in self.class_pairs:
            idx = np.where((y == cl1) | (y == cl2))[0]
            X_pair = X[idx]
            y_pair = y[idx]
            covs = [np.cov(trial) / np.trace(np.cov(trial)) for trial in X_pair]
            covs = np.stack(covs)
            cov1 = np.mean(covs[y_pair == cl1], axis=0)
            cov2 = np.mean(covs[y_pair == cl2], axis=0)
            I = np.eye(cov1.shape[0])
            cov1 = (1 - reg_lambda) * cov1 + reg_lambda * I
            cov2 = (1 - reg_lambda) * cov2 + reg_lambda * I
            eigvals, eigvecs = eigh(cov1, cov1 + cov2)
            W = eigvecs[:, np.argsort(eigvals)[::-1]]
            self.pairwise_filters[(cl1, cl2)] = W[:, :self.n_components]
        return self

    def transform(self, X):
        projected = {}
        for (cl1, cl2), W in self.pairwise_filters.items():
            X_proj = np.array([W.T @ trial for trial in X])
            projected[(cl1, cl2)] = X_proj
        return projected

def encode_projected_signals_to_spikes(projected_data_dict, base_thresh=0.02, adapt_inc=0.04, decay=0.95, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    all_encoded = []
    for pair in sorted(projected_data_dict):
        data = projected_data_dict[pair]
        tensor_data = torch.tensor(data).float()
        tensor_data = tensor_data.permute(2, 0, 1)
        time_steps, batch_size, num_channels = tensor_data.shape
        spikes = torch.zeros_like(tensor_data)
        thresholds = torch.full((batch_size, num_channels), base_thresh)
        for t in range(1, time_steps):
            delta = (tensor_data[t] - tensor_data[t - 1]).abs()
            spike_t = (delta > thresholds).float()
            spikes[t] = spike_t
            thresholds = thresholds * decay + spike_t * adapt_inc
        all_encoded.append(spikes)
    return torch.cat(all_encoded, dim=2)

# 4. Paths & Load Model
base_directory = r'/Users/hsprde/Documents/GitHub/fbcsp-snn-sgd-mi-classifier'


results_dir = os.path.join(base_directory, 'Codes/results_9')
filename = 'model_and_history_LR1e-05_FB[(4, 10), (10, 14), (14, 30)]_SP0.6_BaseThres0.1_AdaptInc0.6_Decay0.95_HN64_NPPC20_SubID9.pth'
model_path = os.path.join(results_dir, filename)
params = parse_parameters_from_filename(filename)

checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')
input_size = checkpoint['model_state_dict']['fc1.weight'].shape[1]
hidden_size = checkpoint['model_state_dict']['fc1.weight'].shape[0]
output_size = int(checkpoint['test_cm'].shape[0])
population_per_class = int(checkpoint['model_state_dict']['fc2.weight'].shape[0] // output_size)

model = SNNClassifier(input_size, hidden_size, output_size, population_per_class)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 5. Load and Preprocess Training Data (for CSP fit)
subject_str = str(params['subject_id'])
data_path_train = os.path.join(base_directory, 'Dataset', f'EEG_python_ready_without_ICA_A0{subject_str}T.mat')
with h5py.File(data_path_train, 'r') as file:
    X_train = file['X'][:]
    X_train = np.transpose(X_train, (2, 0, 1))  # (samples, channels, time)
    y_train = file['y'][:].flatten()

X_train_filtered_bands = [bandpass_filter(X_train, low, high) for (low, high) in params['freq_bands']]
X_train_filtered = np.concatenate(X_train_filtered_bands, axis=1)

# 6. Load and Preprocess Validation Data
# Load EEG
data_path_val = os.path.join(base_directory, 'Dataset', f'EEG_python_ready_without_ICA_A0{subject_str}E.mat')
with h5py.File(data_path_val, 'r') as file:
    X_val = file['X'][:]
    X_val = np.transpose(X_val, (2, 0, 1))  # (samples, channels, time)
    y_val = file['y'][:].flatten()

X_val_filtered_bands = [bandpass_filter(X_val, low, high) for (low, high) in params['freq_bands']]
X_val_filtered = np.concatenate(X_val_filtered_bands, axis=1)

# 7. Fit CSP on training, transform validation
unique_classes = np.unique(np.concatenate([y_train, y_val]))
csp = PairwiseCSP(n_components=X_train_filtered.shape[1], selected_classes=unique_classes)
csp.fit(X_train_filtered, y_train, reg_lambda=params['lr'])
projected_val = csp.transform(X_val_filtered)

# 8. Encode validation to spikes
spike_train_val = encode_projected_signals_to_spikes(
    projected_val,
    base_thresh=params['base_thresh'],
    adapt_inc=params['adapt_inc'],
    decay=params['decay']
)

for i in range(spike_train_val.shape[1]):
# 9. Layerwise Activations
    trial_idx = i
    x = spike_train_val[:, trial_idx:trial_idx+1, :]  # Shape: [T, 1, Features]
    mem1 = model.lif1.init_leaky()
    mem2 = model.lif2.init_leaky()
    spk1_list, spk2_list = [], []
    for step in range(x.size(0)):
        h1 = model.fc1(x[step])
        spk1, mem1 = model.lif1(h1, mem1)
        h2 = model.fc2(spk1)
        spk2, mem2 = model.lif2(h2, mem2)
        spk1_list.append(spk1.cpu())
        spk2_list.append(spk2.cpu())
    spk1_tensor = torch.stack(spk1_list)  # [T, 1, hidden_size]
    spk2_tensor = torch.stack(spk2_list)  # [T, 1, output_size * population_per_class]
    
    spike_count_input = x.sum(dim=0).squeeze(0).numpy()
    spike_count_hidden = spk1_tensor.sum(dim=0).squeeze(0).detach().numpy()
    spike_count_output = spk2_tensor.sum(dim=0).squeeze(0).detach().numpy()
    
    plt.figure(figsize=(15, 4))
    plt.suptitle(f"Parameters: {params}", fontsize=10)
    plt.subplot(1, 3, 1)
    plt.title('Input Layer (features)')
    plt.bar(np.arange(len(spike_count_input)), spike_count_input)
    plt.xlabel('Feature'); plt.ylabel('Total spikes')
    plt.subplot(1, 3, 2)
    plt.title('Hidden Layer')
    plt.bar(np.arange(len(spike_count_hidden)), spike_count_hidden)
    plt.xlabel('Hidden neuron'); plt.ylabel('Total spikes')
    plt.subplot(1, 3, 3)
    plt.title('Output Layer')
    plt.bar(np.arange(len(spike_count_output)), spike_count_output)
    plt.xlabel('Output neuron'); plt.ylabel('Total spikes')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Unbatch the spike tensors (assume each: [T, 1, n_units])
    spk_input = x.squeeze(1).detach().cpu().numpy()         # [T, input_size]
    spk1_raster = spk1_tensor.squeeze(1).detach().cpu().numpy()  # [T, hidden_size]
    spk2_raster = spk2_tensor.squeeze(1).detach().cpu().numpy()  # [T, output_size * population_per_class]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # Input layer raster
    axes[0].imshow(spk_input.T, aspect='auto', cmap='Greys', interpolation='nearest', origin='lower')
    axes[0].set_title('Input Layer Raster')
    axes[0].set_ylabel('Input feature')
    axes[0].set_yticks(np.arange(spk_input.shape[1]))
    
    # Hidden layer raster
    axes[1].imshow(spk1_raster.T, aspect='auto', cmap='Greys', interpolation='nearest', origin='lower')
    axes[1].set_title('Hidden Layer Raster')
    axes[1].set_ylabel('Hidden neuron')
    axes[1].set_yticks(np.arange(spk1_raster.shape[1]))
    
    # Output layer raster
    axes[2].imshow(spk2_raster.T, aspect='auto', cmap='Greys', interpolation='nearest', origin='lower')
    axes[2].set_title('Output Layer Raster')
    axes[2].set_ylabel('Output neuron')
    axes[2].set_xlabel('Timestep')
    axes[2].set_yticks(np.arange(spk2_raster.shape[1]))
    
    plt.tight_layout()
    plt.show()
    
