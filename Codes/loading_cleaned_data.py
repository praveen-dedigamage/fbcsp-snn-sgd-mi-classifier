import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.linalg import eigh
from itertools import combinations
from sklearn.model_selection import train_test_split
import logging
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate, spikegen
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# -------------------------- Bandpass Filter --------------------------
def bandpass_filter(data, lowcut, highcut, fs=250.0, order=5):
    b, a = butter(order, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype='band')
    filtered = np.zeros_like(data)
    for trial in range(data.shape[0]):
        for ch in range(data.shape[1]):
            signal = data[trial, ch, :]
            if signal.shape[0] > 27:
                filtered[trial, ch, :] = filtfilt(b, a, signal)
            else:
                raise ValueError(f"Signal too short: {signal.shape[0]} samples")
    return filtered

# -------------------------- Pairwise CSP --------------------------
class PairwiseCSP:
    def __init__(self, n_components=2, selected_classes=None, verbose=True):
        self.n_components = n_components
        self.selected_classes = selected_classes
        self.pairwise_filters = {}
        self.class_pairs = []
        self.logger = logging.getLogger("PairwiseCSP")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
            self.logger.addHandler(handler)
        self.logger.propagate = False
        if not verbose:
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

    def fit(self, X, y, reg_lambda=0.01):
        self.class_pairs = list(combinations(
            np.unique(y) if self.selected_classes is None else self.selected_classes, 2
        ))
        for (cl1, cl2) in self.class_pairs:
            idx = np.where((y == cl1) | (y == cl2))[0]
            X_pair = X[idx]
            y_pair = y[idx]
            covs = []
            for trial in X_pair:
                cov = np.cov(trial)
                cov /= np.trace(cov)
                covs.append(cov)
            covs = np.stack(covs)
            cov1 = np.mean(covs[y_pair == cl1], axis=0)
            cov2 = np.mean(covs[y_pair == cl2], axis=0)
    
            # Riemannian regularization
            I = np.eye(cov1.shape[0])
            cov1 = (1 - reg_lambda) * cov1 + reg_lambda * I
            cov2 = (1 - reg_lambda) * cov2 + reg_lambda * I
    
            eigvals, eigvecs = eigh(cov1, cov1 + cov2)
            ix = np.argsort(eigvals)[::-1]
            W = eigvecs[:, ix]
            self._validate_filter(W[:, :self.n_components], cl1, cl2)
            self.pairwise_filters[(cl1, cl2)] = W[:, :self.n_components]
        return self


    def transform(self, X):
        projected = {}
        for (cl1, cl2), W in self.pairwise_filters.items():
            X_proj = np.array([W.T @ trial for trial in X])
            X_proj = np.array([trial / np.std(trial) if np.std(trial) > 0 else trial for trial in X_proj])
            projected[(cl1, cl2)] = X_proj
        return projected

# -------------------------- Plotting Functions --------------------------
def plot_eeg_style_raw(signal_data, trial_idx=0, fs=250, spacing=20, channel_names=None):
    n_channels = signal_data.shape[1]
    time = np.arange(signal_data.shape[2]) / fs
    plt.figure(figsize=(14, 0.5 * n_channels + 4))
    for ch in range(n_channels):
        signal = signal_data[trial_idx, ch, :] + ch * spacing
        plt.plot(time, signal, label=channel_names[ch] if channel_names else f'Ch {ch+1}')
    plt.title(f'EEG-Style Plot of Raw/Bandpassed Signals (Trial {trial_idx})')
    plt.xlabel('Time (s)')
    plt.yticks(np.arange(0, spacing * n_channels, spacing), 
               channel_names if channel_names else [f'Ch {i+1}' for i in range(n_channels)])
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_csp_eeg_style(projected, y, trial_idx=0, class_pair=(1, 2), fs=250, spacing=10):
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
    

def plot_classwise_csp_variance(projected, y, class_pair, title_suffix=''):
    X_proj = projected[class_pair]  # shape: [n_trials, n_components, n_samples]
    cl1, cl2 = class_pair
    idx_cl1 = np.where(y == cl1)[0]
    idx_cl2 = np.where(y == cl2)[0]

    # Step 1: Compute variance over time (samples)
    var_proj = np.var(X_proj, axis=2)  # shape: [n_trials, n_components]

    # Step 2: Normalize each trial's variances to range [0, 1]
    min_vals = np.min(var_proj, axis=1, keepdims=True)
    max_vals = np.max(var_proj, axis=1, keepdims=True)
    var_proj_norm = (var_proj - min_vals) / (max_vals - min_vals + 1e-8)  # shape: [n_trials, n_components]

    # Step 3: Class-wise statistics
    mean_var_cl1 = np.mean(var_proj_norm[idx_cl1], axis=0)
    std_var_cl1 = np.std(var_proj_norm[idx_cl1], axis=0)

    mean_var_cl2 = np.mean(var_proj_norm[idx_cl2], axis=0)
    std_var_cl2 = np.std(var_proj_norm[idx_cl2], axis=0)

    # Step 4: Plot
    x = np.arange(len(mean_var_cl1))
    width = 0.35
    plt.figure(figsize=(12, 5))
    plt.bar(x - width/2, mean_var_cl1, width, yerr=std_var_cl1, capsize=4,
            label=f'Class {cl1}', alpha=0.7)
    plt.bar(x + width/2, mean_var_cl2, width, yerr=std_var_cl2, capsize=4,
            label=f'Class {cl2}', alpha=0.7)
    plt.xlabel("CSP Component Index")
    plt.ylabel("Mean Normalized Variance ± STD")
    plt.title(f"CSP Component Variance by Class – Pair {class_pair} {title_suffix}")
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()



def plot_classwise_csp_variance_all_classes_from_pair(projected_pair, y_true, class_pair):
    """
    Plots normalized variance for ALL classes using the CSP projection from a given class pair.
    Each trial's CSP variances are min-max normalized to [0, 1] before class-wise averaging.
    """
    # Step 1: Compute variances across time axis
    var_proj = np.var(projected_pair, axis=2)  # shape: (n_trials, n_components)

    # Step 2: Normalize each trial's variances to [0, 1]
    min_vals = np.min(var_proj, axis=1, keepdims=True)
    max_vals = np.max(var_proj, axis=1, keepdims=True)
    var_proj_norm = (var_proj - min_vals) / (max_vals - min_vals + 1e-8)

    # Step 3: Plot class-wise mean ± std
    cl_labels = np.unique(y_true)
    x = np.arange(var_proj.shape[1])  # CSP components
    width = 0.8 / len(cl_labels)

    plt.figure(figsize=(12, 5))
    for i, cl in enumerate(cl_labels):
        idx = np.where(y_true == cl)[0]
        if len(idx) == 0:
            continue
        mean_var = np.mean(var_proj_norm[idx], axis=0)
        std_var = np.std(var_proj_norm[idx], axis=0)
        plt.bar(x + (i - len(cl_labels)/2) * width + width/2, mean_var, width,
                yerr=std_var, capsize=4, label=f'Class {cl}', alpha=0.7)

    plt.xlabel("CSP Component Index")
    plt.ylabel("Normalized Mean Variance ± STD")
    plt.title(f"All-Class Normalized Variance Using CSP from Pair {class_pair}")
    plt.grid(axis='y')
    plt.legend()
    plt.tight_layout()
    plt.show()

    
def extract_csp_variance_features(projected_dict):
    """
    From a CSP-projected dictionary, extract per-trial variance features across all class pairs.
    Output shape: (n_trials, total_features)
    """
    features_per_pair = []
    for pair in sorted(projected_dict):  # sort for consistent order
        X_proj = projected_dict[pair]  # shape: (n_trials, n_components, time)
        var_features = np.var(X_proj, axis=2)  # shape: (n_trials, n_components)
        features_per_pair.append(var_features)
    return np.concatenate(features_per_pair, axis=1)  # shape: (n_trials, n_pairs * n_components)



class SNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        beta = 0.95
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec = []
        for step in range(x.size(0)):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
        return torch.stack(spk2_rec)
    
def create_ideal_spikes(y, num_classes, num_steps, batch_size, spike_value=1.0):
    """
    Creates ideal spike trains:
    - Correct class neuron: spike_value at every timestep
    - Others: 0
    """
    ideal_spikes = torch.zeros((num_steps, batch_size, num_classes))
    for i in range(batch_size):
        ideal_spikes[:, i, y[i]] = spike_value
    return ideal_spikes


def train_with_ideal_spikes(model, X_train, y_train, X_val, y_val, LR=1e-3, epochs=10):
    
    
    X_train = X_train.float().to(next(model.parameters()).device)
    X_val = X_val.float().to(next(model.parameters()).device)
    y_train = y_train.to(next(model.parameters()).device)
    y_val = y_val.to(next(model.parameters()).device)
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), LR)
    loss_fn = nn.MSELoss()

    loss_list = []
    accuracy_list = []
    val_loss_list = []
    val_accuracy_list = []

    num_classes = len(torch.unique(y_train))
    num_steps = X_train.shape[0]

    for epoch in range(epochs):
        # Forward
        optimizer.zero_grad()
        output_spikes = model(X_train)

        # Create ideal spike pattern
        ideal_spikes = create_ideal_spikes(y_train, num_classes, num_steps, X_train.shape[1])
        ideal_spikes = ideal_spikes.to(output_spikes.device)

        # Compute loss (MSE between predicted and ideal spike trains)
        loss = loss_fn(output_spikes, ideal_spikes)
        loss.backward()
        optimizer.step()

        # Accuracy (based on summed spikes)
        output_sum = output_spikes.sum(dim=0)
        predicted = torch.argmax(output_sum, dim=1)
        acc = accuracy_score(y_train.cpu(), predicted.cpu())

        loss_list.append(loss.item())
        accuracy_list.append(acc)

        # Validation
        model.eval()
        with torch.no_grad():
            val_output_spikes = model(X_val)
            val_output_sum = val_output_spikes.sum(dim=0)
            val_predicted = torch.argmax(val_output_sum, dim=1)
            val_acc = accuracy_score(y_val.cpu(), val_predicted.cpu())

            val_ideal = create_ideal_spikes(y_val, num_classes, num_steps, X_val.shape[1]).to(val_output_spikes.device)
            val_loss = loss_fn(val_output_spikes, val_ideal)

            val_loss_list.append(val_loss.item())
            val_accuracy_list.append(val_acc)

        model.train()

        print(f"Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Train Acc: {acc*100:.2f}%, Test Loss: {val_loss.item():.4f}, Test Acc: {val_acc*100:.2f}%")

    return loss_list, accuracy_list, val_loss_list, val_accuracy_list


def evaluate(model, X_eval, y_eval):
    model.eval()
    with torch.no_grad():
        output_spikes = model(X_eval)
        output_sum = output_spikes.sum(dim=0)
        predicted = torch.argmax(output_sum, dim=1)
        acc = accuracy_score(y_eval.cpu(), predicted.cpu())
        cm = confusion_matrix(y_eval.cpu(), predicted.cpu())
        return acc, cm
    
def encode_projected_signals_to_spikes(projected_data_dict, time_steps=100, gain=1.0, offset=0.0, seed=None):
    """
    Encodes CSP-projected signals into spike trains using rate-based encoding.

    Parameters:
    - projected_data_dict: dict with class pair keys and numpy arrays of shape [n_trials, n_components, n_timepoints]
    - time_steps: number of time steps for spike train encoding
    - gain: scaling factor for the rate
    - offset: added offset to rate before spike generation
    - seed: random seed for reproducibility

    Returns:
    - spike_tensor_dict: dict with same keys as projected_data_dict and torch spike tensors
                         shape: [time_steps, n_trials, n_components * n_pairs]
    """
    import torch
    from snntorch import spikegen

    if seed is not None:
        torch.manual_seed(seed)

    all_encoded = []
    for pair in sorted(projected_data_dict):  # keep consistent order
        data = projected_data_dict[pair]  # shape: [n_trials, n_components, n_timepoints]

        # Feature extraction: variance across time (like in original pipeline)
        variance_features = np.var(data, axis=2)  # shape: [n_trials, n_components]

        # Normalize to [0, 1] range for spike rate encoding
        min_vals = np.min(variance_features, axis=0, keepdims=True)
        max_vals = np.max(variance_features, axis=0, keepdims=True)
        norm_variance = (variance_features - min_vals) / (max_vals - min_vals + 1e-8)

        # Apply gain and offset, clip to [0, 1]
        rates = np.clip(norm_variance * gain + offset, 0, 1)  # shape: [n_trials, n_components]

        # Convert to spike trains
        spike_tensor = spikegen.rate(torch.tensor(rates).float(), num_steps=time_steps)  # [time_steps, n_trials, n_components]
        all_encoded.append(spike_tensor)  # append as list of spike tensors

    # Concatenate all pairs along feature dim (last dimension)
    spike_tensor_all = torch.cat(all_encoded, dim=2)  # shape: [time_steps, n_trials, total_features]

    return spike_tensor_all

import matplotlib.pyplot as plt
import torch

def plot_spike_train(spike_tensor, trial_idx=0, title='Spike Train for Trial', max_neurons=132):
    """
    Plots the spike raster for a single trial from a spike tensor.
    
    Parameters:
    - spike_tensor: torch.Tensor of shape [time_steps, n_trials, n_neurons]
    - trial_idx: index of the trial to visualize
    - title: title for the plot
    - max_neurons: maximum number of neurons to plot (for readability)
    """
    if not isinstance(spike_tensor, torch.Tensor):
        raise ValueError("Expected spike_tensor to be a torch.Tensor")
    
    time_steps, n_trials, n_neurons = spike_tensor.shape
    if trial_idx >= n_trials:
        raise IndexError(f"Trial index {trial_idx} out of range. Only {n_trials} trials available.")

    spikes = spike_tensor[:, trial_idx, :].cpu().numpy()  # shape: [time_steps, n_neurons]

    plt.figure(figsize=(12, 6))
    for neuron_idx in range(min(n_neurons, max_neurons)):
        spike_times = np.where(spikes[:, neuron_idx])[0]
        plt.scatter(spike_times, [neuron_idx]*len(spike_times), s=10, color='black')

    plt.xlabel("Time step")
    plt.ylabel("Neuron Index")
    plt.title(f"{title} {trial_idx}")
    plt.yticks(range(min(n_neurons, max_neurons)))
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
import numpy as np
import matplotlib.pyplot as plt
import torch

def compute_and_plot_mean_spike_counts(spike_tensor, title="Mean Spike Count per CSP Component"):
    """
    Computes and plots the mean spike count per component across all trials.
    
    Parameters:
    - spike_tensor: torch.Tensor of shape [time_steps, n_trials, n_components]
    - title: plot title
    
    Returns:
    - mean_counts: numpy array of shape [n_components]
    """
    if not isinstance(spike_tensor, torch.Tensor):
        raise ValueError("Expected spike_tensor to be a torch.Tensor")

    # Sum over time: shape -> [n_trials, n_components]
    spike_counts = spike_tensor.sum(dim=0).cpu().numpy()
    
    # Mean over trials: shape -> [n_components]
    mean_counts = np.mean(spike_counts, axis=0)
    
    # Plot
    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(len(mean_counts)), mean_counts)
    plt.xlabel("CSP Component Index")
    plt.ylabel("Mean Spike Count")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return mean_counts

def compute_and_plot_classwise_spike_counts(spike_tensor, labels, target_class, title_prefix=""):
    """
    Compute and plot the mean spike count per component for a specific class.
    
    Parameters:
    - spike_tensor: torch.Tensor of shape [time_steps, n_trials, n_components]
    - labels: array-like of shape [n_trials], class labels
    - target_class: class to filter trials
    - title_prefix: optional string to prepend to the plot title
    
    Returns:
    - mean_counts: numpy array of shape [n_components]
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Convert labels to numpy if needed
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Identify trial indices for the target class
    class_indices = np.where(labels == target_class)[0]
    
    if len(class_indices) == 0:
        raise ValueError(f"No trials found for class {target_class}")

    # Select trials of interest: shape -> [time_steps, n_class_trials, n_components]
    selected_spikes = spike_tensor[:, class_indices, :]

    # Compute spike count: sum over time -> shape [n_class_trials, n_components]
    spike_counts = selected_spikes.sum(dim=0).cpu().numpy()

    # Mean over trials: shape [n_components]
    mean_counts = np.mean(spike_counts, axis=0)

    # Plot
    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(len(mean_counts)), mean_counts)
    plt.xlabel("CSP Component Index")
    plt.ylabel("Mean Spike Count")
    plt.title(f"{title_prefix}Class {target_class} – Mean Spike Count per CSP Component")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return mean_counts

def plot_single_trial_spike_counts(spike_tensor, labels, target_class, trial_number_in_class=0, title_prefix=""):
    """
    Plot spike counts for a single trial (from a specific class) across all CSP components.
    
    Parameters:
    - spike_tensor: torch.Tensor of shape [time_steps, n_trials, n_components]
    - labels: array-like of shape [n_trials], class labels
    - target_class: int, class label to choose the trial from
    - trial_number_in_class: int, index (within that class) of the trial to use
    - title_prefix: optional string to prepend to plot title
    
    Returns:
    - spike_counts: numpy array of shape [n_components], spike counts for the selected trial
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    class_indices = np.where(labels == target_class)[0]

    if len(class_indices) == 0:
        raise ValueError(f"No trials found for class {target_class}")
    
    if trial_number_in_class >= len(class_indices):
        raise IndexError(f"Trial index {trial_number_in_class} exceeds available class {target_class} trials ({len(class_indices)} found)")

    trial_idx = class_indices[trial_number_in_class]
    spike_counts = spike_tensor[:, trial_idx, :].sum(dim=0).cpu().numpy()  # shape: [n_components]

    # Plot
    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(len(spike_counts)), spike_counts)
    plt.xlabel("CSP Component Index")
    plt.ylabel("Spike Count")
    plt.title(f"{title_prefix}Class {target_class} – Trial {trial_number_in_class} (index {trial_idx})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return spike_counts

class SNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        beta = 0.95
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec = []
        for step in range(x.size(0)):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
        return torch.stack(spk2_rec)

def create_ideal_spikes(y, num_classes, num_steps, batch_size, spike_value=1.0):
    """
    Creates ideal spike trains:
    - Correct class neuron: spike_value at every timestep
    - Others: 0
    """
    ideal_spikes = torch.zeros((num_steps, batch_size, num_classes))
    for i in range(batch_size):
        ideal_spikes[:, i, y[i]] = spike_value
    return ideal_spikes



def train_with_ideal_spikes(model, X_train, y_train, X_val, y_val, LR=1e-3, epochs=10):
    
    
    X_train = X_train.float().to(next(model.parameters()).device)
    X_val = X_val.float().to(next(model.parameters()).device)
    y_train = y_train.to(next(model.parameters()).device)
    y_val = y_val.to(next(model.parameters()).device)
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), LR)
    loss_fn = nn.MSELoss()

    loss_list = []
    accuracy_list = []
    val_loss_list = []
    val_accuracy_list = []

    num_classes = len(torch.unique(y_train))
    num_steps = X_train.shape[0]

    for epoch in range(epochs):
        # Forward
        optimizer.zero_grad()
        output_spikes = model(X_train)

        # Create ideal spike pattern
        ideal_spikes = create_ideal_spikes(y_train, num_classes, num_steps, X_train.shape[1])
        ideal_spikes = ideal_spikes.to(output_spikes.device)

        # Compute loss (MSE between predicted and ideal spike trains)
        loss = loss_fn(output_spikes, ideal_spikes)
        loss.backward()
        optimizer.step()

        # Accuracy (based on summed spikes)
        output_sum = output_spikes.sum(dim=0)
        predicted = torch.argmax(output_sum, dim=1)
        acc = accuracy_score(y_train.cpu(), predicted.cpu())

        loss_list.append(loss.item())
        accuracy_list.append(acc)

        # Validation
        model.eval()
        with torch.no_grad():
            val_output_spikes = model(X_val)
            val_output_sum = val_output_spikes.sum(dim=0)
            val_predicted = torch.argmax(val_output_sum, dim=1)
            val_acc = accuracy_score(y_val.cpu(), val_predicted.cpu())

            val_ideal = create_ideal_spikes(y_val, num_classes, num_steps, X_val.shape[1]).to(val_output_spikes.device)
            val_loss = loss_fn(val_output_spikes, val_ideal)

            val_loss_list.append(val_loss.item())
            val_accuracy_list.append(val_acc)

        model.train()

        print(f"Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Train Acc: {acc*100:.2f}%, Test Loss: {val_loss.item():.4f}, Test Acc: {val_acc*100:.2f}%")

    return loss_list, accuracy_list, val_loss_list, val_accuracy_list

def evaluate(model, X_eval, y_eval):
    model.eval()
    with torch.no_grad():
        output_spikes = model(X_eval)
        output_sum = output_spikes.sum(dim=0)
        predicted = torch.argmax(output_sum, dim=1)
        acc = accuracy_score(y_eval.cpu(), predicted.cpu())
        cm = confusion_matrix(y_eval.cpu(), predicted.cpu())
        return acc, cm
    
def plot_curves(train_loss, train_acc, val_loss, val_acc):
    import matplotlib.pyplot as plt

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label='Train Acc')
    plt.plot(epochs, val_acc, label='Val Acc')
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()




# -------------------------- Main Script --------------------------
if __name__ == "__main__":
    base_directory = r'C:/Users/USER/Desktop/fbcsp-snn-mi-classifier/fbcsp-snn-mi-classifier'
    relative_directory = r'Dataset'

    with h5py.File(fr'{base_directory}/{relative_directory}/EEG_python_ready_without_ICA_A01T.mat', 'r') as file:
        X_train = file['X'][:]                          # (22, 800, 273)
        X_train = np.transpose(X_train, (2, 0, 1))      # (273, 22, 800)
        y_train = file['y'][:].flatten()                # (273,)
        
    
    with h5py.File(fr'{base_directory}/{relative_directory}/EEG_python_ready_without_ICA_A01E.mat', 'r') as file:
        X_val = file['X'][:]
        X_val = np.transpose(X_val, (2, 0, 1)) 
        y_val = file['y'][:].flatten()
    

    # Split into training and validation
    #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42, stratify=y_train)

    # Filter band
    freq_bands = [(4, 30)]
    train_filtered = {band: bandpass_filter(X_train, band[0], band[1]) for band in freq_bands}
    val_filtered = {band: bandpass_filter(X_val, band[0], band[1]) for band in freq_bands}

    # Fit CSP
    csp = PairwiseCSP(n_components=22, selected_classes=[1, 2, 3, 4], verbose=True)
    csp.fit(train_filtered[(4, 30)], y_train)
    projected_train = csp.transform(train_filtered[(4, 30)])
    projected_val = csp.transform(val_filtered[(4, 30)])

    # Visualizations
    plot_eeg_style_raw(train_filtered[(4, 30)], trial_idx=0)
    plot_csp_eeg_style(projected_train, y_train, trial_idx=0, class_pair=(1, 2))
    #plot_classwise_csp_variance(projected_train, y_train, class_pair=(1, 2))

    #for pair in projected_train:
        #plot_classwise_csp_variance(projected_train, y_train, class_pair=pair)
        
    #for pair in projected_val:
        #plot_classwise_csp_variance_all_classes_from_pair(projected_val[pair], y_val, class_pair=pair)
    
    i = 100
    
    spike_train_train = encode_projected_signals_to_spikes(projected_train, time_steps=i, gain=1.0, offset=0.0)
    spike_train_val = encode_projected_signals_to_spikes(projected_val, time_steps=i, gain=1.0, offset=0.0)
    
    #plot_spike_train(spike_train_train, trial_idx=5)
    
    #mean_counts_train = compute_and_plot_mean_spike_counts(spike_train_train, title="Train Set Spike Count per CSP Component")

    # For class 1
    #mean_class1 = compute_and_plot_classwise_spike_counts(spike_train_train, y_train, target_class=1)
    
    # For class 2
    #mean_class2 = compute_and_plot_classwise_spike_counts(spike_train_train, y_train, target_class=2)

    # For class 3
    #mean_class3 = compute_and_plot_classwise_spike_counts(spike_train_train, y_train, target_class=3)
    
    # For class 4
    #mean_class4 = compute_and_plot_classwise_spike_counts(spike_train_train, y_train, target_class=4)    
    
    # Visualize the 3rd trial (index 2) from class 2
    #spike_counts = plot_single_trial_spike_counts(spike_train_train, y_train, target_class=2, trial_number_in_class=2)

    # Step 1: Define input, hidden, and output sizes
    input_size = spike_train_train.shape[2]  # total CSP features
    hidden_size = 128
    output_size = len(torch.unique(torch.tensor(y_train)))  # usually 4
    
    # Step 2: Instantiate the model
    model = SNNClassifier(input_size, hidden_size, output_size)
    
    # Step 3: Train the model
    loss_list, accuracy_list, val_loss_list, val_accuracy_list = train_with_ideal_spikes(
        model, 
        spike_train_train, 
        torch.tensor(y_train-1, dtype=torch.long), 
        spike_train_val, 
        torch.tensor(y_val-1, dtype=torch.long), 
        LR=1e-3, 
        epochs=1000
    )

    train_acc, train_cm = evaluate(model, spike_train_train, torch.tensor(y_train-1))
    test_acc, test_cm = evaluate(model, spike_train_val, torch.tensor(y_val-1))
    
    print(f"Train Accuracy: {train_acc*100:.2f}%")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print("Confusion Matrix (Test):\n", test_cm)

        
    