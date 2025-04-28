import h5py
import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate, spikegen
from sklearn.metrics import accuracy_score, confusion_matrix
from itertools import combinations
from scipy.signal import butter, filtfilt
from scipy.linalg import eigh
import logging
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# -------------------------- Bandpass Filter --------------------------
def bandpass_filter(data, lowcut, highcut, fs=250.0, order=5):
    b, a = butter(order, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype='band')
    filtered = np.zeros_like(data)
    for trial in range(data.shape[0]):
        for ch in range(data.shape[1]):
            signal = data[trial, ch, :]
            filtered[trial, ch, :] = filtfilt(b, a, signal)
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

    def fit(self, X, y, reg_lambda=0.01):
        self.class_pairs = list(combinations(
            np.unique(y) if self.selected_classes is None else self.selected_classes, 2
        ))
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
            X_proj = np.array([trial / np.std(trial) if np.std(trial) > 0 else trial for trial in X_proj])
            projected[(cl1, cl2)] = X_proj
        return projected

# -------------------------- Spike Encoding --------------------------
def encode_selected_signals_to_spikes(selected_projected_data, base_thresh=0.02, adapt_inc=0.04, decay=0.95, seed=None):
    """
    selected_projected_data: numpy array or torch tensor of shape (n_trials, n_components, n_times)
    """
    if seed is not None:
        torch.manual_seed(seed)

    if isinstance(selected_projected_data, np.ndarray):
        tensor_data = torch.tensor(selected_projected_data).float()
    else:
        tensor_data = selected_projected_data.float()

    # Reorder to (time, batch, channels)
    tensor_data = tensor_data.permute(2, 0, 1)  # (time_steps, batch_size, n_components)

    time_steps, batch_size, num_channels = tensor_data.shape
    spikes = torch.zeros_like(tensor_data)
    thresholds = torch.full((batch_size, num_channels), base_thresh)

    for t in range(1, time_steps):
        delta = (tensor_data[t] - tensor_data[t - 1]).abs()
        spike_t = (delta > thresholds).float()
        spikes[t] = spike_t
        thresholds = thresholds * decay + spike_t * adapt_inc

    return spikes  # shape: (time, batch, channels)




# -------------------------- SNN Model --------------------------
class SNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, population_per_class=5):
        super().__init__()
        self.population_per_class = population_per_class
        self.total_outputs = output_size * population_per_class

        beta = 0.95
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.fc2 = nn.Linear(hidden_size, self.total_outputs)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec = []
        for step in range(x.size(0)):
            spk1, mem1 = self.lif1(self.fc1(x[step]), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            spk2_rec.append(spk2)
        return torch.stack(spk2_rec)  # shape: (time, batch, total_outputs)


# -------------------------- Target Spike Generator --------------------------
def create_sparse_temporal_population_spikes(y, num_classes, population_per_class, num_steps, batch_size, spike_prob=0.7):
    total_outputs = num_classes * population_per_class
    ideal_spikes = torch.zeros((num_steps, batch_size, total_outputs))
    for i in range(batch_size):
        class_idx = y[i]
        start = class_idx * population_per_class
        end = start + population_per_class
        for t in range(num_steps):
            for n in range(start, end):
                if torch.rand(1).item() < spike_prob:
                    ideal_spikes[t, i, n] = 1.0
    return ideal_spikes


# -------------------------- Van Rossum Loss --------------------------
def van_rossum_convolution(spikes, tau, dt=1.0):
    alpha = dt / tau
    filtered = torch.zeros_like(spikes)
    filtered[0] = spikes[0]
    for t in range(1, spikes.shape[0]):
        filtered[t] = (1 - alpha) * filtered[t - 1] + alpha * spikes[t]
    return filtered

def van_rossum_loss(output_spikes, target_spikes, tau=20.0, dt=1.0):
    f_pred = van_rossum_convolution(output_spikes, tau, dt)
    f_target = van_rossum_convolution(target_spikes, tau, dt)
    return torch.mean((f_pred - f_target) ** 2)


# -------------------------- Training and Evaluation --------------------------
def train_with_ideal_spikes(model, X_train, y_train, X_val, y_val, LR=1e-3, epochs=10):
    X_train, X_val = X_train.float(), X_val.float()
    y_train, y_val = y_train.long(), y_val.long()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    loss_fn = lambda out, tgt: van_rossum_loss(out, tgt, tau=20.0)

    num_classes = len(torch.unique(y_train))
    population_per_class = model.population_per_class
    num_steps = X_train.shape[0]

    # Precompute ideal spike targets (FIXED over all epochs)
    train_ideal_spikes = create_sparse_temporal_population_spikes(y_train, num_classes, population_per_class, num_steps, X_train.shape[1])
    val_ideal_spikes = create_sparse_temporal_population_spikes(y_val, num_classes, population_per_class, num_steps, X_val.shape[1])

    for epoch in range(epochs):
        optimizer.zero_grad()
        output_spikes = model(X_train)
        loss = loss_fn(output_spikes, train_ideal_spikes)
        loss.backward()
        optimizer.step()

        output_sum = output_spikes.sum(dim=0)
        predicted = torch.argmax(output_sum.view(X_train.shape[1], num_classes, population_per_class).sum(dim=2), dim=1)
        acc = accuracy_score(y_train.cpu(), predicted.cpu())

        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_pred = torch.argmax(val_output.sum(dim=0).view(X_val.shape[1], num_classes, population_per_class).sum(dim=2), dim=1)
            val_acc = accuracy_score(y_val.cpu(), val_pred.cpu())
            val_loss = loss_fn(val_output, val_ideal_spikes)
        model.train()

        print(f"Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Train Acc: {acc*100:.2f}%, Test Loss: {val_loss.item():.4f}, Test Acc: {val_acc*100:.2f}%)")


def evaluate(model, X_eval, y_eval):
    model.eval()
    num_classes = len(torch.unique(y_eval))
    population_per_class = model.population_per_class
    with torch.no_grad():
        output = model(X_eval)
        output_sum = output.sum(dim=0)
        pred = torch.argmax(output_sum.view(X_eval.shape[1], num_classes, population_per_class).sum(dim=2), dim=1)
        acc = accuracy_score(y_eval.cpu(), pred.cpu())
        cm = confusion_matrix(y_eval.cpu(), pred.cpu())
    return acc, cm

# 1. Compute F-statistics for all 132 CSP signals
def compute_f_scores(projected_data_dict, y_labels):
    all_signals = []
    signal_names = []

    for pair, projected_data in projected_data_dict.items():
        # projected_data shape: (n_trials, n_components, n_times)
        n_trials, n_components, n_times = projected_data.shape
        for comp_idx in range(n_components):
            # For each component, average over time to get 1 value per trial
            feature = projected_data[:, comp_idx, :].mean(axis=1)  # shape: (n_trials,)
            all_signals.append(feature)
            signal_names.append(f"Pair{pair}_Comp{comp_idx}")

    all_signals = np.stack(all_signals, axis=1)  # (n_trials, 132)
    return all_signals, signal_names

def compute_anova_f_scores(features, labels):
    f_scores = []
    n_features = features.shape[1]
    for i in range(n_features):
        # Split data by class
        groups = []
        for c in np.unique(labels):
            groups.append(features[labels == c, i])
        # Perform one-way ANOVA
        f_val, _ = f_oneway(*groups)
        f_scores.append(f_val)
    return np.array(f_scores)

def select_top_csp_components(projected_data_dict, selected_names):
    selected_features = []

    for name in selected_names:
        # Parse name like "Pair(1, 4)_Comp8"
        pair_str, comp_str = name.split('_')
        pair = tuple(map(int, pair_str.strip('Pair()').split(',')))
        comp_idx = int(comp_str.replace('Comp', ''))

        # Extract the corresponding component
        projected_data = projected_data_dict[pair]  # (n_trials, n_components, n_times)
        selected_features.append(projected_data[:, comp_idx, :])  # (n_trials, n_times)

    selected_features = np.stack(selected_features, axis=1)  # (n_trials, top_n, n_times)
    return selected_features


# -------------------------- Main Script --------------------------
if __name__ == "__main__":
    import os
    base_directory = r'C:/Users/USER/Desktop/fbcsp-snn-mi-classifier/fbcsp-snn-mi-classifier'
    relative_directory = r'Dataset'

    # Load training data
    with h5py.File(os.path.join(base_directory, relative_directory, 'EEG_python_ready_without_ICA_A01T.mat'), 'r') as file:
        X_train = file['X'][:]
        X_train = np.transpose(X_train, (2, 0, 1))
        y_train = file['y'][:].flatten()

    # Load validation data
    with h5py.File(os.path.join(base_directory, relative_directory, 'EEG_python_ready_without_ICA_A01E.mat'), 'r') as file:
        X_val = file['X'][:]
        X_val = np.transpose(X_val, (2, 0, 1))
        y_val = file['y'][:].flatten()

    # Bandpass filtering
    freq_band = (4, 30)
    X_train_filtered = bandpass_filter(X_train, *freq_band)
    X_val_filtered = bandpass_filter(X_val, *freq_band)

    # Pairwise CSP
    csp = PairwiseCSP(n_components=22, selected_classes=[1, 2, 3, 4])
    csp.fit(X_train_filtered, y_train)
    projected_train = csp.transform(X_train_filtered)
    projected_val = csp.transform(X_val_filtered)
    
    # 2. Run the process
    X_features, signal_names = compute_f_scores(projected_train, y_train)
    f_scores = compute_anova_f_scores(X_features, y_train)
    
    # 3. Rank CSP components by F-score
    sorted_idx = np.argsort(f_scores)[::-1]  # descending order
    sorted_scores = f_scores[sorted_idx]
    sorted_names = [signal_names[i] for i in sorted_idx]
    
    top_n = 30

    # Get the names of top-N CSP components
    selected_component_names = sorted_names[:top_n]
    
    # You can also find their indices
    selected_indices = sorted_idx[:top_n]
    
    X_train_selected = select_top_csp_components(projected_train, selected_component_names)
    X_val_selected = select_top_csp_components(projected_val, selected_component_names)
    
    spike_train_train = encode_selected_signals_to_spikes(X_train_selected)
    spike_train_val = encode_selected_signals_to_spikes(X_val_selected)
    
    # Model definition
    input_size = spike_train_train.shape[2]
    hidden_size = 128
    output_size = len(np.unique(y_train))

    population_per_class = 5
    model = SNNClassifier(input_size, hidden_size, output_size=len(np.unique(y_train)), population_per_class=population_per_class)

    # Train the model
    train_with_ideal_spikes(
        model,
        spike_train_train,
        torch.tensor(y_train - 1),
        spike_train_val,
        torch.tensor(y_val - 1),
        LR=1e-3,
        epochs=1000
    )

    # Evaluate
    train_acc, train_cm = evaluate(model, spike_train_train, torch.tensor(y_train - 1))
    test_acc, test_cm = evaluate(model, spike_train_val, torch.tensor(y_val - 1))

    print(f"Train Accuracy: {train_acc*100:.2f}%")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print("Confusion Matrix (Test):\n", train_cm)
    print("Confusion Matrix (Test):\n", test_cm)

    
    print(X_train_selected.shape)  # Should be (n_trials, top_n, n_times)
