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
def encode_projected_signals_to_spikes(projected_data_dict, base_thresh=0.02, adapt_inc=0.04, decay=0.95, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    all_encoded = []

    for pair in sorted(projected_data_dict):
        data = projected_data_dict[pair]  # shape: (batch, channels, time)
        tensor_data = torch.tensor(data).float()  # (batch, channels, time)
        tensor_data = tensor_data.permute(2, 0, 1)  # -> (time, batch, channels)

        time_steps, batch_size, num_channels = tensor_data.shape
        spikes = torch.zeros_like(tensor_data)
        thresholds = torch.full((batch_size, num_channels), base_thresh)

        for t in range(1, time_steps):
            delta = (tensor_data[t] - tensor_data[t - 1]).abs()
            spike_t = (delta > thresholds).float()
            spikes[t] = spike_t
            thresholds = thresholds * decay + spike_t * adapt_inc

        all_encoded.append(spikes)

    # Concatenate over channel dimension
    return torch.cat(all_encoded, dim=2)  # shape: (time, batch, total_channels)



# -------------------------- SNN Model --------------------------
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
            spk1, mem1 = self.lif1(self.fc1(x[step]), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            spk2_rec.append(spk2)
        return torch.stack(spk2_rec)

# -------------------------- Training and Evaluation --------------------------
def create_ideal_spikes(y, num_classes, num_steps, batch_size, spike_value=1.0):
    ideal_spikes = torch.zeros((num_steps, batch_size, num_classes))
    for i in range(batch_size):
        ideal_spikes[:, i, y[i]] = spike_value
    return ideal_spikes

def train_with_ideal_spikes(model, X_train, y_train, X_val, y_val, LR=1e-3, epochs=10):
    X_train, X_val = X_train.float(), X_val.float()
    y_train, y_val = y_train.long(), y_val.long()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), LR)                            #works
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)     #works
    optimizer = torch.optim.Adagrad(model.parameters(), lr=LR)                      #works
    optimizer = torch.optim.Adadelta(model.parameters())                            #works

    loss_fn = nn.MSELoss()

    num_classes = len(torch.unique(y_train))
    num_steps = X_train.shape[0]

    for epoch in range(epochs):
        optimizer.zero_grad()
        output_spikes = model(X_train)
        ideal_spikes = create_ideal_spikes(y_train, num_classes, num_steps, X_train.shape[1])
        loss = loss_fn(output_spikes, ideal_spikes)
        loss.backward()
        optimizer.step()

        output_sum = output_spikes.sum(dim=0)
        predicted = torch.argmax(output_sum, dim=1)
        acc = accuracy_score(y_train.cpu(), predicted.cpu())

        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_pred = torch.argmax(val_output.sum(dim=0), dim=1)
            val_acc = accuracy_score(y_val.cpu(), val_pred.cpu())
            val_loss = loss_fn(val_output, create_ideal_spikes(y_val, num_classes, num_steps, X_val.shape[1]))
        model.train()

        print(f"Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Train Acc: {acc*100:.2f}%, Test Loss: {val_loss.item():.4f}, Test Acc: {val_acc*100:.2f}%")

def evaluate(model, X_eval, y_eval):
    model.eval()
    with torch.no_grad():
        output = model(X_eval)
        pred = torch.argmax(output.sum(dim=0), dim=1)
        acc = accuracy_score(y_eval.cpu(), pred.cpu())
        cm = confusion_matrix(y_eval.cpu(), pred.cpu())
    return acc, cm


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

    # Spike encoding
    time_steps = 100
    spike_train_train = encode_projected_signals_to_spikes(projected_train)
    spike_train_val = encode_projected_signals_to_spikes(projected_val)

    # Model definition
    input_size = spike_train_train.shape[2]
    hidden_size = 128
    output_size = len(np.unique(y_train))

    model = SNNClassifier(input_size, hidden_size, output_size)

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
    train_acc, _ = evaluate(model, spike_train_train, torch.tensor(y_train - 1))
    test_acc, test_cm = evaluate(model, spike_train_val, torch.tensor(y_val - 1))

    print(f"Train Accuracy: {train_acc*100:.2f}%")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print("Confusion Matrix (Test):\n", test_cm)