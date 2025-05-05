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

from snntorch import surrogate, spikegen
from sklearn.metrics import accuracy_score, confusion_matrix
from itertools import combinations
from scipy.signal import butter, filtfilt
from scipy.linalg import eigh

# -------------------------- Device --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

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
        return torch.stack(spk2_rec)

# -------------------------- Target Spike Generator --------------------------
def create_sparse_temporal_population_spikes(y, num_classes, population_per_class, num_steps, batch_size, spike_prob=0.7):
    total_outputs = num_classes * population_per_class
    ideal_spikes = torch.zeros((num_steps, batch_size, total_outputs), device=device)
    for i in range(batch_size):
        class_idx = int(y[i].item())
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
    import time
    from sklearn.metrics import accuracy_score

    start = time.time()
    X_train, X_val = X_train.to(device), X_val.to(device)
    y_train, y_val = y_train.to(device), y_val.to(device)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    loss_fn = lambda out, tgt: van_rossum_loss(out, tgt, tau=20.0)

    num_classes = len(torch.unique(y_train))
    population_per_class = model.population_per_class
    num_steps = X_train.shape[0]

    train_ideal_spikes = create_sparse_temporal_population_spikes(
        y_train, num_classes, population_per_class, num_steps, X_train.shape[1])
    val_ideal_spikes = create_sparse_temporal_population_spikes(
        y_val, num_classes, population_per_class, num_steps, X_val.shape[1])

    best_test_acc = 0.0
    best_model_state = None

    # ⬇️ Initialize history lists
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    print("Checkpoint 3.1", time.time() - start)

    for epoch in range(epochs):
        start = time.time()

        optimizer.zero_grad()
        output_spikes = model(X_train)
        loss = loss_fn(output_spikes, train_ideal_spikes)
        loss.backward()
        optimizer.step()

        # Train accuracy
        output_sum = output_spikes.sum(dim=0)
        predicted = torch.argmax(
            output_sum.view(X_train.shape[1], num_classes, population_per_class).sum(dim=2), dim=1)
        acc = accuracy_score(y_train.cpu(), predicted.cpu())

        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_pred = torch.argmax(
                val_output.sum(dim=0).view(X_val.shape[1], num_classes, population_per_class).sum(dim=2), dim=1)
            val_acc = accuracy_score(y_val.cpu(), val_pred.cpu())
            val_loss = loss_fn(val_output, val_ideal_spikes)
        model.train()

        # ⬇️ Append to history
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        train_accuracies.append(acc)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Train Acc: {acc*100:.2f}%, "
              f"Test Loss: {val_loss.item():.4f}, Test Acc: {val_acc*100:.2f}%")
        print(f"Checkpoint 3.1.{epoch}", time.time() - start)
        
        sys.stdout.flush()

        # Save best model
        if val_acc > best_test_acc:
            best_test_acc = val_acc
            best_model_state = model.state_dict()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, train_accuracies, val_losses, val_accuracies


def evaluate(model, X_eval, y_eval):
    model.eval()
    X_eval = X_eval.to(device)
    y_eval = y_eval.to(device)
    num_classes = len(torch.unique(y_eval))
    population_per_class = model.population_per_class
    with torch.no_grad():
        output = model(X_eval)
        output_sum = output.sum(dim=0)
        pred = torch.argmax(output_sum.view(X_eval.shape[1], num_classes, population_per_class).sum(dim=2), dim=1)
        acc = accuracy_score(y_eval.cpu(), pred.cpu())
        cm = confusion_matrix(y_eval.cpu(), pred.cpu())
    return acc, cm

# -------------------------- Main Script --------------------------
if __name__ == "__main__":
    start = time.time()
    #base_directory = r'/scratch/project_2003397/praveen'
    #relative_directory = r'Dataset'
    base_directory = r'C:/Users/USER/Desktop/fbcsp-snn-mi-classifier/fbcsp-snn-mi-classifier'
    relative_directory = r'Dataset'

    with h5py.File(os.path.join(base_directory, relative_directory, 'EEG_python_ready_without_ICA_A01T.mat'), 'r') as file:
        X_train = file['X'][:]
        X_train = np.transpose(X_train, (2, 0, 1))
        y_train = file['y'][:].flatten()

    with h5py.File(os.path.join(base_directory, relative_directory, 'EEG_python_ready_without_ICA_A01E.mat'), 'r') as file:
        X_val = file['X'][:]
        X_val = np.transpose(X_val, (2, 0, 1))
        y_val = file['y'][:].flatten()
    
    print("Checkpoint 1", time.time() - start)
    
    start = time.time()
    
    lambda_R = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
    freq_bands = ast.literal_eval(sys.argv[2]) if len(sys.argv) > 2 else [(8, 12), (12, 20), (20, 30)]

    X_train_filtered_bands = [bandpass_filter(X_train, low, high) for (low, high) in freq_bands]
    X_val_filtered_bands = [bandpass_filter(X_val, low, high) for (low, high) in freq_bands]
    
    X_train_filtered = np.concatenate(X_train_filtered_bands, axis=1)  # shape: (samples, n_channels * n_bands, time)
    X_val_filtered = np.concatenate(X_val_filtered_bands, axis=1)

    csp = PairwiseCSP(n_components=X_train_filtered.shape[1], selected_classes=[1, 2, 3, 4])
    csp.fit(X_train_filtered, y_train, reg_lambda=lambda_R)
    projected_train = csp.transform(X_train_filtered)
    projected_val = csp.transform(X_val_filtered)
    
    print("Checkpoint 2", time.time() - start)
    start = time.time()
    
    spike_train_train = encode_projected_signals_to_spikes(projected_train).to(device)
    spike_train_val = encode_projected_signals_to_spikes(projected_val).to(device)

    input_size = spike_train_train.shape[2]
    hidden_size = 128
    output_size = len(np.unique(y_train))
    population_per_class = 10

    model = SNNClassifier(input_size, hidden_size, output_size, population_per_class).to(device)
    
    print("Checkpoint 3", time.time() - start)
    start = time.time()
    
    best_model, train_losses, train_accuracies, val_losses, val_accuracies = train_with_ideal_spikes(
        model,
        spike_train_train,
        torch.tensor(y_train - 1).to(device),
        spike_train_val,
        torch.tensor(y_val - 1).to(device),
        LR=1e-3,
        epochs=1000
    )
    print("Checkpoint 4", time.time() - start)
    start = time.time()

    train_acc, train_cm = evaluate(model, spike_train_train, torch.tensor(y_train - 1))
    test_acc, test_cm = evaluate(model, spike_train_val, torch.tensor(y_val - 1))

    results_dir = os.path.join(base_directory, "results")
    os.makedirs(results_dir, exist_ok=True)
    summary_file = os.path.join(results_dir, f"summary_lambda_{lambda_R:.2f}.csv")

    with open(summary_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Train Accuracy", train_acc])
        writer.writerow(["Test Accuracy", test_acc])
        writer.writerow([])
        writer.writerow(["Confusion Matrix - Train"])
        writer.writerows(train_cm)
        writer.writerow([])
        writer.writerow(["Confusion Matrix - Test"])
        writer.writerows(test_cm)

    print(f"\n✅ Final results saved to: {summary_file}")

    save_path = f'model_and_history{lambda_R:.2f}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'train_cm': train_cm,
        'test_cm': test_cm,
        'train_acc': train_acc,
        'test_acc': test_acc,
    }, save_path)

    print(f"\n✅ Final results saved to: {save_path}")
    print("Checkpoint 5", time.time() - start)