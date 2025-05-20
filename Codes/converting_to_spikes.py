import h5py
import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
import logging
import os
import sys
import time
import ast

from snntorch import surrogate
from sklearn.metrics import accuracy_score, confusion_matrix
from itertools import combinations
from scipy.signal import butter, filtfilt
from scipy.linalg import eigh

# -------------------------- Device --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f" Using device: {device}")

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
            #X_proj = np.array([trial / np.std(trial) if np.std(trial) > 0 else trial for trial in X_proj])
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


# -------------------------- Main Script --------------------------
if __name__ == "__main__":

    start = time.time()
    #base_directory = r'/scratch/project_2003397/praveen'
    #base_directory = r'C:/Users/USER/Desktop/fbcsp-snn-mi-classifier/fbcsp-snn-mi-classifier'
    base_directory = r'/Users/hsprde/Documents/GitHub/fbcsp-snn-sgd-mi-classifier'
    relative_directory = r'Dataset'

    # Parameters from command line or defaults
    lambda_R = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0001
    freq_bands = ast.literal_eval(sys.argv[2]) if len(sys.argv) > 2 else [(4, 10), (10,14), (14,30)]
    base_thresh_val = float(sys.argv[3]) if len(sys.argv) > 3 else 0.001
    adapt_inc_val = float(sys.argv[4]) if len(sys.argv) > 4 else 0.6
    decay_val = float(sys.argv[5]) if len(sys.argv) > 5 else 0.95

    all_subjects = list(range(1, 10))  # A01–A09

    # Load all T and E sessions per subject
    T_data_all, T_labels_all, T_subjects_all = [], [], []
    E_data_dict, E_labels_dict = {}, {}

    for subjectID in all_subjects:
        # --- Load Training (T) Data ---
        fname_T = f"EEG_python_ready_without_ICA_A0{subjectID}T.mat"
        with h5py.File(os.path.join(base_directory, relative_directory, fname_T), 'r') as f:
            X_T = np.transpose(f['X'][:], (2, 0, 1))
            y_T = f['y'][:].flatten()
            T_data_all.append(X_T)
            T_labels_all.append(y_T)
            T_subjects_all.extend([subjectID] * len(y_T))

        # --- Load Evaluation (E) Data ---
        fname_E = f"EEG_python_ready_without_ICA_A0{subjectID}E.mat"
        with h5py.File(os.path.join(base_directory, relative_directory, fname_E), 'r') as f:
            X_E = np.transpose(f['X'][:], (2, 0, 1))
            y_E = f['y'][:].flatten()
            E_data_dict[subjectID] = X_E
            E_labels_dict[subjectID] = y_E
            
    leave_one_out_train_accuracies, leave_one_out_test_accuracies, leave_one_out_val_accuracies = [], [], []
    
    print("1")

    # Leave-One-Subject-Out Cross Validation
    for val_subject in all_subjects:
        print(f"\n===== Leaving out Subject A0{val_subject} for validation =====")

        # Training data: T sessions of all other subjects
        train_data = [T_data_all[i] for i in range(len(all_subjects)) if all_subjects[i] != val_subject]
        train_labels = [T_labels_all[i] for i in range(len(all_subjects)) if all_subjects[i] != val_subject]
        X_train = np.concatenate(train_data, axis=0)
        y_train = np.concatenate(train_labels, axis=0)
        
        # Test data: E sessions of all training subjects
        test_data = [E_data_dict[sid] for sid in all_subjects if sid != val_subject]
        test_labels = [E_labels_dict[sid] for sid in all_subjects if sid != val_subject]
        X_test = np.concatenate(test_data, axis=0)
        y_test = np.concatenate(test_labels, axis=0)

        # Validation data: E session of left-out subject
        X_val = E_data_dict[val_subject]
        y_val = E_labels_dict[val_subject]

        
        print(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}, Validation set size: {X_val.shape}")
    
        X_train_filtered_bands = [bandpass_filter(X_train, low, high) for (low, high) in freq_bands]
        X_test_filtered_bands = [bandpass_filter(X_test, low, high) for (low, high) in freq_bands]
        X_val_filtered_bands = [bandpass_filter(X_val, low, high) for (low, high) in freq_bands]
        
        X_train_filtered = np.concatenate(X_train_filtered_bands, axis=1)  # shape: (samples, n_channels * n_bands, time)
        X_test_filtered = np.concatenate(X_test_filtered_bands, axis=1)
        X_val_filtered = np.concatenate(X_val_filtered_bands, axis=1)
    
        csp = PairwiseCSP(n_components=X_train_filtered.shape[1], selected_classes=[1, 2, 3, 4])
        csp.fit(X_train_filtered, y_train, reg_lambda=lambda_R)
        
        projected_train = csp.transform(X_train_filtered)
        projected_test = csp.transform(X_test_filtered)
        projected_val = csp.transform(X_val_filtered)
        
        spike_train_train = encode_projected_signals_to_spikes(projected_train, base_thresh=base_thresh_val, adapt_inc=adapt_inc_val, decay=decay_val).to(device)
        spike_train_test = encode_projected_signals_to_spikes(projected_test, base_thresh=base_thresh_val, adapt_inc=adapt_inc_val, decay=decay_val).to(device)
        spike_train_val = encode_projected_signals_to_spikes(projected_val, base_thresh=base_thresh_val, adapt_inc=adapt_inc_val, decay=decay_val).to(device)
        
        # Convert all tensors to NumPy uint8 (saves space)
        train_spikes = spike_train_train.cpu().numpy().astype(np.uint8)
        test_spikes  = spike_train_test.cpu().numpy().astype(np.uint8)
        val_spikes   = spike_train_val.cpu().numpy().astype(np.uint8)
        
        train_labels = y_train.astype(np.uint8)
        test_labels  = y_test.astype(np.uint8)
        val_labels   = y_val.astype(np.uint8)
        
        input("Press Enter")

        # Save everything in one compressed file
        np.savez_compressed(f"spike_trains_with_labels_val_subject_{val_subject}.npz",
            train=train_spikes,
            test=test_spikes,
            val=val_spikes,
            y_train=train_labels,
            y_test=test_labels,
            y_val=val_labels
        )
        
        print(f"✅ Saved compressed spike trains with labels {val_subject}.")
        
        
        
        
    
        