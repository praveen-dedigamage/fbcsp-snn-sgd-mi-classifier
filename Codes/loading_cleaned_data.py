import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.linalg import eigh
from itertools import combinations
from sklearn.model_selection import train_test_split
import logging

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

    def fit(self, X, y):
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

def plot_classwise_csp_variance(projected, y, class_pair, title_suffix=''):
    X_proj = projected[class_pair]
    cl1, cl2 = class_pair
    idx_cl1 = np.where(y == cl1)[0]
    idx_cl2 = np.where(y == cl2)[0]
    var_proj = np.var(X_proj, axis=2)
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

def plot_classwise_csp_variance_all_classes_from_pair(projected_pair, y_true, class_pair):
    """
    Plots variance for ALL classes using the CSP projection from a given class pair.
    """
    var_proj = np.var(projected_pair, axis=2)  # (n_trials, n_components)
    cl_labels = np.unique(y_true)
    x = np.arange(var_proj.shape[1])  # CSP components
    width = 0.8 / len(cl_labels)

    plt.figure(figsize=(12, 5))
    for i, cl in enumerate(cl_labels):
        idx = np.where(y_true == cl)[0]
        if len(idx) == 0:
            continue
        mean_var = np.mean(var_proj[idx], axis=0)
        std_var = np.std(var_proj[idx], axis=0)
        plt.bar(x + (i - len(cl_labels)/2) * width + width/2, mean_var, width,
                yerr=std_var, capsize=4, label=f'Class {cl}', alpha=0.7)

    plt.xlabel("CSP Component Index")
    plt.ylabel("Mean Variance ± STD")
    plt.title(f"All-Class Variance Using CSP from Pair {class_pair}")
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


# -------------------------- Main Script --------------------------
if __name__ == "__main__":
    base_directory = r'C:/Users/USER/Desktop/fbcsp-snn-mi-classifier/fbcsp-snn-mi-classifier'
    relative_directory = r'Dataset'

    with h5py.File(fr'{base_directory}/{relative_directory}/EEG_python_ready_A01T.mat', 'r') as file:
        X_train = file['X'][:]                          # (22, 800, 273)
        X_train = np.transpose(X_train, (2, 0, 1))            # (273, 22, 800)
        y_train = file['y'][:].flatten()                # (273,)
        
    
    with h5py.File(fr'{base_directory}/{relative_directory}/EEG_python_ready_A01E.mat', 'r') as file:
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
    plot_classwise_csp_variance(projected_train, y_train, class_pair=(1, 2))

    for pair in projected_train:
        plot_classwise_csp_variance(projected_train, y_train, class_pair=pair)
        
    for pair in projected_val:
        plot_classwise_csp_variance_all_classes_from_pair(projected_val[pair], y_val, class_pair=pair)
        
    # -------------------------- SNN Encoding and Classification --------------------------
    import torch
    import torch.nn as nn
    import snntorch as snn
    from snntorch import surrogate, spikegen
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, accuracy_score
    
    # Extract CSP variance features
    X_train_feat = extract_csp_variance_features(projected_train)
    X_val_feat = extract_csp_variance_features(projected_val)
    
    # Standardize variance features before encoding
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feat)
    X_val_scaled = scaler.transform(X_val_feat)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    
    # Rate-based encoding (e.g., 50 timesteps)
    n_steps = 50
    X_train_spike = spikegen.rate(X_train_tensor, num_steps=n_steps)
    X_val_spike = spikegen.rate(X_val_tensor, num_steps=n_steps)
    
    # Define SNN model
    class SNNClassifier(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.lif1 = snn.Leaky(beta=0.9)
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.lif2 = snn.Leaky(beta=0.9)
    
        def forward(self, x):
            # x shape: (time, batch, features)
            batch_size = x.shape[1]
        
            mem1 = self.lif1.init_leaky(batch_size=batch_size, device=x.device)
            mem2 = self.lif2.init_leaky(batch_size=batch_size, device=x.device)
        
            spk2_rec = []
        
            for step in range(x.size(0)):  # time dimension
                cur1 = self.fc1(x[step])
                spk1, mem1 = self.lif1(cur1, mem1)
        
                cur2 = self.fc2(spk1)
                spk2, mem2 = self.lif2(cur2, mem2)
        
                spk2_rec.append(spk2)
        
            return torch.stack(spk2_rec)

    
    # Model parameters
    input_dim = X_train_tensor.shape[1]
    hidden_dim = 128
    output_dim = len(np.unique(y_train))
    
    model = SNNClassifier(input_size=input_dim, hidden_size=hidden_dim, output_size=output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    
    # Training loop
    n_epochs = 15
    batch_size = 32
    
    def get_batches(X, y, batch_size):
        for i in range(0, X.size(1), batch_size):
            yield X[:, i:i+batch_size], y[i:i+batch_size]
    
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in get_batches(X_train_spike.permute(1, 0, 2), y_train_tensor, batch_size):
            optimizer.zero_grad()
            out_spikes = model(X_batch.permute(1, 0, 2))
            spike_counts = out_spikes.sum(dim=0)
            loss = loss_fn(spike_counts, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{n_epochs} - Loss: {total_loss:.4f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        out_spikes_test = model(X_val_spike)
        spike_counts_test = out_spikes_test.sum(dim=0)
        preds = spike_counts_test.argmax(dim=1).cpu().numpy()
        true = y_val_tensor.cpu().numpy()
        print("\nTest Accuracy:", accuracy_score(true, preds))
        print(classification_report(true, preds))



