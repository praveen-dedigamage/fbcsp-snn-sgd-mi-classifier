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

# -------------------------- Feature Extraction --------------------------
def extract_csp_variance_features(projected_dict):
    features_per_pair = []
    for pair in sorted(projected_dict):
        X_proj = projected_dict[pair]
        var_features = np.var(X_proj, axis=2)
        features_per_pair.append(var_features)
    return np.concatenate(features_per_pair, axis=1)

# -------------------------- Main Script --------------------------
if __name__ == "__main__":
    base_directory = r'C:/Users/USER/Desktop/fbcsp-snn-mi-classifier/fbcsp-snn-mi-classifier'
    relative_directory = r'Dataset'

    with h5py.File(fr'{base_directory}/{relative_directory}/EEG_python_ready_A01T.mat', 'r') as file:
        X_train = file['X'][:]
        X_train = np.transpose(X_train, (2, 0, 1))
        y_train = file['y'][:].flatten()

    with h5py.File(fr'{base_directory}/{relative_directory}/EEG_python_ready_A01E.mat', 'r') as file:
        X_val = file['X'][:]
        X_val = np.transpose(X_val, (2, 0, 1))
        y_val = file['y'][:].flatten()
        
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42, stratify=y_train)

    freq_bands = [(4, 30)]
    train_filtered = {band: bandpass_filter(X_train, band[0], band[1]) for band in freq_bands}
    val_filtered = {band: bandpass_filter(X_val, band[0], band[1]) for band in freq_bands}

    csp = PairwiseCSP(n_components=22, selected_classes=[1, 2, 3, 4], verbose=True)
    csp.fit(train_filtered[(4, 30)], y_train)
    projected_train = csp.transform(train_filtered[(4, 30)])
    projected_val = csp.transform(val_filtered[(4, 30)])

    X_train_features = extract_csp_variance_features(projected_train)
    X_val_features = extract_csp_variance_features(projected_val)

    # ------------------- SNN Classifier using snnTorch -------------------
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    import snntorch as snn
    import snntorch.functional as SF
    from snntorch import spikegen
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import classification_report, ConfusionMatrixDisplay

    y_train_snn = y_train - 1
    y_val_snn = y_val - 1

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_features)
    X_val_scaled = scaler.transform(X_val_features)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_snn, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val_snn, dtype=torch.long)

    batch_size = 64
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    class SNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.lif1 = snn.Leaky(beta=0.9)
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.lif2 = snn.Leaky(beta=0.9)

        def forward(self, x):
            # x shape: (time, batch, input_dim)
            spk2_rec = []
        
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
        
            for step in range(x.size(0)):  # time
                cur1 = self.fc1(x[step])
                spk1, mem1 = self.lif1(cur1, mem1)
        
                cur2 = self.fc2(spk1)
                spk2, mem2 = self.lif2(cur2, mem2)
        
                spk2_rec.append(spk2)
        
            return torch.stack(spk2_rec)  # shape: [time, batch, output_dim]



    model = SNN(input_size=X_train_tensor.shape[1], hidden_size=128, output_size=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = SF.mse_count_loss()
    num_epochs = 15
    num_steps = 25

    def one_hot_spike_labels(labels, num_steps=25, num_classes=4):
        one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
        return one_hot.unsqueeze(0).repeat(num_steps, 1, 1)

    print("\nTraining SNN...")
    for xb, yb in train_loader:
        xb_spk = spikegen.rate(xb, num_steps=num_steps)
    
        # 🛠️ Fix the label shape: should be [batch]
        if yb.ndim > 1:
            yb = yb.squeeze()  # This will reduce [batch, 1] or [1, batch] to [batch]
    
        spk_out = model(xb_spk)
        loss = loss_fn(spk_out, yb)  # Use 1D labels here
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



    print("\nEvaluating SNN...")
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb_spk = spikegen.rate(xb, num_steps=num_steps)
            spk_out = model(xb_spk)
            pred = spk_out.sum(0).argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
            all_preds.append(pred)
            all_targets.append(yb)

    accuracy = 100 * correct / total
    print(f"\n✅ SNN Validation Accuracy: {accuracy:.2f}%")

    all_preds = torch.cat(all_preds).cpu().numpy()
    all_targets = torch.cat(all_targets).cpu().numpy()
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=["Class 1", "Class 2", "Class 3", "Class 4"]))

    ConfusionMatrixDisplay.from_predictions(all_targets, all_preds, cmap='Blues')
    plt.title("SNN Confusion Matrix (Validation)")
    plt.tight_layout()
    plt.show()