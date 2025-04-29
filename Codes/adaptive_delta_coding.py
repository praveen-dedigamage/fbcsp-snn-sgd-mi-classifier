import h5py
import numpy as np
import torch
import torch.nn as nn
import pickle
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
    import torch
    from sklearn.metrics import accuracy_score

    X_train, X_val = X_train.float(), X_val.float()
    y_train, y_val = y_train.long(), y_val.long()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    loss_fn = lambda out, tgt: van_rossum_loss(out, tgt, tau=20.0)

    num_classes = len(torch.unique(y_train))
    population_per_class = model.population_per_class
    num_steps = X_train.shape[0]

    # Precompute ideal spike targets (FIXED over all epochs)
    train_ideal_spikes = create_sparse_temporal_population_spikes(
        y_train, num_classes, population_per_class, num_steps, X_train.shape[1]
    )
    val_ideal_spikes = create_sparse_temporal_population_spikes(
        y_val, num_classes, population_per_class, num_steps, X_val.shape[1]
    )

    # Lists to store losses and accuracies
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

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

        train_losses.append(loss.item())
        train_accuracies.append(acc)
        val_losses.append(val_loss.item())
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Train Acc: {acc*100:.2f}%, Test Loss: {val_loss.item():.4f}, Test Acc: {val_acc*100:.2f}%")

    return train_losses, train_accuracies, val_losses, val_accuracies

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



# -------------------------- Main Script --------------------------
if __name__ == "__main__":
    for i in range(5):
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
        lambda_R = 0.05 + 0.05*i
        csp.fit(X_train_filtered, y_train, reg_lambda=lambda_R)
        projected_train = csp.transform(X_train_filtered)
        projected_val = csp.transform(X_val_filtered)
    
    
        spike_train_train = encode_projected_signals_to_spikes(projected_train)
        spike_train_val = encode_projected_signals_to_spikes(projected_val)
    
        # Model definition
        input_size = spike_train_train.shape[2]
        hidden_size = 128
        output_size = len(np.unique(y_train))
    
        population_per_class = 10
        model = SNNClassifier(input_size, hidden_size, output_size=len(np.unique(y_train)), population_per_class=population_per_class)
    
        # Train the model
        train_losses, train_accuracies, val_losses, val_accuracies = train_with_ideal_spikes(
            model,
            spike_train_train,
            torch.tensor(y_train - 1),
            spike_train_val,
            torch.tensor(y_val - 1),
            LR=1e-3,
            epochs=1
        )
    
        # Evaluate
        train_acc, train_cm = evaluate(model, spike_train_train, torch.tensor(y_train - 1))
        test_acc, test_cm = evaluate(model, spike_train_val, torch.tensor(y_val - 1))
    
        print(f"Train Accuracy: {train_acc*100:.2f}%")
        print(f"Test Accuracy: {test_acc*100:.2f}%")
        print("Confusion Matrix (Test):\n", test_cm)
    
        # Save model and training history
        save_path = f'model_and_history{i}.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'train_cm':train_cm,
            'test_cm':test_cm,
            'trian_acc':train_acc,
            'test_acc':test_acc,
        }, save_path)
        
        """
        
        # Load model and training history
        checkpoint = torch.load('model_and_history.pth', weights_only=False)
    
        
        # Load model parameters
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load training history
        ltrain_losses = checkpoint['train_losses']
        ltrain_accuracies = checkpoint['train_accuracies']
        lval_losses = checkpoint['val_losses']
        lval_accuracies = checkpoint['val_accuracies']
        ltrain_cm = checkpoint['train_cm']
        ltest_cm = checkpoint['test_cm']
        ltrian_acc = checkpoint['trian_acc']
        ltest_acc = checkpoint['test_acc']
        """


"""   
def plot_two_confusion_matrices(train_cm, test_cm, class_names):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

    # Plot Train Confusion Matrix
    axes[0].imshow(train_cm, interpolation='nearest', cmap='Blues')
    axes[0].set_title('Confusion Matrix (Train)')
    axes[0].set_xlabel('Predicted label')
    axes[0].set_ylabel('True label')
    tick_marks = np.arange(len(class_names))
    axes[0].set_xticks(tick_marks)
    axes[0].set_xticklabels(class_names, rotation=45)
    axes[0].set_yticks(tick_marks)
    axes[0].set_yticklabels(class_names)

    thresh = train_cm.max() / 2.
    for i in range(train_cm.shape[0]):
        for j in range(train_cm.shape[1]):
            axes[0].text(j, i, format(train_cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if train_cm[i, j] > thresh else "black")

    # Plot Test Confusion Matrix
    axes[1].imshow(test_cm, interpolation='nearest', cmap='Blues')
    axes[1].set_title('Confusion Matrix (Test)')
    axes[1].set_xlabel('Predicted label')
    axes[1].set_ylabel('True label')
    axes[1].set_xticks(tick_marks)
    axes[1].set_xticklabels(class_names, rotation=45)
    axes[1].set_yticks(tick_marks)
    axes[1].set_yticklabels(class_names)

    thresh = test_cm.max() / 2.
    for i in range(test_cm.shape[0]):
        for j in range(test_cm.shape[1]):
            axes[1].text(j, i, format(test_cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if test_cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.show()

# Then use:
class_names = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
plot_two_confusion_matrices(train_cm, test_cm, class_names)
"""
