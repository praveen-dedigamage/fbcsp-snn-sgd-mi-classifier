import os
import sys
import time
import ast
import logging
import argparse
from typing import List, Optional, Tuple, Dict

import h5py
import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from sklearn.metrics import accuracy_score, confusion_matrix
from itertools import combinations
from scipy.signal import butter, filtfilt
from scipy.linalg import eigh


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
        logger.addHandler(handler)
    logger.propagate = False
    return logger

logger = setup_logger(__name__)

# -------------------------- Device --------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")


# -------------------------- Bandpass Filter --------------------------

def bandpass_filter( data, lowcut, highcut, fs = 250.0, order= 5):
    """
    Apply a Butterworth bandpass filter to each channel of each trial.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    
    filtered = np.zeros_like(data)
    
    n_trials, n_channels, n_samples = data.shape
    
    for trial in range(n_trials):
        for ch in range(n_channels):
            signal = data[trial, ch, :]
            filtered[trial, ch, :] = filtfilt(b, a, signal)
            
    return filtered


# -------------------------- Pairwise CSP --------------------------

class PairwiseCSP:
    def __init__(
        self,
        n_components: int = 2,
        selected_classes: Optional[List[int]] = None,
        reg_lambda: float = 0.01,
        verbose: bool = True
    ):
        self.n_components = n_components
        self.selected_classes = selected_classes
        self.reg_lambda = reg_lambda
        self.pairwise_filters: Dict[Tuple[int, int], np.ndarray] = {}
        self.class_pairs: List[Tuple[int, int]] = []
        self.logger = setup_logger(self.__class__.__name__) if verbose else logging.getLogger()
    
    def fit(self, X, y):
        """
        Fit CSP filters for each pair of classes.
        """
        classes = np.unique(y)
        if self.selected_classes is not None:
            classes = self.selected_classes
        self.class_pairs = list(combinations(classes, 2))

        for cl1, cl2 in self.class_pairs:
            idx = np.where((y == cl1) | (y == cl2))[0]
            X_pair = X[idx]
            y_pair = y[idx]

            covs = [np.cov(trial) for trial in X_pair]
            covs = np.array([cov / np.trace(cov) for cov in covs])
            cov1 = np.mean(covs[y_pair == cl1], axis=0)
            cov2 = np.mean(covs[y_pair == cl2], axis=0)

            # Regularization
            I = np.eye(cov1.shape[0])
            cov1 = (1 - self.reg_lambda) * cov1 + self.reg_lambda * I
            cov2 = (1 - self.reg_lambda) * cov2 + self.reg_lambda * I

            # Solve generalized eigenvalue problem
            eigvals, eigvecs = eigh(cov1, cov1 + cov2)
            W = eigvecs[:, np.argsort(eigvals)[::-1]]

            # Keep n_components spatial filters
            self.pairwise_filters[(cl1, cl2)] = W[:, : self.n_components]
            self.logger.info(f"Computed CSP for classes {cl1} vs {cl2}")

        return self

    def transform(self, X: np.ndarray):
        """
        Project data onto CSP subspaces for each class pair.
        Returns a dict mapping class‐pairs to projected signals of shape (trials, n_components, time).
        """
        projected = {}
        
        for pair, W in self.pairwise_filters.items():
            projected[pair] = np.einsum('ij,tjk->tik', W.T, X)
            
        return projected


# -------------------------- Spike Encoding --------------------------

def encode_projected_signals_to_spikes(projected_data, base_thresh = 0.02, adapt_inc = 0.04, decay = 0.95, seed = None):
    """
    Convert continuous projected signals into spike trains.
    Returns a tensor of shape (time_steps, batch_size, total_channels).
    """
    if seed is not None:
        torch.manual_seed(seed)

    all_spikes: List[torch.Tensor] = []
    for pair in sorted(projected_data.keys()):
        data = projected_data[pair]  # (trials, components, time)
        trials, components, time_steps = data.shape
        tensor_data = torch.tensor(data, dtype=torch.float32, device=DEVICE).permute(2, 0, 1)

        spikes = torch.zeros_like(tensor_data)
        thresholds = torch.full((trials, components), base_thresh, device=DEVICE)

        for t in range(1, time_steps):
            delta = (tensor_data[t] - tensor_data[t - 1]).abs()
            spike_t = (delta > thresholds).float()
            spikes[t] = spike_t
            thresholds = thresholds * decay + spike_t * adapt_inc

        all_spikes.append(spikes)

    return torch.cat(all_spikes, dim=2)


# -------------------------- SNN Model --------------------------
    
class SNNClassifier(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        population_per_class = 5,
        beta= 0.95,
        dropout_prob = 0.5, # New parameter for dropout probability
    ):
        super().__init__()
        self.population_per_class = population_per_class
        self.total_outputs = output_size * population_per_class
        self.beta = beta
        self.dropout_prob = dropout_prob

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(self.dropout_prob) # Dropout layer 1
        self.lif1 = snn.Leaky(beta=self.beta, spike_grad=surrogate.fast_sigmoid())

        self.fc2 = nn.Linear(hidden_size, self.total_outputs)
        self.dropout2 = nn.Dropout(self.dropout_prob) # Dropout layer 2
        self.lif2 = snn.Leaky(beta=self.beta, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (time_steps, batch_size, input_size)
        Returns: Tensor of shape (time_steps, batch_size, total_outputs)
        """
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec: List[torch.Tensor] = []

        for t in range(x.size(0)):
            out1 = self.fc1(x[t])
            out1_dropped = self.dropout1(out1) # Apply dropout
            spk1, mem1 = self.lif1(out1_dropped, mem1)
            
            out2 = self.fc2(spk1)
            out2_dropped = self.dropout2(out2) # Apply dropout
            spk2, mem2 = self.lif2(out2_dropped, mem2)
            spk2_rec.append(spk2)

        return torch.stack(spk2_rec)


# -------------------------- Target Spikes and Loss --------------------------

def create_sparse_temporal_population_spikes(y,num_classes,population_per_class,num_steps,batch_size,target_spike_prob = 0.7,):
    """
    Generate ideal target spikes for supervised training.
    """
    total_outputs = num_classes * population_per_class
    ideal_spikes = torch.zeros((num_steps, batch_size, total_outputs), device=DEVICE)

    for sample_idx in range(batch_size):
        class_idx = int(y[sample_idx].item())
        start = class_idx * population_per_class
        end = start + population_per_class
        for t in range(num_steps):
            mask = torch.rand(population_per_class, device=DEVICE) < target_spike_prob
            ideal_spikes[t, sample_idx, start:end] = mask.float()

    return ideal_spikes


def van_rossum_convolution(spikes,tau,dt = 1.0,):
    """
    Apply Van Rossum kernel to spike trains.
    """
    alpha = dt / tau
    filtered = torch.zeros_like(spikes)
    filtered[0] = spikes[0]
    for t in range(1, spikes.shape[0]):
        filtered[t] = (1 - alpha) * filtered[t - 1] + alpha * spikes[t]
    return filtered


def van_rossum_loss(output_spikes,target_spikes, tau = 20.0, dt = 1.0,):
    f_pred = van_rossum_convolution(output_spikes, tau, dt)
    f_target = van_rossum_convolution(target_spikes, tau, dt)
    return torch.mean((f_pred - f_target) ** 2)


# -------------------------- Training and Evaluation --------------------------

class EarlyStopping:
    def __init__(
        self,
        patience = 25,
        min_delta = 1e-4,
        mode = 'max',
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score: Optional[float] = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_score: float):
        if self.best_score is None:
            self.best_score = current_score
        else:
            if (
                (self.mode == 'max' and current_score < self.best_score + self.min_delta) or
                (self.mode == 'min' and current_score > self.best_score - self.min_delta)
            ):
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = current_score
                self.counter = 0


def compute_metrics(output_spikes,y_true,num_classes,population_per_class):
    """
    Compute accuracy and incorrect spike ratio.
    """
    time_steps, batch_size, total_outputs = output_spikes.shape
    summed = output_spikes.sum(dim=0)  # (batch, total_outputs)
    reshaped = summed.view(batch_size, num_classes, population_per_class)
    class_scores = reshaped.sum(dim=2)  # (batch, num_classes)
    preds = torch.argmax(class_scores, dim=1)

    acc = float((preds == y_true).sum().item() / batch_size)

    total_spikes = summed.sum().item()
    incorrect_spikes = 0.0
    for i in range(batch_size):
        class_idx = int(y_true[i].item())
        start = class_idx * population_per_class
        end = start + population_per_class
        sample_spikes = summed[i]
        non_target = sample_spikes.clone()
        non_target[start:end] = 0
        incorrect_spikes += non_target.sum().item()

    incorrect_ratio = incorrect_spikes / (total_spikes + 1e-6)
    return acc, incorrect_ratio


def train_with_ideal_spikes( model, X_train, y_train, X_val, y_val, lr = 1e-3, epochs = 10, target_spike_prob = 0.7, weight_decay = 1e-2):
    """
    Train SNN model using Van Rossum supervised loss on ideal spikes.
    """
    print(weight_decay)
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    num_classes = len(torch.unique(y_train))
    population_per_class = model.population_per_class
    num_steps, batch_size, _ = X_train.shape

    # Generate target spikes
    train_targets = create_sparse_temporal_population_spikes(
        y_train, num_classes, population_per_class, num_steps, batch_size, target_spike_prob
    )
    test_targets = create_sparse_temporal_population_spikes(
        y_val, num_classes, population_per_class, num_steps, X_val.shape[1], target_spike_prob
    )

    early_stopper = EarlyStopping(patience=100, min_delta=1e-4, mode='max')

    history = {
        'train_losses': [],
        'test_losses': [],
        'train_accuracies': [],
        'test_accuracies': [],
        'train_incorrect_ratios': [],
        'test_incorrect_ratios': [],
    }

    best_test_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        optimizer.zero_grad()

        output_spikes = model(X_train)
        loss = van_rossum_loss(output_spikes, train_targets)
        loss.backward()
        optimizer.step()

        # Compute train metrics
        train_acc, train_incorrect = compute_metrics(
            output_spikes, y_train, num_classes, population_per_class
        )

        # Validation
        model.eval()
        with torch.no_grad():
            test_output = model(X_val)
            test_loss = van_rossum_loss(test_output, test_targets)
            test_acc, test_incorrect = compute_metrics(
                test_output, y_val, num_classes, population_per_class
            )

        history['train_losses'].append(loss.item())
        history['test_losses'].append(test_loss.item())
        history['train_accuracies'].append(train_acc)
        history['test_accuracies'].append(test_acc)
        history['train_incorrect_ratios'].append(train_incorrect)
        history['test_incorrect_ratios'].append(test_incorrect)

        logger.info(
            f"Epoch {epoch}/{epochs} "
            f"Train Loss: {loss.item():.4f}, Train Acc: {train_acc*100:.2f}%, "
            f"Val Loss: {test_loss.item():.4f}, Val Acc: {test_acc*100:.2f}%, "
            f"Time: {time.time() - epoch_start:.2f}s"
        )

        # Update best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_state = model.state_dict().copy()
            logger.info(f"Best model updated at epoch {epoch} with test_acc={test_acc:.4f}")

        # Early stopping
        if epoch > 1000:
            early_stopper(test_acc)
            if early_stopper.early_stop:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # Load best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return (
        model,
        history['train_losses'],
        history['train_accuracies'],
        history['train_incorrect_ratios'],
        history['test_losses'],
        history['test_accuracies'],
        history['test_incorrect_ratios'],
    )


def evaluate(modelr,X_eval,y_eval):
    """
    Evaluate model on data, returning accuracy and confusion matrix.
    """
    model.eval()
    X_eval = X_eval.to(DEVICE)
    y_eval = y_eval.to(DEVICE)
    num_classes = len(torch.unique(y_eval))
    population_per_class = model.population_per_class

    with torch.no_grad():
        output = model(X_eval)
        time_steps, batch_size, _ = output.shape
        summed = output.sum(dim=0)
        class_scores = summed.view(batch_size, num_classes, population_per_class).sum(dim=2)
        preds = torch.argmax(class_scores, dim=1)

    acc = accuracy_score(y_eval.cpu(), preds.cpu())
    cm = confusion_matrix(y_eval.cpu(), preds.cpu())
    return acc, cm


# -------------------------- Data Loading --------------------------

def load_data(base_dir, subject_id, session_type = 'T'):
    filename = f"EEG_python_ready_1250_sample_pntsA0{subject_id}{session_type}.mat"
    path = os.path.join(base_dir, 'Dataset', filename)
    with h5py.File(path, 'r') as f:
        X = f['X'][:]
        X = np.transpose(X, (2, 0, 1))
        y = f['y'][:].flatten()
    return X, y

# -------------------------- Main --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SNN with CSP preprocessing on EEG data."
    )
    parser.add_argument('lambda_R', type=float, nargs='?', default=0.0001, help="CSP regularization lambda")
    parser.add_argument('freq_bands', type=str, nargs='?', default='[(4,10),(10,14),(14,30)]', help="List of (low, high) frequency bands")
    parser.add_argument('spiking_prob', type=float, nargs='?', default=0.7, help="Target spike probability")
    parser.add_argument('base_thresh', type=float, nargs='?', default=0.001, help="Base threshold for spike encoding")
    parser.add_argument('adapt_inc', type=float, nargs='?', default=0.6, help="Adaptive threshold increment")
    parser.add_argument('decay', type=float, nargs='?', default=0.95, help="Threshold decay factor")
    parser.add_argument('hidden_neurons', type=int, nargs='?', default=64, help="Number of hidden neurons")
    parser.add_argument('population_per_class', type=int, nargs='?', default=20, help="Number of output neurons per class")
    parser.add_argument('subject_id', type=int, nargs='?', default=1, help="Subject ID to load data")
    parser.add_argument('CSP_Compenents_Per_band', type=int, nargs='?', default=22, help="Subject ID to load data")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, f"Trained single models/Subject_{args.subject_id}")
    os.makedirs(results_dir, exist_ok=True)

    # Parse frequency bands string
    freq_bands = ast.literal_eval(args.freq_bands)

    # Load train and validation data
    X_train, y_train = load_data(base_dir, args.subject_id, session_type='T')
    X_val, y_val = load_data(base_dir, args.subject_id, session_type='E')

    # Bandpass filtering for each frequency band
    X_train_filtered: List[np.ndarray] = []
    X_test_filtered: List[np.ndarray] = []
    
    for low, high in freq_bands:
        X_train_filtered.append(bandpass_filter(X_train, low, high))
        X_test_filtered.append(bandpass_filter(X_val, low, high))

    X_train_filtered = np.concatenate(X_train_filtered, axis=1)  # (samples, channels*bands, time)
    X_test_filtered = np.concatenate(X_test_filtered, axis=1)

    # Fit CSP
    csp = PairwiseCSP(
        n_components=(int((X_train_filtered.shape[1])/X_train.shape[1])*args.CSP_Compenents_Per_band),
        selected_classes=[1, 2, 3, 4],
        reg_lambda=args.lambda_R
    )
    csp.fit(X_train_filtered, y_train)
    projected_train = csp.transform(X_train_filtered)
    projected_val = csp.transform(X_test_filtered)

    # Encode to spikes
    spikes_train = encode_projected_signals_to_spikes(
        projected_train,
        base_thresh=args.base_thresh,
        adapt_inc=args.adapt_inc,
        decay=args.decay
    ).to(DEVICE)
    spikes_val = encode_projected_signals_to_spikes(
        projected_val,
        base_thresh=args.base_thresh,
        adapt_inc=args.adapt_inc,
        decay=args.decay
    ).to(DEVICE)

    # Prepare labels (zero-indexed)
    y_train_tensor = torch.tensor(y_train - 1, dtype=torch.long, device=DEVICE)
    y_test_tensor = torch.tensor(y_val - 1, dtype=torch.long, device=DEVICE)

    # Initialize model
    input_size = spikes_train.shape[2]
    hidden_size = args.hidden_neurons
    output_size = len(np.unique(y_train))
    population_per_class = args.population_per_class
    beta = 0.95
    dropout_prob = 0.5
    model = SNNClassifier(input_size, hidden_size, output_size, population_per_class, beta, dropout_prob)
    
    wd = 1e-1

    # Train
    (
        best_model,
        train_losses,
        train_accs,
        train_incorrects,
        test_losses,
        test_accs,
        test_incorrects
    ) = train_with_ideal_spikes(
        model,
        spikes_train,
        y_train_tensor,
        spikes_val,
        y_test_tensor,
        lr=1e-3,
        epochs=2,
        target_spike_prob=args.spiking_prob,
        weight_decay = wd
    )

    # Evaluate on train and validation sets
    train_acc, train_cm = evaluate(best_model, spikes_train, y_train_tensor)
    test_acc, test_cm = evaluate(best_model, spikes_val, y_test_tensor)
    logger.info(f"Final Train Acc: {train_acc*100:.2f}%, Test Acc: {test_acc*100:.2f}%")
    logger.info(f"weight_decay : {wd}")
    logger.info(f"dropout_prob : {dropout_prob}")

    # Save model and history
    save_name = (
        f"model_LR{args.lambda_R}_FB{freq_bands}_SP{args.spiking_prob}_"
        f"BT{args.base_thresh}_AI{args.adapt_inc}_D{args.decay}_"
        f"HN{args.hidden_neurons}_NPC{args.population_per_class}_Sub{args.subject_id}_weight_decay{wd}_dropout_prob{dropout_prob}.pth"
    )
    save_path = os.path.join(results_dir, save_name)
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accuracies': train_accs,
        'test_accuracies': test_accs,
        'train_incorrect_ratios': train_incorrects,
        'test_incorrect_ratios': test_incorrects,
        'train_cm': train_cm,
        'test_cm': test_cm,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'params': vars(args)
    }, save_path)
    logger.info(f"Saved model and history to {save_path}")