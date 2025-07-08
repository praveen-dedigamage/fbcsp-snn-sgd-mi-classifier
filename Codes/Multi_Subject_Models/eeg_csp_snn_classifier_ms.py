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

# -------------------------- Logger --------------------------
def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
        logger.addHandler(handler)
    logger.propagate = False
    return logger

logger = setup_logger("LOSO")

# -------------------------- Device --------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# -------------------------- Bandpass Filter --------------------------
def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float = 250.0, order: int = 5) -> np.ndarray:
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered = np.zeros_like(data)
    n_trials, n_channels, n_samples = data.shape
    for trial in range(n_trials):
        for ch in range(n_channels):
            filtered[trial, ch, :] = filtfilt(b, a, data[trial, ch, :])
    return filtered

# -------------------------- Pairwise CSP --------------------------
class PairwiseCSP:
    def __init__(self, n_components: int = 2, selected_classes: Optional[List[int]] = None, reg_lambda: float = 0.01, verbose: bool = True):
        self.n_components = n_components
        self.selected_classes = selected_classes
        self.reg_lambda = reg_lambda
        self.pairwise_filters: Dict[Tuple[int, int], np.ndarray] = {}
        self.class_pairs: List[Tuple[int, int]] = []
        self.logger = setup_logger(self.__class__.__name__) if verbose else logging.getLogger()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PairwiseCSP":
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

            I = np.eye(cov1.shape[0])
            cov1 = (1 - self.reg_lambda) * cov1 + self.reg_lambda * I
            cov2 = (1 - self.reg_lambda) * cov2 + self.reg_lambda * I

            eigvals, eigvecs = eigh(cov1, cov1 + cov2)
            W = eigvecs[:, np.argsort(eigvals)[::-1]]
            self.pairwise_filters[(cl1, cl2)] = W[:, : self.n_components]
            self.logger.info(f"Computed CSP for classes {cl1} vs {cl2}")
        return self

    def transform(self, X: np.ndarray) -> Dict[Tuple[int, int], np.ndarray]:
        projected = {}
        for pair, W in self.pairwise_filters.items():
            projected[pair] = np.einsum('ij,tjk->tik', W.T, X)
        return projected

# -------------------------- Spike Encoding --------------------------
def encode_projected_signals_to_spikes(projected_data: Dict[Tuple[int, int], np.ndarray], base_thresh: float = 0.02, adapt_inc: float = 0.04, decay: float = 0.95, seed: Optional[int] = None) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)
    all_spikes = []
    for pair in sorted(projected_data.keys()):
        data = projected_data[pair]
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

def create_sparse_temporal_population_spikes(y, num_classes, population_per_class, num_steps, batch_size, target_spike_prob=0.7):
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

def compute_metrics(output_spikes, y_true, num_classes, population_per_class):
    time_steps, batch_size, total_outputs = output_spikes.shape
    summed = output_spikes.sum(dim=0)
    reshaped = summed.view(batch_size, num_classes, population_per_class)
    class_scores = reshaped.sum(dim=2)
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

class EarlyStopping:
    def __init__(self, patience=100, min_delta=1e-4, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_score):
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

def train_with_ideal_spikes(model, X_train, y_train, X_val, y_val, lr=1e-3, epochs=10, target_spike_prob=0.7, show_progress=False):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_incorrects, val_incorrects = [], []
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    num_classes = len(torch.unique(y_train))
    population_per_class = model.population_per_class
    num_steps, batch_size, _ = X_train.shape

    train_targets = create_sparse_temporal_population_spikes(y_train, num_classes, population_per_class, num_steps, batch_size, target_spike_prob)
    val_targets = create_sparse_temporal_population_spikes(y_val, num_classes, population_per_class, num_steps, X_val.shape[1], target_spike_prob)

    best_val_acc = 0.0
    best_state = None
    early_stopper = EarlyStopping(patience=100, min_delta=1e-4, mode='min')
    train_incorrect = 0.0

    from tqdm import tqdm
    epoch_iter = tqdm(range(1, epochs + 1), desc="Training", ncols=80) if show_progress else range(1, epochs + 1)
    for epoch in epoch_iter:
        start_time = time.time()
        model.train()
        optimizer.zero_grad()
        output_spikes = model(X_train)
        loss = van_rossum_loss(output_spikes, train_targets) + 0.0001 * train_incorrect
        loss.backward()
        optimizer.step()

        train_acc, train_incorrect = compute_metrics(output_spikes, y_train, num_classes, population_per_class)
        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = van_rossum_loss(val_output, val_targets)
            val_acc, val_incorrect = compute_metrics(val_output, y_val, num_classes, population_per_class)

        logger.info(
            f"Epoch {epoch} Train Loss: {loss.item():.4f}, Train Acc: {train_acc*100:.2f}%, Train Incorrect: {train_incorrect:.4f}, "
            f"Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc*100:.2f}%, Val Incorrect: {val_incorrect:.4f}"
            f"Vl = {loss}, train incorrect spike ratios = {train_incorrect}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            logger.info(f"Best model updated at epoch {epoch} with val_acc={val_acc:.4f}")

        elapsed = time.time() - start_time
        logger.info(f"Epoch {epoch} took {elapsed:.2f} seconds")

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        train_incorrects.append(train_incorrect)
        val_incorrects.append(val_incorrect)

        if epoch > 1000:
            early_stopper(val_incorrect)
            if early_stopper.early_stop:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_losses, train_accuracies, train_incorrects, val_losses, val_accuracies, val_incorrects

  
class SNNClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, population_per_class: int = 5, beta: float = 0.95):
        super().__init__()
        self.population_per_class = population_per_class
        self.total_outputs = output_size * population_per_class
        self.beta = beta
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=self.beta, spike_grad=surrogate.fast_sigmoid())
        self.fc2 = nn.Linear(hidden_size, self.total_outputs)
        self.lif2 = snn.Leaky(beta=self.beta, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec = []
        for t in range(x.size(0)):
            spk1, mem1 = self.lif1(self.fc1(x[t]), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            spk2_rec.append(spk2)
        return torch.stack(spk2_rec)

def evaluate(model, X_eval, y_eval):
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
def load_data(base_dir: str, subject_id: int, session_type: str = 'T') -> Tuple[np.ndarray, np.ndarray]:
    filename = f"EEG_python_ready_1250_sample_pntsA0{subject_id}{session_type}.mat"
    path = os.path.join(base_dir, 'Dataset', filename)
    with h5py.File(path, 'r') as f:
        X = f['X'][:]
        X = np.transpose(X, (2, 0, 1))
        y = f['y'][:].flatten()
    return X, y

def load_all_subjects(subject_ids, base_dir, freq_bands):
    all_X, all_y = [], []
    for sid in subject_ids:
        for session_type in ["T", "E"]:
            X, y = load_data(base_dir, sid, session_type)
            band_filtered = [bandpass_filter(X, low, high) for (low, high) in freq_bands]
            X_filtered = np.concatenate(band_filtered, axis=1)
            all_X.append(X_filtered)
            all_y.append(y)
    return np.concatenate(all_X), np.concatenate(all_y)

# -------------------------- Main --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LOSO SNN Training")
    parser.add_argument('--progress_bar', action='store_true', help="Show progress bar during training")
    parser.add_argument('--lambda_R', type=float, default=0.0001)
    parser.add_argument('--freq_bands', type=str, default='[(4,10),(10,14),(14,30)]')
    parser.add_argument('--spiking_prob', type=float, default=0.7)
    parser.add_argument('--base_thresh', type=float, default=0.001)
    parser.add_argument('--adapt_inc', type=float, default=0.6)
    parser.add_argument('--decay', type=float, default=0.95)
    parser.add_argument('--hidden_neurons', type=int, default=64)
    parser.add_argument('--population_per_class', type=int, default=20)
    parser.add_argument('--base_dir', type=str, default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument('--test_sub', type=int, default=9)
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))

    freq_bands = ast.literal_eval(args.freq_bands)
    all_subjects = list(range(1, 10))
    train_subs = [s for s in all_subjects if s != args.test_sub]

    X_train, y_train = load_all_subjects(train_subs, base_dir, freq_bands)
    X_test, y_test = load_all_subjects([args.test_sub])
    
    X_train_clipped = X_train[:, :, 625:1125]
    X_test_clipped = X_test[:, :, 625:1125]

    csp = PairwiseCSP(n_components=X_train_clipped.shape[1], selected_classes=[1, 2, 3, 4], reg_lambda=args.lambda_R)
    # CSP weights filename now includes frequency bands and lambda
    csp_weights_filename = f"pairwise_filters_FB{args.freq_bands}_Lambda{args.lambda_R}.npy"
    csp_weights_dir = os.path.join(args.base_dir, "csp_weights", f"subject_{args.test_sub}")
    os.makedirs(csp_weights_dir, exist_ok=True)
    csp_weights_path = os.path.join(csp_weights_dir, csp_weights_filename)
    
    if os.path.exists(csp_weights_path):
        logger.info(f"Loading existing CSP weights for Subject {args.test_sub}, FB: {args.freq_bands}, Lambda: {args.lambda_R} from {csp_weights_path}")
        csp.pairwise_filters = np.load(csp_weights_path, allow_pickle=True).item()
    else:
        csp.fit(X_train_clipped, y_train)
        np.save(csp_weights_path, csp.pairwise_filters)
        logger.info(f"Saved CSP weights for Subject {args.test_sub}, FB: {args.freq_bands}, Lambda: {args.lambda_R} to {csp_weights_path}")

    projected_train = csp.transform(X_train_clipped)
    projected_test = csp.transform(X_test_clipped)

    spikes_train = encode_projected_signals_to_spikes(projected_train, args.base_thresh, args.adapt_inc, args.decay).to(DEVICE)
    spikes_test = encode_projected_signals_to_spikes(projected_test, args.base_thresh, args.adapt_inc, args.decay).to(DEVICE)

    y_train_tensor = torch.tensor(y_train - 1, dtype=torch.long, device=DEVICE)
    y_test_tensor = torch.tensor(y_test - 1, dtype=torch.long, device=DEVICE)

    model = SNNClassifier(input_size=spikes_train.shape[2], hidden_size=args.hidden_neurons, output_size=len(np.unique(y_train)), population_per_class=args.population_per_class)

    #progress_bar = args.progress_bar
    progress_bar = True

    model_history = train_with_ideal_spikes(model, spikes_train, y_train_tensor, spikes_test, y_test_tensor, lr=1e-3, epochs=2000, target_spike_prob=args.spiking_prob, show_progress=progress_bar)
    train_acc, train_cm = evaluate(model_history[0], spikes_train, y_train_tensor)
    test_acc, test_cm = evaluate(model_history[0], spikes_test, y_test_tensor)
    logger.info(f"Subject {args.test_sub} Accuracy: {test_acc * 100:.2f}%")

    model_filename = (
        f"Multiple_Subject_model_LR{args.lambda_R}_FB{args.freq_bands}_SP{args.spiking_prob}_"
        f"BT{args.base_thresh}_AI{args.adapt_inc}_D{args.decay}_"
        f"HN{args.hidden_neurons}_NPC{args.population_per_class}_TestSub{args.test_sub}.pth"
    )
    save_dir = os.path.join(args.base_dir, "loso_models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, model_filename)
    #torch.save(model_history, save_path)
    #return model, train_losses, train_accuracies, train_incorrects, val_losses, val_accuracies, val_incorrects
    torch.save({
        'model_state_dict': model_history[0].state_dict(),
        'train_losses': model_history[1],
        'train_accuracies':model_history[2],
        'train_incorrect_ratios':model_history[3],
        'test_losses': model_history[4],
        'test_accuracies': model_history[5],
        'test_incorrect_ratios': model_history[6],
        'train_cm': train_cm,
        'test_cm': test_cm,
        'params': vars(args)
    }, save_path)
    logger.info(f"Saved best model for Subject {args.test_sub} to {save_path}")


    
    
    