import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy.linalg import eigh
from tqdm import tqdm
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import spikegen
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

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

# Ensure all trials have the same number of samples
def clip_trials(eeg_data, start_idx=0, stop_idx = 1250):
    return np.array([trial[start_idx:stop_idx] for trial in eeg_data])

def bandpass_filter(data, lowcut, highcut, fs=250, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data, axis=0)

def compute_covariance_matrices(X):
    covariances = []
    for trial in X:
        cov = np.dot(trial, trial.T)
        cov /= np.trace(cov)
        covariances.append(cov)
    return np.mean(covariances, axis=0)

def CSP(X1, X2):
    # Compute average covariance matrices for both classes
    C1 = compute_covariance_matrices(X1)
    C2 = compute_covariance_matrices(X2)

    # Composite covariance matrix
    Cc = C1 + C2

    # Eigenvalue decomposition for whitening
    eigvals, eigvecs = np.linalg.eigh(Cc)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Whitening transform
    P = np.dot(np.diag(eigvals ** -0.5), eigvecs.T)

    # Whitened covariance matrix
    S1 = np.dot(P, np.dot(C1, P.T))

    # Eigen-decomposition of S1
    eigvals_s1, eigvecs_s1 = np.linalg.eigh(S1)
    order_s1 = np.argsort(eigvals_s1)[::-1]
    eigvecs_s1 = eigvecs_s1[:, order_s1]

    # CSP filters
    W = np.dot(eigvecs_s1.T, P)

    return W

def PW_CSP(X, y, num_classes=4):
    from itertools import combinations
    Ws = {}
    
    class_pairs = list(combinations(range(1, num_classes + 1), 2))
    
    for (c1, c2) in class_pairs:
        idx_c1 = np.where(y == c1)[0]
        idx_c2 = np.where(y == c2)[0]

        X_c1 = X[idx_c1]
        X_c2 = X[idx_c2]

        # Ensure equal number of trials per class for CSP
        min_len = min(len(X_c1), len(X_c2))
        X_c1 = X_c1[:min_len]
        X_c2 = X_c2[:min_len]

        # Compute CSP between class c1 and c2
        W = CSP(X_c1, X_c2)
        Ws[(c1, c2)] = W

    return Ws


def apply_CSP_trials(W, trials):
    """
    Apply CSP filter W to multiple EEG trials.

    Parameters:
        W: CSP projection matrix (channels x channels)
        trials: EEG trials (trials x channels x samples)
        m: Number of spatial filters to select from each side

    Returns:
        Enhanced trials using CSP (trials x selected_filters x samples)
    """
    enhanced_trials = np.array([np.dot(W, trial) for trial in trials])
    return enhanced_trials

def plot_overlapping_class_trial_variances(enhanced_trials, y, classes=[1, 2, 3, 4], colors=['blue', 'red', 'green', 'orange'], alpha=0.6):
    """
    Plots the mean signal value per channel for multiple class trials in the same figure with overlapping bars.

    Parameters:
        enhanced_trials (numpy.ndarray): The EEG data array.
        y (numpy.ndarray): The class labels corresponding to the trials.
        classes (list): The list of class labels to plot. Default is [1, 2, 3, 4].
        colors (list): The list of colors for each class. Default is ['blue', 'red', 'green', 'orange'].
        alpha (float): Transparency level for bars to visualize overlapping better. Default is 0.6.

    Returns:
        None
    """
    labels = [f"Class {c}" for c in classes]
    channels = np.arange(enhanced_trials.shape[2])  # FIXED

    plt.figure(figsize=(12, 6))

    for c, color, label in zip(classes, colors, labels):
        class_trials = enhanced_trials[y == c]
        class_trial_variances = np.var(class_trials, axis=1).mean(axis=0)

        # Overlapping bars at the same x-position
        plt.bar(channels, class_trial_variances, color=color, alpha=alpha, label=label)

    plt.xlabel("Channel Index")
    plt.ylabel("Mean Signal Value")
    plt.title("Mean Signal Value per Channel for Different Classes (Overlapping)")
    plt.xticks(channels, rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()
    


def visualize_spikes(spike_tensor, trial_index=0, num_neurons=50):
    """
    Plot a spike raster for a subset of neurons.
    Args:
        spike_tensor: shape (num_steps, batch, num_neurons)
        trial_index: which trial to visualize (0 to batch_size-1)
        num_neurons: number of neurons to visualize (across batch)
    """
    spike_tensor = spike_tensor[:, trial_index, :num_neurons]  # Take selected trial
    fig, ax = plt.subplots(figsize=(12, 6))
    total_spikes = spike_tensor.sum().item()
    print(f"Total spikes in trial {trial_index}:", total_spikes)
    for neuron_idx in range(spike_tensor.shape[1]):
        spike_times = torch.nonzero(spike_tensor[:, neuron_idx]).squeeze()
        if spike_times.numel() > 0:
            ax.scatter(spike_times.cpu(), torch.full_like(spike_times, neuron_idx), s=2)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Neuron index")
    ax.set_title(f"Spike Raster Plot for Trial {trial_index}")
    plt.show()
    
def encode_feature_tensor(feature_tensor, num_steps=100):
    """
    Normalizes and encodes the feature tensor using rate-based encoding.
    Returns a tensor of shape [num_trials, num_channels, num_steps].
    """
    num_trials, num_channels = feature_tensor.shape
    encoded_list = []

    for i in range(num_trials):
        feature_matrix = feature_tensor[i]
        min_val = feature_matrix.min()
        max_val = feature_matrix.max()
        normalized = (feature_matrix - min_val) / (max_val - min_val)
        encoded = spikegen.rate(normalized, num_steps=num_steps)
        encoded_list.append(encoded)

    return torch.stack(encoded_list)

def temp_plot_class_trials(encoded_tensor, trial_indices, class_labels):
    """
    encoded_tensor: Tensor of shape [num_trials, num_channels, num_steps] (already spike-encoded)
    trial_indices: List or tensor of 4 trial indices, one per class.
    class_labels: List or tensor of corresponding class labels (for titles).
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(4):
        trial_idx = trial_indices[i]
        label = class_labels[i]

        encoded = encoded_tensor[trial_idx]
        spikes, nid = encoded.nonzero(as_tuple=True)

        ax = axes[i]
        ax.scatter(spikes, nid, s=2, color='black')
        ax.set_title(f"Class {label} - Trial {trial_idx}")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Neuron ID")

    plt.tight_layout()
    plt.show()

def plot_all_trials_by_class(encoded_tensor, Y, limit = None):
    """
    Loops through all trials in Y, and for each unique group of 4 classes,
    finds one trial per class and plots them using `temp_plot_class_trials`.
    """
    # Convert Y to a tensor if it's not already
    if not isinstance(Y, torch.Tensor):
        Y = torch.tensor(Y)

    unique_classes = torch.unique(Y)
    class_to_trials = {cls.item(): (Y == cls).nonzero(as_tuple=True)[0].tolist() for cls in unique_classes}
    
    if limit == None :
        # Find the minimum number of available trials per class (for fair looping)
        min_trials_per_class = min(len(trials) for trials in class_to_trials.values())
    else:
        min_trials_per_class = limit

    for i in range(min_trials_per_class):
        trial_indices = []
        class_labels = []
        for cls in unique_classes:
            cls = cls.item()
            trial_indices.append(class_to_trials[cls][i])
            class_labels.append(cls)

        temp_plot_class_trials(encoded_tensor, trial_indices, class_labels)
        
def plot_spike_histogram(encoded_tensor, Y):
    """
    Plots class-specific histograms of spike counts per channel.
    X-axis: Channels
    Y-axis: Total spikes per channel aggregated over all trials in each class.
    """
    if not isinstance(Y, torch.Tensor):
        Y = torch.tensor(Y)

    unique_classes = torch.unique(Y)
    num_steps = encoded_tensor.shape[1]
    num_channels = encoded_tensor.shape[2]

    plt.figure(figsize=(12, 8))
    for i, cls in enumerate(unique_classes):
        cls = cls.item()
        indices = (Y == cls).nonzero(as_tuple=True)[0]
        cls_tensor = encoded_tensor[indices].transpose(1, 2)  # shape: [trials, channels, time]
        spike_counts_per_channel = cls_tensor.sum(dim=(0, 2))  # sum over trials and time
        plt.bar(torch.arange(num_channels) + i * 0.2,  # small offset for visual separation
                spike_counts_per_channel.cpu().numpy(),
                width=0.2, label=f"Class {cls}")

    plt.xlabel("Channel Index")
    plt.ylabel("Total Spike Count")
    plt.title("Spike Count per Channel by Class")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_spike_lines_per_class(encoded_tensor, Y):
    """
    Plots class-specific spike count per channel using line plots.
    Each line represents a class.
    """
    if not isinstance(Y, torch.Tensor):
        Y = torch.tensor(Y)

    unique_classes = torch.unique(Y)
    num_steps = encoded_tensor.shape[1]
    num_channels = encoded_tensor.shape[2]

    plt.figure(figsize=(12, 8))
    for cls in unique_classes:
        cls = cls.item()
        indices = (Y == cls).nonzero(as_tuple=True)[0]
        cls_tensor = encoded_tensor[indices].transpose(1, 2)  # shape: [trials, channels, time]
        spike_counts_per_channel = cls_tensor.sum(dim=(0, 2))  # sum over trials and time
        plt.plot(torch.arange(num_channels), spike_counts_per_channel.cpu().numpy(), label=f"Class {cls}")

    plt.xlabel("Channel Index")
    plt.ylabel("Total Spike Count")
    plt.title("Spike Count per Channel by Class (Line Plot)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
def prepare_data(encoded_tensor, y, test_size=0.2, random_state=42):
    # Reshape encoded_tensor for SNN input: [time, batch, features]
    encoded_tensor = encoded_tensor.permute(1, 0, 2)  # [time, trials, features]
    Y = torch.tensor(y) - 1  # Adjust labels to be 0-based

    # Train/test split
    train_indices, test_indices = train_test_split(torch.arange(encoded_tensor.shape[1]), test_size=test_size, stratify=Y, random_state=random_state)

    X_train = encoded_tensor[:, train_indices, :]
    X_test = encoded_tensor[:, test_indices, :]
    y_train = Y[train_indices]
    y_test = Y[test_indices]

    output_size = len(torch.unique(Y))

    return X_train, X_test, y_train, y_test, output_size

def create_random_ideal_spikes(y, num_classes, num_steps, batch_size, target_strength=1.0, distractor_strength=0.1):
    ideal_spikes = distractor_strength * torch.rand((num_steps, batch_size, num_classes))
    for i in range(batch_size):
        class_idx = y[i]
        ideal_spikes[:, i, class_idx] = target_strength * torch.bernoulli(torch.full((num_steps,), 0.9))
    return ideal_spikes


def train(model, X_train, y_train, X_val, y_val, LR = 1e-3, epochs=10):
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
        # Training
        optimizer.zero_grad()
        output_spikes = model(X_train)
        ideal_spikes = create_random_ideal_spikes(y_train, num_classes, num_steps, X_train.shape[1]).to(output_spikes.device)
        loss = loss_fn(output_spikes, ideal_spikes)
        loss.backward()
        optimizer.step()

        output_sum = output_spikes.sum(dim=0)
        predicted = torch.argmax(output_sum, dim=1)
        acc = accuracy_score(y_train.cpu(), predicted.cpu())

        loss_list.append(loss.item())
        accuracy_list.append(acc)

        # Evaluation
        model.eval()
        with torch.no_grad():
            val_output_spikes = model(X_val)
            val_output_sum = val_output_spikes.sum(dim=0)
            val_predicted = torch.argmax(val_output_sum, dim=1)
            val_acc = accuracy_score(y_val.cpu(), val_predicted.cpu())

            val_ideal = create_random_ideal_spikes(y_val, num_classes, num_steps, X_val.shape[1]).to(val_output_spikes.device)
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
    
def plot_curves(loss_list, accuracy_list, val_loss_list, val_accuracy_list):
    epochs = range(1, len(loss_list) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(epochs, loss_list, color='red', label='Train Loss')
    ax1.plot(epochs, val_loss_list, color='gold', label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    ax2.plot(epochs, accuracy_list, color='blue', label='Train Accuracy')
    ax2.plot(epochs, val_accuracy_list, color='green', label='Test Accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.tick_params(axis='y')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.title("Training and Testing Loss & Accuracy Curves")
    fig.tight_layout()
    plt.show()
    
def plot_confusion_matrices(cm_train, cm_test, class_names=None):
    """
    Plots side-by-side confusion matrices for training and testing data.

    Parameters:
        cm_train: Confusion matrix for training data (2D array)
        cm_test: Confusion matrix for testing data (2D array)
        class_names: list of class names (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot training confusion matrix
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names if class_names else 'auto',
                yticklabels=class_names if class_names else 'auto',
                ax=axes[0])
    axes[0].set_title('Training Confusion Matrix')
    axes[0].set_xlabel('Predicted Labels')
    axes[0].set_ylabel('True Labels')

    # Plot testing confusion matrix
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names if class_names else 'auto',
                yticklabels=class_names if class_names else 'auto',
                ax=axes[1])
    axes[1].set_title('Testing Confusion Matrix')
    axes[1].set_xlabel('Predicted Labels')
    axes[1].set_ylabel('True Labels')

    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    # Define Path
    base_directory = r'C:\Users\USER\Desktop\Extending the thesis\Dataset\BCICIV_2a_gdf'
    SubjectNo = input("Please Enter Subject No : ")
    filename = fr"new_A0{SubjectNo}T"  # Change dynamically
    file_path = fr'{base_directory}\{filename}.pkl'

    # Load data
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    eeg_signals_loaded = data["eeg_signals"]
    trial_labels_loaded = data["trial_labels"]

    # Define frequency bands for FBCSP
    #freq_bands = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)]
    
    freq_bands = [(4, 30)]

    # Apply filtering for each frequency band
    filtered_signals = {band: bandpass_filter(clip_trials(eeg_signals_loaded, 125, 750), band[0], band[1]) for band in freq_bands}
    
    X = np.concatenate(list(filtered_signals.values()), axis=-1)
    y = np.asarray(trial_labels_loaded).flatten()
    
    unique_labels = np.unique(trial_labels_loaded)

    # Compute CSP filters for each class
    Ws = PW_CSP(X, y, num_classes=len(unique_labels))

    # Choose class to enhance trials for (example: class 1)
    trials_to_enhance = X
    
    total_enhanced_trials = []
    
    for key in Ws:
        print(key)
        
        W = Ws[key]
        
        enhanced_trials = apply_CSP_trials(W, trials_to_enhance)
        
        plot_overlapping_class_trial_variances(enhanced_trials, y)
        
        total_enhanced_trials.append(enhanced_trials)

        
    # Flatten features for SNN input (e.g., variance over time per channel)
    feature_list = [np.var(class_data, axis=1) for class_data in total_enhanced_trials]
    combined_features = np.concatenate(feature_list, axis=1) 
    
    print("Feature tensor shape:", combined_features.shape)
    
    feature_tensor = torch.from_numpy(combined_features).float()

    # Rate encode the features
    encoded_tensor = encode_feature_tensor(feature_tensor, num_steps=50)
    
    #plot_all_trials_by_class(encoded_tensor, y, limit=2)
    
    #plot_spike_histogram(encoded_tensor, y)
    
    #plot_spike_lines_per_class(encoded_tensor, y)
    
    #Train/test split
    X_train, X_test, y_train, y_test, output_size = prepare_data(encoded_tensor, y)
    
    # Define model
    input_size = encoded_tensor.shape[2]  
    hidden_size = 128

    model = SNNClassifier(input_size, hidden_size, output_size)
    
    # Train and evaluate
    loss_list, accuracy_list, val_loss_list, val_accuracy_list = train(model, X_train, y_train, X_test, y_test, LR=1e-3, epochs=100)
    plot_curves(loss_list, accuracy_list, val_loss_list, val_accuracy_list)


    train_acc, train_cm = evaluate(model, X_train, y_train)
    test_acc, test_cm = evaluate(model, X_test, y_test)

    print(f"Train Accuracy: {train_acc * 100:.2f}%")
    print("Train Confusion Matrix:\n", train_cm)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    print("Test Confusion Matrix:\n", test_cm)
    
    plot_confusion_matrices(train_cm, test_cm, class_names=['Class 1', 'Class 2', 'Class 3', 'Class 4'])
    
    save = input("Do you want to save the model (Y/N) : ")
    
    if save == "Y" or save =="y":
        model_name = input("Enter a name for the model : ")
        model_save_path = fr'{base_directory}\{model_name}.pt'
        torch.save({
        "model_state_dict": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
        }, f"{model_save_path}")
            
        print(f"Model and data saved to {model_save_path}")
