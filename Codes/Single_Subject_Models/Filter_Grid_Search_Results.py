import os
import sys
import re
import ast
import torch
import numpy as np
from types import MethodType
import h5py
import matplotlib.pyplot as plt
from eeg_csp_snn_classifier_ss_5_classes import SNNClassifier, bandpass_filter, PairwiseCSP, RestVsOneCSP, encode_projected_signals_to_spikes, load_data

# ===== PATCH RestVsOneCSP.transform locally (no changes to training code) =====
# Patch immediately after import
_original_transform = RestVsOneCSP.transform

def patched_transform(self, X):
    projected = {}
    for mi_class, W in self.filters.items():
        print(f"[CSP DEBUG] mi_class: {mi_class}  W shape: {W.shape}  W.T shape: {W.T.shape}  X shape: {X.shape}")
        # Check channel alignment
        assert W.shape[0] == X.shape[1], (
            f"CSP filter channels ({W.shape[0]}) != Data channels ({X.shape[1]})"
        )
        projected[mi_class] = np.einsum('ac,tcb->tab', W.T, X)
    return projected

RestVsOneCSP.transform = patched_transform
# ============================================================================

# ----------- 1. Utility to extract parameters from filename -----------
def parse_filename(filename):
    """
    Extracts parameters from a model filename using regex.
    Specifically handles frequency band format: _FB[(a, b)]
    """
    m = re.match(
        r'model_LR([0-9.eE+-]+)_FB\[(\(.*?\))\]_SP([0-9.]+)_BT([0-9.eE+-]+)_AI([0-9.eE+-]+)_D([0-9.]+)'
        r'_HN([0-9]+)_NPC([0-9]+)_Sub([0-9]+)_weight_decay([0-9.eE+-]+)_dropout_prob([0-9.]+)\.pth$',
        filename
    )

    if not m:
        raise ValueError(f"Could not parse filename: {filename}")

    lambda_R = float(m.group(1))
    freq_bands = [ast.literal_eval(m.group(2))]  # convert string tuple and wrap in list
    spiking_prob = float(m.group(3))
    base_thresh = float(m.group(4))
    adapt_inc = float(m.group(5))
    decay = float(m.group(6))
    hidden_neurons = int(m.group(7))
    population_per_class = int(m.group(8))
    subject_id = int(m.group(9))
    weight_decay = float(m.group(10))
    dropout_prob = float(m.group(11))

    return {
        "lambda_R": lambda_R,
        "freq_bands": freq_bands,
        "spiking_prob": spiking_prob,
        "base_thresh": base_thresh,
        "adapt_inc": adapt_inc,
        "decay": decay,
        "hidden_neurons": hidden_neurons,
        "population_per_class": population_per_class,
        "subject_id": subject_id,
        "weight_decay": weight_decay,
        "dropout_prob": dropout_prob
    }

# ----------- 2. Data loading (matches your training script) -----------
# ----------- 3. Forward method to extract hidden layer membrane traces -----------
def forward_with_hidden(self, x):
    mem1 = self.lif1.init_leaky()
    mem2 = self.lif2.init_leaky()
    spk1_rec = []
    mem1_rec = []
    spk2_rec = []
    mem2_rec = []
    in_rec = []
    for t in range(x.size(0)):
        in_rec.append(x[t].detach().cpu().numpy())
        out1 = self.fc1(x[t])
        out1_dropped = self.dropout1(out1)
        spk1, mem1 = self.lif1(out1_dropped, mem1)
        spk1_rec.append(spk1.detach().cpu().numpy())
        mem1_rec.append(mem1.detach().cpu().numpy())
        out2 = self.fc2(spk1)
        out2_dropped = self.dropout2(out2)
        spk2, mem2 = self.lif2(out2_dropped, mem2)
        spk2_rec.append(spk2.detach().cpu().numpy())
        mem2_rec.append(mem2.detach().cpu().numpy())
    return {
        'input': np.stack(in_rec),
        'hidden_spikes': np.stack(spk1_rec),
        'hidden_mem': np.stack(mem1_rec),
        'output_spikes': np.stack(spk2_rec),
        'output_mem': np.stack(mem2_rec),
    }

# ----------- 4. Main extraction loop -----------
model_directory = "Trained models/Subject_2"
base_dir = os.path.abspath(".")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------- Sort files by test accuracy (highest first) with Windows long path fix ---------
all_files = [f for f in os.listdir(model_directory) if f.endswith(".pth")]
file_acc = []
for f in all_files:
    file_path = os.path.join(model_directory, f)
    if sys.platform.startswith("win"):
        abs_path = os.path.abspath(file_path)
        abs_path = abs_path.replace('/', '\\')
        if not abs_path.startswith(r'\\?\\'):
            file_path = r'\\?\\' + abs_path
    try:
        checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        acc = checkpoint.get("test_acc", 0.0)
        file_acc.append((f, acc))
    except Exception as e:
        print(f"Failed to load {f}: {e}")

# Sort by test accuracy, descending
sorted_files = [(f, acc) for f, acc in sorted(file_acc, key=lambda x: x[1], reverse=True)]

for filename, acc in sorted_files:
    if not filename.endswith(".pth"):
        continue
    file_path = os.path.join(model_directory, filename)
    print(f"=== {filename} ===")
    params = parse_filename(filename)
    for key, value in params.items():
        print(f"{key} : {value}")

    if sys.platform.startswith("win"):
        abs_path = os.path.abspath(file_path)
        abs_path = abs_path.replace('/', '\\')
        if not abs_path.startswith(r'\\?\\'):
            file_path = r'\\?\\' + abs_path

    try:
        checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        print(checkpoint['test_acc'])
    except Exception as e:
        print(f"  → failed to load: {e}")
        continue

    freq_bands = params["freq_bands"]
    X_train, y_train = load_data(base_dir, params["subject_id"], session_type='T')
    rest_indices = np.where(y_train == 0)[0]
    mi_indices = np.where(y_train > 0)[0]
    mi_class_counts = [np.sum(y_train == cls) for cls in range(1, 5)]
    min_per_class = min(mi_class_counts)
    n_rest = min_per_class
    np.random.seed(42)
    selected_rest_indices = np.random.choice(rest_indices, size=n_rest, replace=False)
    balanced_indices = np.concatenate([selected_rest_indices, mi_indices])
    np.random.shuffle(balanced_indices)
    X_train_balanced = X_train[balanced_indices]
    y_train_balanced = y_train[balanced_indices]
    
    bands = params["freq_bands"]
    if isinstance(bands[0][0], int):
        freq_band_list = bands
    elif isinstance(bands[0][0], tuple):
        freq_band_list = list(bands[0])
    else:
        raise ValueError("Unrecognized freq_bands structure")
        
    X_train_filtered = []
    
    for low, high in freq_band_list:
        X_train_filtered.append(bandpass_filter(X_train_balanced, low, high))
    X_train_filtered = np.concatenate(X_train_filtered, axis=1)
    
    X_eval, y_eval = load_data(base_dir, params["subject_id"], session_type='E')
        
    X_test_filtered = []
    for low, high in freq_band_list:
        X_test_filtered.append(bandpass_filter(X_eval, low, high))
    X_test_filtered = np.concatenate(X_test_filtered, axis=1)
    X_test_filtered = X_test_filtered[:, :, :]
    print("############")
    
    n_components = int((X_test_filtered.shape[1] / X_eval.shape[1]) * 22)
    
    L_R = float(input("Enter the correct Value"))
    
    csp = PairwiseCSP(
        n_components=n_components,
        selected_classes=[0, 1, 2, 3, 4],
        reg_lambda=L_R
        )
    """

    csp = RestVsOneCSP(
        n_components=n_components,
        reg_lambda=params["lambda_R"]
    )
    """
    csp.fit(X_train_filtered, y_train_balanced)
    projected_val = csp.transform(X_test_filtered)
    spikes_val = encode_projected_signals_to_spikes(
        projected_val,
        base_thresh=params["base_thresh"],
        adapt_inc=params["adapt_inc"],
        decay=params["decay"]
    )
    y_eval_tensor = torch.tensor(y_eval, dtype=torch.long)
    input_size = spikes_val.shape[2]
    hidden_size = params["hidden_neurons"]
    output_size = 5
    population_per_class = params["population_per_class"]
    beta = 0.95
    dropout_prob = params["dropout_prob"]
    model = SNNClassifier(input_size, hidden_size, output_size, population_per_class, beta, dropout_prob)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    model.forward = MethodType(forward_with_hidden, model)
    spikes_val = spikes_val.to(DEVICE)
    with torch.no_grad():
        act = model(spikes_val)
        print("Hidden activations shape:", act['hidden_mem'].shape)
        def plot_multi_raster(act, filename, y_eval):
            spikes_input = (act['input'] > 0).astype(int)
            spikes_hidden = (act['hidden_spikes'] > 0).astype(int)
            spikes_output = (act['output_spikes'] > 0).astype(int)
            n_trials = int(input("How Many Trials to Plot"))
            time_steps = spikes_input.shape[0]
            layers = [
                (spikes_input, 'Input'),
                (spikes_hidden, 'Hidden'),
                (spikes_output, 'Output'),
            ]
            correct_results = 0
            true_labels = []
            pred_labels = []
            
            Do_Plots = True
            
            for trial in range(y_eval.shape[0]):
                
                print(f'\n--- Output Spike Counts (Grouped by 20) for Trial {trial} ---')
                spike_arr = spikes_output  # only output layer
                n_units = spike_arr.shape[2]
                spike_counts = np.sum(spike_arr[:, trial, :], axis=0)  # shape: (n_units,)
                
                n_groups = (n_units + 19) // 20  # ceil division
                
                max_group_sum = -1
                max_group_idx = -1
                
                for group_idx in range(n_groups):
                    start = group_idx * 20
                    end = min(start + 20, n_units)
                    group_sum = np.sum(spike_counts[start:end])
                    print(f'  Group {group_idx + 1} (Units {start}–{end - 1}): {group_sum} spikes')
                    
                    if group_sum > max_group_sum:
                        max_group_sum = group_sum
                        max_group_idx = group_idx
                
                label = y_eval[trial] if y_eval is not None else '?'
                prediction = max_group_idx
                
                true_labels.append(label)
                pred_labels.append(prediction)

                
                if label == prediction:
                    correct_results = correct_results + 1
                    
                if Do_Plots == True:
                
                    for j in range(n_trials):
                        fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
                        for i, (spike_arr, layer_name) in enumerate(layers):
                            n_units = spike_arr.shape[2]
                            ax = axes[i]
                            for h in range(n_units):
                                spike_times = np.where(spike_arr[:, trial, h])[0]
                                ax.vlines(spike_times, h + 0.5, h + 1.5, color='k', linewidth=0.5)
                            ax.set_ylim(0.5, n_units + 0.5)
                            ax.set_ylabel(f'{layer_name} unit')
                            ax.set_title(f'{layer_name} Layer Raster')
                        axes[-1].set_xlabel('Time step')
                        plt.suptitle(f'All Layer Rasters: {filename} | Trial {trial} | Label: {label} | Pred: {max_group_idx}')
                        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                        plt.show()
                        if j+1 == n_trials:
                            Do_Plots = False
                    
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

            cm = confusion_matrix(true_labels, pred_labels)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plt.show()


            print(correct_results)
            print(n_trials)
                
        plot_multi_raster(act, filename, y_eval)
        input("Press Enter to continue…")
