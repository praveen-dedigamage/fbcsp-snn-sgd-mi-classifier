import os
import sys
import re
import ast
import torch
import numpy as np
from types import MethodType
import h5py
import matplotlib.pyplot as plt
from eeg_csp_snn_classifier_ss import SNNClassifier, bandpass_filter, PairwiseCSP, encode_projected_signals_to_spikes

# ----------- 1. Utility to extract parameters from filename -----------
def parse_filename(filename):
    """
    Extracts parameters from a model filename using regex.
    Supports both single and multiple frequency bands.
    """
    m = re.match(
        r'model_LR([0-9.eE+-]+)_FB(\[.*?\])_SP([0-9.]+)_BT([0-9.eE+-]+)_AI([0-9.eE+-]+)_D([0-9.]+)_HN([0-9]+)_NPC([0-9]+)_Sub([0-9]+)_weight_decay([0-9.eE+-]+)_dropout_prob([0-9.]+)_N_CSP_per_band([0-9]+)\.pth$',
        filename
    )
    if not m:
        raise ValueError(f"Could not parse filename: {filename}")
    lambda_R = float(m.group(1))
    freq_bands = ast.literal_eval(m.group(2))
    # always wrap as list if it's a tuple, so downstream is robust
    if isinstance(freq_bands, tuple):
        freq_bands = [freq_bands]
    spiking_prob = float(m.group(3))
    base_thresh = float(m.group(4))
    adapt_inc = float(m.group(5))
    decay = float(m.group(6))
    hidden_neurons = int(m.group(7))
    population_per_class = int(m.group(8))
    subject_id = int(m.group(9))
    weight_decay = float(m.group(10))
    dropout_prob = float(m.group(11))
    CSP_Components_Per_band = int(m.group(12))
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
        "dropout_prob": dropout_prob,
        "CSP_Components_Per_band": CSP_Components_Per_band
    }

# ----------- 2. Data loading (matches your training script) -----------
def load_data(base_dir, subject_id, session_type='E'):
    filename = f"EEG_python_ready_1250_sample_pntsA0{subject_id}{session_type}.mat"
    path = os.path.join(base_dir, 'Dataset', filename)
    with h5py.File(path, 'r') as f:
        X = f['X'][:]
        X = np.transpose(X, (2, 0, 1))
        y = f['y'][:].flatten()
    return X, y

# ----------- 3. Forward method to extract hidden layer membrane traces -----------
def forward_with_hidden(self, x):
    # For 3-layer SNN: input → hidden1 → hidden2 → output
    mem1 = self.lif1.init_leaky()
    mem2 = self.lif2.init_leaky()
    mem3 = self.lif3.init_leaky()
    spk1_rec, mem1_rec = [], []
    spk2_rec, mem2_rec = [], []
    spk3_rec, mem3_rec = [], []
    in_rec = []
    for t in range(x.size(0)):
        in_rec.append(x[t].detach().cpu().numpy())
        out1 = self.fc1(x[t])
        out1 = self.dropout1(out1)
        spk1, mem1 = self.lif1(out1, mem1)
        spk1_rec.append(spk1.detach().cpu().numpy())
        mem1_rec.append(mem1.detach().cpu().numpy())
        out2 = self.fc2(spk1)
        out2 = self.dropout2(out2)
        spk2, mem2 = self.lif2(out2, mem2)
        spk2_rec.append(spk2.detach().cpu().numpy())
        mem2_rec.append(mem2.detach().cpu().numpy())
        out3 = self.fc3(spk2)
        out3 = self.dropout3(out3)
        spk3, mem3 = self.lif3(out3, mem3)
        spk3_rec.append(spk3.detach().cpu().numpy())
        mem3_rec.append(mem3.detach().cpu().numpy())
    return {
        'input': np.stack(in_rec),
        'hidden1_spikes': np.stack(spk1_rec),
        'hidden1_mem': np.stack(mem1_rec),
        'hidden2_spikes': np.stack(spk2_rec),
        'hidden2_mem': np.stack(mem2_rec),
        'output_spikes': np.stack(spk3_rec),
        'output_mem': np.stack(mem3_rec),
    }

# ----------- 4. Main extraction loop -----------
model_directory = "Trained models/Subject_1"
base_dir = os.path.abspath(".")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------- Sort files by test accuracy (highest first) with Windows long path fix ---------
all_files = [f for f in os.listdir(model_directory) if f.endswith(".pth")]
file_acc = []
for f in all_files:
    file_path = os.path.join(model_directory, f)
    # Windows long-path fix
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
sorted_files = [f for f, acc in sorted(file_acc, key=lambda x: x[1], reverse=True)]

for filename in sorted_files:
    if not filename.endswith(".pth"):
        continue
    file_path = os.path.join(model_directory, filename)
    print(f"=== {filename} ===")
    params = parse_filename(filename)
    for key, value in params.items():
       print (f"{key} : {value}")

    # Windows long-path fix (if needed)
    if sys.platform.startswith("win"):
        abs_path = os.path.abspath(file_path)
        abs_path = abs_path.replace('/', '\\')
        if not abs_path.startswith(r'\\?\\'):
            file_path = r'\\?\\' + abs_path

    # Load checkpoint
    try:
        checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        print("Test accuracy:", checkpoint['test_acc'])
    except Exception as e:
        print(f"  → failed to load: {e}")
        continue

    # (1) Load and preprocess data with filename parameters
    X_eval, y_eval = load_data(base_dir, params["subject_id"], session_type='E')
    X_test_filtered = []
    for low, high in params["freq_bands"]:
        X_test_filtered.append(bandpass_filter(X_eval, low, high))
    X_test_filtered = np.concatenate(X_test_filtered, axis=1)
    
    n_components = int((X_test_filtered.shape[1] / X_eval.shape[1]) * params["CSP_Components_Per_band"])
    csp = PairwiseCSP(
        n_components=n_components,
        selected_classes=[1, 2, 3, 4],
        reg_lambda=params["lambda_R"]
    )
    csp.fit(X_test_filtered, y_eval)
    projected_val = csp.transform(X_test_filtered)
    spikes_val = encode_projected_signals_to_spikes(
        projected_val,
        base_thresh=params["base_thresh"],
        adapt_inc=params["adapt_inc"],
        decay=params["decay"]
    )

    y_eval_tensor = torch.tensor(y_eval - 1, dtype=torch.long)

    # (2) Restore model
    input_size = spikes_val.shape[2]
    hidden_size = params["hidden_neurons"]
    output_size = 4  # Or len(np.unique(y_eval))
    population_per_class = params["population_per_class"]
    beta = 0.95
    dropout_prob = params["dropout_prob"]

    # Must use SNNClassifier with three layers as defined previously
    model = SNNClassifier(input_size, hidden_size, output_size, population_per_class, beta, dropout_prob)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()

    # (3) Monkey-patch forward for activations
    model.forward = MethodType(forward_with_hidden, model)

    spikes_val = spikes_val.to(DEVICE)
    
    with torch.no_grad():
        act = model(spikes_val)  # act is now a dict with all activations
        print("Hidden1 activations shape:", act['hidden1_mem'].shape)
        print("Hidden2 activations shape:", act['hidden2_mem'].shape)
        print("Output activations shape:", act['output_mem'].shape)
        
        # (4) Save all activations if desired:
        # np.savez(f"{filename}_activations.npz", **act)
        
        def plot_multi_raster(act, filename, y_eval, num_trials=10):
            # All shapes: [time, batch, units]
            spikes_input = (act['input'] > 0).astype(int)
            spikes_hidden1 = (act['hidden1_spikes'] > 0).astype(int)
            spikes_hidden2 = (act['hidden2_spikes'] > 0).astype(int)
            spikes_output = (act['output_spikes'] > 0).astype(int)
            n_trials = min(num_trials, spikes_input.shape[1])
            time_steps = spikes_input.shape[0]
            layers = [
                (spikes_input, 'Input'),
                (spikes_hidden1, 'Hidden1'),
                (spikes_hidden2, 'Hidden2'),
                (spikes_output, 'Output'),
            ]
            for trial in range(n_trials):
                label = y_eval[trial] if y_eval is not None else '?'
                fig, axes = plt.subplots(len(layers), 1, figsize=(14, 12), sharex=True)
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
                plt.suptitle(f'All Layer Rasters: {filename} | Trial {trial} | Label: {label}')
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.show()

        plot_multi_raster(act, filename, y_eval, num_trials=10)

        input("Press Enter to continue…")
