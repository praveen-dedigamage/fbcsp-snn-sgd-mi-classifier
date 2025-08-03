import os
import torch
import numpy as np
from types import MethodType
import ast
import matplotlib.pyplot as plt
import h5py


from eeg_csp_snn_classifier_ss_5_classes import (
    SNNClassifier, bandpass_filter, PairwiseCSP, RestVsOneCSP,
    encode_projected_signals_to_spikes
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DATA_DIR = r"C:\Users\USER\Desktop\fbcsp-snn-mi-classifier\fbcsp-snn-mi-classifier\Codes\Single_Subject_Models\New_Dataset"
TEST_DATA_DIR  = r"C:\Users\USER\Desktop\fbcsp-snn-mi-classifier\fbcsp-snn-mi-classifier\Codes\Single_Subject_Models\Dataset"

def load_data(subject_id, session_type='T'):
    if session_type == 'T':
        filename = f"EEG_restMI_split_3s_A0{subject_id}T.mat"
        dataset_path = os.path.join(TRAIN_DATA_DIR, filename)
    elif session_type == 'E':
        filename = f"EEG_python_ready_1250_sample_pntsA0{subject_id}E.mat"
        dataset_path = os.path.join(TEST_DATA_DIR, filename)
    else:
        raise ValueError("session_type must be 'T' (train) or 'E' (test)")

    print(f"Trying to open: {dataset_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with h5py.File(dataset_path, 'r') as f:
        X = f['X'][:]
        X = np.transpose(X, (2, 0, 1))
        y = f['y'][:].flatten()
    return X, y



def forward_with_hidden(self, x):
    mem1 = self.lif1.init_leaky()
    mem2 = self.lif2.init_leaky()
    spk1_rec, mem1_rec, spk2_rec, mem2_rec, in_rec = [], [], [], [], []
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

def plot_multi_raster(act, filename, y_eval, num_trials=1, save_folder=None):
    spikes_input = (act['input'] > 0).astype(int)
    spikes_hidden = (act['hidden_spikes'] > 0).astype(int)
    spikes_output = (act['output_spikes'] > 0).astype(int)
    n_trials = min(num_trials, spikes_input.shape[1])
    time_steps = spikes_input.shape[0]
    layers = [
        (spikes_input, 'Input'),
        (spikes_hidden, 'Hidden'),
        (spikes_output, 'Output'),
    ]
    for trial in range(n_trials):
        label = y_eval[trial] if y_eval is not None else '?'
        fig, axes = plt.subplots(3, 1, figsize=(15, 8), sharex=True)
        for i, (spike_arr, layer_name) in enumerate(layers):
            n_units = spike_arr.shape[2]
            ax = axes[i]
            for h in range(n_units):
                spike_times = np.where(spike_arr[:, trial, h])[0]
                ax.vlines(spike_times, h + 0.5, h + 1.5, color='k', linewidth=0.5)
            ax.set_ylim(0.5, n_units + 0.5)
            ax.set_ylabel(f'{layer_name} units')
            ax.set_title(f'{layer_name} Layer Raster')
        axes[-1].set_xlabel('Time step')
        plt.suptitle(f'Layer Spiking Rasters: {filename} | Trial {trial} | Label: {label}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_folder is not None:
            imgname = os.path.splitext(filename)[0] + f'_trial{trial}.png'
            savepath = os.path.join(save_folder, imgname)
            plt.savefig(savepath)
            print(f"Saved raster to: {savepath}")
            plt.close()
        else:
            plt.show()

def process_model_file(model_path, num_trials=1, save_folder=None, use_rest_vs_one=False):
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        params = checkpoint['params'] if 'params' in checkpoint else {}
        freq_bands = ast.literal_eval(params.get('freq_bands', '[(4,10),(10,14),(14,30)]'))
        subject_id = int(params.get('subject_id', 1))
        base_dir = os.path.dirname(os.path.abspath(model_path))
        session_type = 'E'
        X_eval, y_eval = load_data(subject_id, session_type=session_type)
        X_eval_filtered = []
        for low, high in freq_bands:
            X_eval_filtered.append(bandpass_filter(X_eval, low, high))
        X_eval_filtered = np.concatenate(X_eval_filtered, axis=1)

        # --- Fit CSP with train data just like training ---
        X_train, y_train = load_data(subject_id, session_type='T')
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
        X_train_filtered = []
        for low, high in freq_bands:
            X_train_filtered.append(bandpass_filter(X_train_balanced, low, high))
        X_train_filtered = np.concatenate(X_train_filtered, axis=1)
        CSP_Components_Per_band = int(params.get('CSP_Compenents_Per_band', 22))
        n_components = len(freq_bands) * CSP_Components_Per_band

        # Use RestVsOneCSP or PairwiseCSP depending on flag
        if use_rest_vs_one:
            csp = RestVsOneCSP(
                n_components=n_components,
                reg_lambda=float(params.get('lambda_R', 0.01))
            )
            print("RVS")
        else:
            csp = PairwiseCSP(
                n_components=n_components,
                selected_classes=[0, 1, 2, 3, 4],
                reg_lambda=float(params.get('lambda_R', 0.01))
            )
            print("PW")
        csp.fit(X_train_filtered, y_train_balanced)
        projected_eval = csp.transform(X_eval_filtered)

        spikes_eval = encode_projected_signals_to_spikes(
            projected_eval,
            base_thresh=float(params.get('base_thresh', 0.001)),
            adapt_inc=float(params.get('adapt_inc', 0.6)),
            decay=float(params.get('decay', 0.95))
        ).to(DEVICE)
        y_eval_tensor = torch.tensor(y_eval, dtype=torch.long)

        input_size = spikes_eval.shape[2]
        hidden_size = int(params.get('hidden_neurons', 64))
        population_per_class = int(params.get('population_per_class', 20))
        n_classes = 5
        output_size = n_classes*population_per_class
        beta = float(params.get('beta', 0.95))
        dropout_prob = float(params.get('dropout_prob', 0.5))
        model = SNNClassifier(input_size, hidden_size, output_size, population_per_class, beta, dropout_prob)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(DEVICE)
        model.eval()
        model.forward = MethodType(forward_with_hidden, model)
        with torch.no_grad():
            act = model(spikes_eval)
        plot_multi_raster(act, os.path.basename(model_path), y_eval, num_trials=num_trials, save_folder=save_folder)
    except Exception as e:
        print(f"Failed to process {model_path}: {e}")

def find_pth_files(root_dir, recursive=True):
    pth_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith('.pth'):
                pth_files.append(os.path.join(dirpath, f))
        if not recursive:
            break
    return pth_files

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Batch visualize SNN .pth models')
    parser.add_argument(
    'root_dir', type=str, nargs='?', 
    default='Trained models/Subject_1',
    help='Folder containing .pth model files (default: Trained models/Subject_1)'
    )
    parser.add_argument('--trials', type=int, default=1, help='How many trials to plot')
    parser.add_argument('--save', type=str, default=None, help='Folder to save PNGs (if not set, will show plots)')
    parser.add_argument('--norecursive', action='store_true', help="Don't search subfolders")
    parser.add_argument('--restvsone', default=True, action='store_true', help="Use RestVsOneCSP instead of PairwiseCSP")
    args = parser.parse_args()
    if args.save is not None:
        os.makedirs(args.save, exist_ok=True)
    pth_files = find_pth_files(args.root_dir, recursive=(not args.norecursive))
    print(f"Found {len(pth_files)} .pth files in {args.root_dir}")
    for model_path in pth_files:
        print(f"Processing: {model_path}")
        process_model_file(
            model_path, num_trials=args.trials, save_folder=args.save,
            use_rest_vs_one=args.restvsone
        )
