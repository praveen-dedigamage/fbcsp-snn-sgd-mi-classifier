import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import re
import ast
from random import sample

# Define relative project root (directory of this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Default model checkpoint path (override with --model_path)
DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, 'loso_models', 'Multiple_Subject_model_LR0.0005_FB[(4, 10), (10, 14), (14, 30)]_SP0.6_BT0.5_AI0.8_D0.95_HN64_NPC20_TestSub2.pth')

# Import your model and preprocessing utilities
from eeg_csp_snn_classifier_ms import (
    SNNClassifier,
    load_all_subjects,
    PairwiseCSP,
    encode_projected_signals_to_spikes,
    bandpass_filter
)


def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    params = checkpoint['params']
    state_dict = checkpoint['model_state_dict']

    # Infer model dimensions directly from saved weights
    fc1_w = state_dict['fc1.weight']
    input_size = fc1_w.shape[1]
    hidden_size = fc1_w.shape[0]

    fc2_w = state_dict['fc2.weight']
    total_outputs = fc2_w.shape[0]
    population_per_class = params['population_per_class']
    output_size = total_outputs // population_per_class

    # recreate model with inferred sizes
    model = SNNClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        population_per_class=population_per_class,
        beta=params.get('beta', 0.95)
    )
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, params


def get_test_spikes(params, device):
    test_sub = params['test_sub']
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    freq_bands = ast.literal_eval(params['freq_bands'])

    X_test, y_test = load_all_subjects([test_sub], base_dir, freq_bands)
    X_test = X_test[:, :, 625:1125]

    csp_dir = os.path.join(SCRIPT_DIR, 'csp_weights', f"subject_{test_sub}")
    
    all_files = os.listdir(csp_dir)
    # only pick files that start with 'csp_filters_' and end with '.npy'
    print(f"pairwise_filters_FB{parsed['FB']}_Lambda{parsed['LR']}")
    candidates = [
        f for f in all_files
        if f.startswith(f"pairwise_filters_FB{parsed['FB']}_Lambda{parsed['LR']}") and f.endswith('.npy')
    ]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one matching .npy in {csp_dir}, found {len(candidates)}")
    csp_file = candidates[0]
    print(csp_file)
    csp = PairwiseCSP(
        n_components=X_test.shape[1],
        selected_classes=params.get('selected_classes', None),
        reg_lambda=params['lambda_R']
    )
    csp.pairwise_filters = np.load(os.path.join(csp_dir, csp_file), allow_pickle=True).item()

    projected = csp.transform(X_test)
    spikes = encode_projected_signals_to_spikes(
        projected,
        base_thresh=params['base_thresh'],
        adapt_inc=params['adapt_inc'],
        decay=params['decay'],
        seed=None
    )
    return spikes.cpu().numpy(), y_test


def plot_raster(spikes, trial_idx=0, title="Raster Plot"):
    time_steps, _, total_neurons = spikes.shape
    trial_spikes = spikes[:, trial_idx, :]
    fig, ax = plt.subplots(figsize=(10, 6))
    for neuron in range(total_neurons):
        spike_times = np.where(trial_spikes[:, neuron] > 0)[0]
        ax.scatter(spike_times, np.full_like(spike_times, neuron), s=2)
    ax.set(xlabel='Time step', ylabel='Neuron index', title=title)
    plt.tight_layout()
    plt.show()

def parse_model_string(s):
    """
    Parses a model configuration string and returns a dictionary of parameters.
    """
    pattern = re.compile(
        r'LR(?P<LR>[0-9\.e-]+)_FB\[(?P<FB>[^\]]+)\]_SP(?P<SP>[0-9\.]+)_BT(?P<BT>[0-9\.]+)_AI(?P<AI>[0-9\.]+)_D(?P<D>[0-9\.]+)_HN(?P<HN>\d+)_NPC(?P<NPC>\d+)_TestSub(?P<TestSub>\d+)'
    )
    m = pattern.search(s)
    if not m:
        raise ValueError("String does not match expected format")
    
    params = m.groupdict()
    # Convert numeric types
    params['LR'] = float(params['LR'])
    params['SP'] = float(params['SP'])
    params['BT'] = float(params['BT'])
    params['AI'] = float(params['AI'])
    params['D'] = float(params['D'])
    params['HN'] = int(params['HN'])
    params['NPC'] = int(params['NPC'])
    params['TestSub'] = int(params['TestSub'])
    # Convert FB list of tuples
    params['FB'] = eval(f"[{params['FB']}]")
    
    return params

def plot_combined_rasters(input_spikes, hidden_spikes, output_spikes, true_label, trial_idx=0):
    datasets = [input_spikes, hidden_spikes, output_spikes]
    titles = ['Encoded Input Spikes', 'Hidden Layer Spikes', 'Model Output Spikes']
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    # Add a supertitle with the true label
    fig.suptitle(f"True Label: {true_label}", fontsize=16)

    for ax, spikes, title in zip(axes, datasets, titles):
        time_steps, _, total_neurons = spikes.shape
        trial_spikes = spikes[:, trial_idx, :]
        for neuron in range(total_neurons):
            spike_times = np.where(trial_spikes[:, neuron] > 0)[0]
            ax.scatter(spike_times, np.full_like(spike_times, neuron), s=2)
        ax.set_ylabel('Neuron index')
        ax.set_title(title)
    axes[-1].set_xlabel('Time step')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # adjust for suptitle
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load .pth model and plot raster of test spikes")
    parser.add_argument('--model_path', type=str,
                        help="Optional path to the .pth checkpoint file (defaults to DEFAULT_MODEL_PATH)")
    parser.add_argument('--trial', type=int, default=0, help="Trial index to plot")
    args = parser.parse_args()
    
    # Use CLI arg if provided, otherwise fall back to default
    model_path = args.model_path if args.model_path else DEFAULT_MODEL_PATH
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
    parsed = parse_model_string(model_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, params = load_checkpoint(model_path, device)

    spikes, labels = get_test_spikes(params, device)
    spike_tensor = torch.tensor(spikes, device=device)

    # forward through model to confirm reproducibility
    with torch.no_grad():
        output_spikes = model(spike_tensor).cpu().numpy()
        
    with torch.no_grad():
        mem1 = model.lif1.init_leaky()
        hidden_rec = []
        for t in range(spike_tensor.size(0)):
            spk1, mem1 = model.lif1(model.fc1(spike_tensor[t]), mem1)
            hidden_rec.append(spk1.cpu().numpy())
    hidden_spikes = np.stack(hidden_rec)

    # plot raster for raw encoded spikes
    #plot_raster(spikes, trial_idx=args.trial, title='Encoded Input Spikes')
    #plot_raster(hidden_spikes, trial_idx=args.trial, title='Hidden Layer Spikes')
    #plot_raster(output_spikes, trial_idx=args.trial, title='Model Output Spikes')
    
    for i in sample(range(len(labels)), 10):
        plot_combined_rasters(spikes, hidden_spikes, output_spikes, labels[i], trial_idx=i)
