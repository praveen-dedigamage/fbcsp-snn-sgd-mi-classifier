import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import matplotlib.pyplot as plt

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
    
def plot_snn_raster_per_layer_combined(model, input_tensor, ground_truth, trial_index=0, max_neurons=50, class_labels=None, debug=False):
    """
    Plot spike rasters for input, hidden, and output layers of an SNN for a single trial.

    Args:
        model: Trained SNN model
        input_tensor: Tensor of shape [time, batch, features]
        trial_index: Trial index to visualize
        max_neurons: Max neurons to show per layer
        class_labels: List of class names for output layer y-axis
        debug: If True, prints tensor shapes and spike summaries
    """
    model.eval()
    with torch.no_grad():
        # Slice out a single trial from the batch
        x = input_tensor[:, trial_index : trial_index + 1, :]
        y = ground_truth[trial_index].item()
        if debug:
            print(f"[DEBUG] Input shape for model: {x.shape}")

        layers = list(model.children())
        mem = {}
        spike_recordings = {}

        # Input spikes (used directly from input_tensor)
        input_spikes = input_tensor[:, trial_index, :max_neurons]
        spike_recordings["Input Layer"] = input_spikes

        layer_index = 0
        for i, layer in enumerate(layers):
            if isinstance(layer, nn.Linear):
                x = layer(x)
                if debug:
                    print(f"[DEBUG] After Linear {i}, shape: {x.shape}")

            elif isinstance(layer, snn.Leaky):
                mem[layer_index] = layer.init_leaky()
                spk_list = []
                for t in range(x.size(0)):
                    spk, mem[layer_index] = layer(x[t], mem[layer_index])
                    spk_list.append(spk)

                spike_tensor = torch.stack(spk_list) if len(spk_list) > 0 else torch.zeros_like(x)

                # 🧠 Handle various shapes
                if spike_tensor.dim() == 3:
                    spk_data = spike_tensor[:, 0, :max_neurons]
                elif spike_tensor.dim() == 2:
                    spk_data = spike_tensor.expand(x.shape[0], -1)[:, :max_neurons]
                    if debug:
                        print(f"[DEBUG] WARNING: Expanded spike tensor from shape {spike_tensor.shape} to {spk_data.shape}")
                else:
                    raise ValueError(f"Unexpected spike_tensor shape: {spike_tensor.shape}")

                if debug:
                    print(f"[DEBUG] Layer {layer_index+1} spikes shape: {spk_data.shape}")
                    print(f"[DEBUG] Layer {layer_index+1} spike sum: {spk_data.sum(dim=0)}")

                spike_recordings[f"Layer {layer_index+1}"] = spk_data
                x = spike_tensor  # ✅ Use spike output instead of membrane potential
                layer_index += 1

        # 🔮 Predict class from final output spikes
        if f"Layer {layer_index}" in spike_recordings:
            output_spikes = spike_recordings[f"Layer {layer_index}"]
            summed = output_spikes.sum(dim=0)
            predicted_class = summed.argmax().item()
            print(f"🔮 Original class for trial {trial_index}: Class {y + 1}")
            print(f"🔮 Predicted class for trial {trial_index}: Class {predicted_class + 1}")
            if debug:
                print("🔎 Output spike sum per neuron:", summed)

        # 📊 Plotting
        num_layers = len(spike_recordings)
        fig, axes = plt.subplots(num_layers, 1, figsize=(12, 4 * num_layers), sharex=True)
        if num_layers == 1:
            axes = [axes]

        for i, (ax, (name, spk_tensor)) in enumerate(zip(axes, spike_recordings.items())):
            for neuron_idx in range(spk_tensor.shape[1]):
                spike_times = torch.nonzero(spk_tensor[:, neuron_idx]).squeeze()
                if spike_times.numel() > 0:
                    ax.scatter(spike_times.cpu(), torch.full_like(spike_times, neuron_idx), s=4)

            # Titles & Y-labels
            if name == "Input Layer":
                ax.set_title("Spike Raster - Input Layer")
                ax.set_ylabel("Input Channel")
            elif "Layer 1" in name:
                ax.set_title("Spike Raster - Hidden Layer")
                ax.set_ylabel("Neuron Index")
            elif "Layer 2" in name:
                ax.set_title("Spike Raster - Output Layer")
                ax.set_ylabel("Predicted Class")
                if class_labels is None:
                    class_labels = [f"Class {i+1}" for i in range(spk_tensor.shape[1])]
                ax.set_yticks(torch.arange(len(class_labels)))
                ax.set_yticklabels(class_labels[:spk_tensor.shape[1]])
            else:
                ax.set_title(f"Spike Raster - {name}")
                ax.set_ylabel("Neuron Index")

        axes[-1].set_xlabel("Time Step")
        plt.suptitle(f"Spike Raster Across Network Layers - Trial {trial_index}\n "
                     f"Original Class {y+1} | Predicted Class {predicted_class + 1})", fontsize=16)
        plt.subplots_adjust(top=0.8)  # Adjust this if the title gets cut off
        plt.tight_layout()
        plt.show()



base_directory = r'C:\Users\USER\Desktop\Extending the thesis\Dataset\BCICIV_2a_gdf'
modelName = input("Enter the model Name (.pt) : ") # Change dynamically
modelPath = fr'{base_directory}\{modelName}.pt'

checkpoint = torch.load(f"{modelPath}")

input_size = checkpoint["input_size"]
hidden_size = checkpoint["hidden_size"]
output_size = checkpoint["output_size"]

model = SNNClassifier(input_size, hidden_size, output_size)
model.load_state_dict(checkpoint["model_state_dict"])

X_train = checkpoint["X_train"]
X_test = checkpoint["X_test"]
y_train = checkpoint["y_train"]
y_test = checkpoint["y_test"]


for i in range(10):

    plot_snn_raster_per_layer_combined(
        model,
        X_test,
        y_test,
        trial_index=i,
        max_neurons=500,
        class_labels=["Class 1", "Class 2", "Class 3", "Class 4"],
        debug=True  # Set to False if you want cleaner output
    )
