import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# ---- Synthetic EEG Generator ----
def generate_synthetic_mi_eeg(batch_size=16, channels=22, time_steps=250, num_classes=4):
    assert batch_size % num_classes == 0, "Batch size must be divisible by number of classes"
    samples_per_class = batch_size // num_classes
    y = torch.tensor([c for c in range(num_classes) for _ in range(samples_per_class)])
    X = torch.randn(batch_size, channels, time_steps) * 0.1
    for i in range(batch_size):
        class_id = y[i].item()
        start = (class_id * 5) % channels
        X[i, start:start+5, 100:120] += 1.0
    return X, y

# ---- Preprocessor Target Generator ----
def create_preprocessor_targets(y, time_steps, channels):
    torch.manual_seed(42)
    unique_patterns = []
    for _ in range(4):
        pattern = torch.zeros(time_steps, channels)
        active_channels = torch.randperm(channels)[:8]
        for ch in active_channels:
            spike_times = torch.randint(0, time_steps, (time_steps // 5,))
            pattern[spike_times, ch] = 1.0
        unique_patterns.append(pattern)
    target = torch.stack([unique_patterns[label.item()] for label in y], dim=1)  # [time, batch, channels]
    return target

# ---- Ideal Classifier Targets ----
def create_ideal_spikes(y, num_classes, num_steps, batch_size, spike_value=1.0):
    ideal_spikes = torch.zeros((num_steps, batch_size, num_classes))
    for i in range(batch_size):
        ideal_spikes[:, i, y[i]] = spike_value
    return ideal_spikes

# ---- Custom Spike Loss ----
def custom_spike_loss(pred, target, alpha=0.01):
    # Spike match error (only where target has spikes)
    mask_positive = target > 0
    mask_negative = target == 0

    # Penalize missing spikes (false negatives)
    loss_missing = ((target - pred)[mask_positive] ** 2).mean() if mask_positive.any() else torch.tensor(0.0, device=pred.device)

    # Penalize false spikes (false positives)
    loss_extra = ((pred)[mask_negative] ** 2).mean() if mask_negative.any() else torch.tensor(0.0, device=pred.device)

    # Spike count alignment
    pred_spikes = pred.sum(dim=0)  # [batch, channels]
    target_spikes = target.sum(dim=0)
    loss_count = ((pred_spikes - target_spikes) ** 2).mean()

    total_loss = loss_missing + loss_extra + alpha * loss_count
    return total_loss

# ---- SNN Preprocessing Module ----
class SNNPreprocessor(nn.Module):
    def __init__(self, input_channels=22, conv_filters=32, time_steps=250):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=input_channels, out_channels=conv_filters, kernel_size=5, padding=2)
        self.lif = snn.Leaky(beta=0.95, spike_grad=surrogate.fast_sigmoid(), learn_beta=True)
        self.time_steps = time_steps

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv(x)
        x = x.permute(2, 0, 1)
        mem = torch.zeros((batch_size, x.size(2)), device=x.device)
        spk_out = []
        for t in range(self.time_steps):
            spk, mem = self.lif(x[t], mem)
            spk_out.append(spk)
        return torch.stack(spk_out, dim=0)

# ---- SNN Classifier Module ----
class SNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=0.95, spike_grad=surrogate.fast_sigmoid())
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.lif2 = snn.Leaky(beta=0.95, spike_grad=surrogate.fast_sigmoid())

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

# ---- Visualization ----
def plot_preprocessor_spikes(encoded_spikes, target_spikes, trial_idx=0, channels=32):
    fig, ax = plt.subplots(figsize=(12, 6))
    encoded = encoded_spikes[:, trial_idx, :channels]
    target = target_spikes[:, trial_idx, :channels]
    e_t, e_c = encoded.nonzero(as_tuple=True)
    t_t, t_c = target.nonzero(as_tuple=True)
    ax.scatter(e_t.cpu(), e_c.cpu(), s=2, color='black', label='Predicted')
    ax.scatter(t_t.cpu(), t_c.cpu(), s=2, color='red', alpha=0.5, label='Reference')
    ax.set_title(f"Preprocessor Output vs Target Spikes (Trial {trial_idx})")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Channel")
    ax.legend()
    plt.tight_layout()
    plt.show()

# ---- Main Execution ----
if __name__ == "__main__":
    batch_size = 16
    time_steps = 250
    input_channels = 22
    conv_filters = 32

    raw_eeg, labels = generate_synthetic_mi_eeg(batch_size, input_channels, time_steps)
    preprocessor = SNNPreprocessor(input_channels, conv_filters, time_steps)
    pre_target = create_preprocessor_targets(labels, time_steps, conv_filters)
    optimizer_pre = torch.optim.Adam(preprocessor.parameters(), lr=1e-3)

    for epoch in range(100):
        optimizer_pre.zero_grad()
        spike_encoded = preprocessor(raw_eeg)
        loss_pre = custom_spike_loss(spike_encoded, pre_target)
        loss_pre.backward()
        optimizer_pre.step()
        print(f"[Preprocessor] Epoch {epoch+1}, Loss: {loss_pre.item():.4f}")

    preprocessor.eval()
    with torch.no_grad():
        spike_encoded = preprocessor(raw_eeg)
        
    for i in range(16):
        plot_preprocessor_spikes(spike_encoded, pre_target, trial_idx=i)

    classifier = SNNClassifier(input_size=conv_filters, hidden_size=128, output_size=4)
    ideal_spikes = create_ideal_spikes(labels, 4, time_steps, batch_size).to(spike_encoded.device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(20):
        optimizer.zero_grad()
        out_spikes = classifier(spike_encoded)
        loss = loss_fn(out_spikes, ideal_spikes)
        loss.backward(retain_graph=True)
        optimizer.step()
        print(f"[Classifier] Epoch {epoch+1}, Loss: {loss.item():.4f}")

    classifier.eval()
    with torch.no_grad():
        out_spikes = classifier(spike_encoded)
        summed_output = out_spikes.sum(dim=0)
        predicted = torch.argmax(summed_output, dim=1)
        acc = accuracy_score(labels.cpu(), predicted.cpu())
        cm = confusion_matrix(labels.cpu(), predicted.cpu())

    print("Labels:", labels)
    print("Predicted:", predicted)
    print(f"Accuracy: {acc * 100:.2f}%")
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[f"C{i}" for i in range(4)], yticklabels=[f"C{i}" for i in range(4)])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
