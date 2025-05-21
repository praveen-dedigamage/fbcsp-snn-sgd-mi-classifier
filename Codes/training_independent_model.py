import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
import os
import sys
import time
import psutil

from snntorch import surrogate
from sklearn.metrics import accuracy_score, confusion_matrix

# -------------------------- Device --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mem = psutil.virtual_memory()
print(f" Using device: {device}")
   
class EarlyStopping:
    def __init__(self, patience=25, min_delta=1e-4, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.mode = mode

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        elif (self.mode == 'max' and current_score < self.best_score + self.min_delta) or \
             (self.mode == 'min' and current_score > self.best_score - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self.counter = 0

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
        x = x.float()
        batch_size = x.shape[1]
        num_steps = x.shape[0]
    
        # Remove batch_size argument — not supported in snnTorch 0.6+
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
    
        spk2_rec = torch.zeros((num_steps, batch_size, self.total_outputs), device=x.device)
    
        for step in range(num_steps):
            spk1, mem1 = self.lif1(self.fc1(x[step]), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            spk2_rec[step] = spk2
    
        return spk2_rec


# -------------------------- Target Spike Generator --------------------------
def create_sparse_temporal_population_spikes(y, num_classes, population_per_class, num_steps, batch_size, target_spike_probability =0.7):
    total_outputs = num_classes * population_per_class
    ideal_spikes = torch.zeros((num_steps, batch_size, total_outputs), device=device)
    for i in range(batch_size):
        class_idx = int(y[i].item())
        start = class_idx * population_per_class
        end = start + population_per_class
        for t in range(num_steps):
            for n in range(start, end):
                if torch.rand(1).item() < target_spike_probability:
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


def train_with_ideal_spikes_lazy_batchwise(model, X_train_np, y_train_np, X_test_np, y_test_np, X_val_np, y_val_np, LR=1e-3, epochs=10, target_spike_probability=0.7, batch_size=1024):

    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    loss_fn = lambda out, tgt: van_rossum_loss(out, tgt, tau=20.0)

    num_classes = len(np.unique(y_train_np.cpu()))
    population_per_class = model.population_per_class
    num_steps = X_train_np.shape[0]  # time steps

    best_test_acc = 0.0
    best_model_state = None

    train_losses, test_losses, val_losses = [], [], []
    train_accuracies, test_accuracies, val_accuracies = [], [], []
    train_incorrect_spike_ratios, val_incorrect_spike_ratios, test_incorrect_spike_ratios = [], [], []

    early_stopper = EarlyStopping(patience=50, min_delta=1e-4, mode='max')

    for epoch in range(epochs):
        etstart = time.time()
        model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        total_output_spikes = 0.0
        total_incorrect_spikes = 0.0

        indices = np.random.permutation(X_train_np.shape[1])
        for i in range(0, len(indices), batch_size):
            idx = indices[i:i+batch_size]
            x_batch = torch.tensor(X_train_np[:, idx, :], dtype=torch.float32, device=device)
            y_batch_np = y_train_np[idx]
            y_batch = y_batch_np.clone().detach().to(dtype=torch.long, device=device) if torch.is_tensor(y_batch_np) else torch.tensor(y_batch_np, dtype=torch.long, device=device)

            optimizer.zero_grad()
            output = model(x_batch)

            target_spikes = create_sparse_temporal_population_spikes(
                y_batch, num_classes, population_per_class, num_steps, len(idx), target_spike_probability)

            loss = loss_fn(output, target_spikes)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            output_sum = output.sum(dim=0)
            pred = torch.argmax(
                output_sum.view(len(idx), num_classes, population_per_class).sum(dim=2), dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

            for b in range(len(idx)):
                class_idx = int(y_batch[b].item())
                start = class_idx * population_per_class
                end = start + population_per_class

                output_sample = output[:, b, :]
                total_output_spikes += output_sample.sum().item()

                non_target = output_sample.clone()
                non_target[:, start:end] = 0
                total_incorrect_spikes += non_target.sum().item()

        epoch_acc = accuracy_score(all_labels, all_preds)
        train_losses.append(total_loss)
        train_accuracies.append(epoch_acc)

        train_incorrect_spike_ratio = total_incorrect_spikes / (total_output_spikes + 1e-6)
        train_incorrect_spike_ratios.append(train_incorrect_spike_ratio)

        # --- Validation ---
        model.eval()
        with torch.no_grad():
            x_val = torch.tensor(X_val_np, dtype=torch.float32, device=device)
            y_val = y_val_np.clone().detach().to(dtype=torch.long, device=device) if torch.is_tensor(y_val_np) else torch.tensor(y_val_np, dtype=torch.long, device=device)

            val_out = model(x_val)
            val_pred = torch.argmax(
                val_out.sum(dim=0).view(x_val.shape[1], num_classes, population_per_class).sum(dim=2), dim=1)
            val_acc = accuracy_score(y_val.cpu(), val_pred.cpu())
            val_loss = loss_fn(val_out, create_sparse_temporal_population_spikes(
                y_val, num_classes, population_per_class, num_steps, x_val.shape[1], target_spike_probability))

            val_total_output_spikes = 0.0
            val_total_incorrect_spikes = 0.0
            for i in range(x_val.shape[1]):
                class_idx = int(y_val[i].item())
                start = class_idx * population_per_class
                end = start + population_per_class
                val_sample = val_out[:, i, :]
                val_total_output_spikes += val_sample.sum().item()
                non_target = val_sample.clone()
                non_target[:, start:end] = 0
                val_total_incorrect_spikes += non_target.sum().item()

            val_incorrect_spike_ratio = val_total_incorrect_spikes / (val_total_output_spikes + 1e-6)
            val_incorrect_spike_ratios.append(val_incorrect_spike_ratio)

        val_losses.append(val_loss.item())
        val_accuracies.append(val_acc)

        # --- Testing (batch-wise) ---
        model.eval()
        all_test_preds = []
        all_test_labels = []
        test_total_loss = 0.0
        test_total_output_spikes = 0.0
        test_total_incorrect_spikes = 0.0

        for i in range(0, X_test_np.shape[1], batch_size):
            x_batch = torch.tensor(X_test_np[:, i:i+batch_size, :], dtype=torch.float32, device=device)
            y_batch_np = y_test_np[i:i+batch_size]
            y_batch = y_batch_np.clone().detach().to(dtype=torch.long, device=device) if torch.is_tensor(y_batch_np) else torch.tensor(y_batch_np, dtype=torch.long, device=device)

            with torch.no_grad():
                output = model(x_batch)
                target_spikes = create_sparse_temporal_population_spikes(
                    y_batch, num_classes, population_per_class, num_steps, x_batch.shape[1], target_spike_probability)
                test_total_loss += loss_fn(output, target_spikes).item()

                output_sum = output.sum(dim=0)
                pred = torch.argmax(
                    output_sum.view(x_batch.shape[1], num_classes, population_per_class).sum(dim=2), dim=1)
                all_test_preds.extend(pred.cpu().numpy())
                all_test_labels.extend(y_batch.cpu().numpy())

                for b in range(x_batch.shape[1]):
                    class_idx = int(y_batch[b].item())
                    start = class_idx * population_per_class
                    end = start + population_per_class
                    sample_out = output[:, b, :]
                    test_total_output_spikes += sample_out.sum().item()
                    non_target = sample_out.clone()
                    non_target[:, start:end] = 0
                    test_total_incorrect_spikes += non_target.sum().item()

        test_acc = accuracy_score(all_test_labels, all_test_preds)
        test_losses.append(test_total_loss)
        test_accuracies.append(test_acc)

        test_incorrect_spike_ratio = test_total_incorrect_spikes / (test_total_output_spikes + 1e-6)
        test_incorrect_spike_ratios.append(test_incorrect_spike_ratio)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_state = model.state_dict()

        early_stopper(test_acc)
        if early_stopper.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

        print(f"Epoch {epoch+1} | Train Acc: {epoch_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f} | Incorrect Spike Ratio: {train_incorrect_spike_ratio:.4f} / {val_incorrect_spike_ratio:.4f} / {test_incorrect_spike_ratio:.4f}")
        print("Epoch Time = ", time.time()-etstart, flush=True)
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, train_losses, train_accuracies, train_incorrect_spike_ratios, test_losses, test_accuracies, test_incorrect_spike_ratios, val_losses, val_accuracies, val_incorrect_spike_ratios

def evaluate(model, X_eval, y_eval, batch_size=64):

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i in range(0, X_eval.shape[1], batch_size):
            print("Evaluating", i)
            print("Total RAM:", mem.total / 1024**3, "GB")
            print("Available:", mem.available / 1024**3, "GB")
            if torch.cuda.is_available():
                print("GPU Memory Allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
                print("GPU Memory Reserved: ", torch.cuda.memory_reserved() / 1024**2, "MB")

            x_batch_np = X_eval[:, i:i+batch_size, :]
            y_batch_np = y_eval[i:i+batch_size]

            # Convert to tensors inside the loop
            x_batch = x_batch_np.clone().detach().to(dtype=torch.uint8, device=device) if torch.is_tensor(x_batch_np) else torch.tensor(x_batch_np, dtype=torch.uint8, device=device)
            y_batch = y_batch_np.clone().detach().to(dtype=torch.long, device=device) if torch.is_tensor(y_batch_np) else torch.tensor(y_batch_np, dtype=torch.long, device=device)

            output = model(x_batch)
            num_classes = len(torch.unique(y_batch))
            population_per_class = model.population_per_class
            output_sum = output.sum(dim=0)
            pred = torch.argmax(
                output_sum.view(x_batch.shape[1], num_classes, population_per_class).sum(dim=2), dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    return acc, cm


# -------------------------- Main Script --------------------------
if __name__ == "__main__":

    start = time.time()
    #ase_directory = r'/scratch/project_2003397/praveen'
    base_directory = r'C:/Users/USER/Desktop/fbcsp-snn-mi-classifier/fbcsp-snn-mi-classifier'
    #base_directory = r'/Users/hsprde/Documents/GitHub/fbcsp-snn-sgd-mi-classifier'
    relative_directory = r'Dataset'
    
    val_subject = int(sys.argv[1]) if len(sys.argv) > 1 else 1 
    hidden_number_of_Neuron = int(sys.argv[1]) if len(sys.argv) > 2 else 64 
    neuro_population_per_class = int(sys.argv[2]) if len(sys.argv) > 3 else 20
    spiking_prob = float(sys.argv[3]) if len(sys.argv) > 4 else 0.7
    
    data = np.load(f"spike_data/spike_trains_with_labels_val_subject_{val_subject}.npz")

    # Extract and convert to PyTorch tensors
    spike_train_train = data["train"]
    spike_train_test  = data["test"]
    spike_train_val   = data["val"]
    
    y_train = data["y_train"]
    y_test  = data["y_test"]
    y_val   = data["y_val"]    
    
    input_size = spike_train_train.shape[2]
    hidden_size = hidden_number_of_Neuron
    output_size = len(np.unique(y_train))
    population_per_class = neuro_population_per_class

    model = SNNClassifier(input_size, hidden_size, output_size, population_per_class).to(device)
    
    best_model, train_losses, train_accuracies, train_incorrect_spike_ratios, test_losses, test_accuracies, test_incorrect_spike_ratios, val_losses, val_accuracies, val_incorrect_spike_ratios = train_with_ideal_spikes_lazy_batchwise(
        model,
        spike_train_train,
        torch.tensor(y_train - 1).to(device),
        spike_train_test,
        torch.tensor(y_test - 1).to(device),
        spike_train_val,
        torch.tensor(y_val - 1).to(device),
        LR=1e-3,
        epochs=2,
        target_spike_probability = spiking_prob
    )

    train_acc, train_cm = evaluate(model, spike_train_train, torch.tensor(y_train - 1))
    test_acc, test_cm = evaluate(model, spike_train_test, torch.tensor(y_test - 1))
    val_acc, val_cm = evaluate(model, spike_train_val, torch.tensor(y_val - 1))

    results_dir = os.path.join(base_directory, "results")
    os.makedirs(results_dir, exist_ok=True)
    save_path = f'model_and_history_LeaveOutSubID{val_subject}.pth'
    path_to_model_history = os.path.join(results_dir, save_path)

    
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'train_incorrect_spike_ratios': train_incorrect_spike_ratios,
        'test_losses': val_losses,
        'test_accuracies': val_accuracies,
        'test_incorrect_spike_ratios': val_incorrect_spike_ratios,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'val_incorrect_spike_ratios': val_incorrect_spike_ratios,
        'train_cm': train_cm,
        'test_cm': test_cm,
        'train_acc': train_acc,
        'test_acc': test_acc,
    }, path_to_model_history)
    
    print("Time for one Fold", time.time() - start)
    
