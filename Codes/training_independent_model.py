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
torch.set_num_threads(4)  # Match this to --cpus-per-task in your SLURM script
print("Torch will use", torch.get_num_threads(), "CPU threads")


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


def evaluate_set(model, data_tensor, label_tensor, loss_fn, num_classes, population_per_class, num_steps, target_spike_probability, batch_size=4096, name="val"):
    model.eval()
    total_loss, total_output_spikes, total_incorrect_spikes = 0.0, 0.0, 0.0
    all_preds, all_labels = [], []
    start_time = time.time()

    for i in range(0, data_tensor.shape[1], batch_size):
        x_batch = data_tensor[:, i:i+batch_size, :]
        y_batch = label_tensor[i:i+batch_size]

        with torch.no_grad():
            output = model(x_batch)
            target_spikes = create_sparse_temporal_population_spikes(
                y_batch, num_classes, population_per_class, num_steps, x_batch.shape[1], target_spike_probability)
            total_loss += loss_fn(output, target_spikes).item()

            output_sum = output.sum(dim=0)
            pred = torch.argmax(output_sum.view(x_batch.shape[1], num_classes, population_per_class).sum(dim=2), dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

            for b in range(x_batch.shape[1]):
                class_idx = int(y_batch[b].item())
                start = class_idx * population_per_class
                end = start + population_per_class
                sample_out = output[:, b, :]
                total_output_spikes += sample_out.sum().item()
                non_target = sample_out.clone()
                non_target[:, start:end] = 0
                total_incorrect_spikes += non_target.sum().item()

    acc = accuracy_score(all_labels, all_preds)
    incorrect_ratio = total_incorrect_spikes / (total_output_spikes + 1e-6)
    print(f"{name.title()} duration: {time.time() - start_time:.2f} sec", flush=True)
    return total_loss, acc, incorrect_ratio

def train_with_ideal_spikes_lazy_batchwise(model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, X_val_tensor, y_val_tensor, LR=1e-3, epochs=10, target_spike_probability=0.7, batch_size=1024):
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    loss_fn = lambda out, tgt: van_rossum_loss(out, tgt, tau=20.0)

    num_classes = len(torch.unique(y_train_tensor))
    population_per_class = model.population_per_class
    num_steps = X_train_tensor.shape[0]  # time steps

    best_test_acc = 0.0
    best_model_state = None

    train_losses, test_losses, val_losses = [], [], []
    train_accuracies, test_accuracies, val_accuracies = [], [], []
    train_incorrect_spike_ratios, val_incorrect_spike_ratios, test_incorrect_spike_ratios = [], [], []

    early_stopper = EarlyStopping(patience=50, min_delta=1e-4, mode='max')

    for epoch in range(epochs):
        etstart = time.time()
        print(epoch)
        model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        total_output_spikes = 0.0
        total_incorrect_spikes = 0.0

        indices = torch.randperm(X_train_tensor.shape[1])
        for i in range(0, len(indices), batch_size):
            tstart = time.time()
            print("training", i, flush=True)

            idx = indices[i:i+batch_size]
            x_batch = X_train_tensor[:, idx, :]
            y_batch = y_train_tensor[idx]

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

            print("Time for one batch", time.time() - tstart, flush=True)

        epoch_acc = accuracy_score(all_labels, all_preds)
        train_losses.append(total_loss)
        train_accuracies.append(epoch_acc)

        train_incorrect_spike_ratio = total_incorrect_spikes / (total_output_spikes + 1e-6)
        train_incorrect_spike_ratios.append(train_incorrect_spike_ratio)

        val_loss, val_acc, val_ratio = evaluate_set(model, X_val_tensor, y_val_tensor, loss_fn, num_classes, population_per_class, num_steps, target_spike_probability, batch_size, name="val")
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_incorrect_spike_ratios.append(val_ratio)

        test_loss, test_acc, test_ratio = evaluate_set(model, X_test_tensor, y_test_tensor, loss_fn, num_classes, population_per_class, num_steps, target_spike_probability, batch_size, name="test")
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        test_incorrect_spike_ratios.append(test_ratio)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_state = model.state_dict()

        early_stopper(test_acc)
        if early_stopper.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

        print(f"Epoch {epoch+1} | Train Acc: {epoch_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f} | Incorrect Spike Ratio: {train_incorrect_spike_ratio:.4f} / {val_ratio:.4f} / {test_ratio:.4f}", flush=True)
        epoch_duration = time.time() - etstart
        print(f"Time for one epoch: {epoch_duration:.2f} seconds", flush=True)

    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, train_losses, train_accuracies, train_incorrect_spike_ratios, test_losses, test_accuracies, test_incorrect_spike_ratios, val_losses, val_accuracies, val_incorrect_spike_ratios

def evaluate(model, X_eval_tensor, y_eval_tensor, batch_size=64):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i in range(0, X_eval_tensor.shape[1], batch_size):
            x_batch = X_eval_tensor[:, i:i+batch_size, :].to(device)
            y_batch = y_eval_tensor[i:i+batch_size].to(device)

            output = model(x_batch)  # Shape: (T, batch, neurons)
            num_classes = len(torch.unique(y_batch))
            population_per_class = model.population_per_class

            output_sum = output.sum(dim=0)  # (batch, output_neurons)
            pred = torch.argmax(
                output_sum.view(x_batch.shape[1], num_classes, population_per_class).sum(dim=2),
                dim=1
            )

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    return acc, cm


# -------------------------- Main Script --------------------------
if __name__ == "__main__":

    start = time.time()
    #ase_directory = r'/scratch/project_2003397/praveen'
    #base_directory = r'C:/Users/USER/Desktop/fbcsp-snn-mi-classifier/fbcsp-snn-mi-classifier'
    #base_directory = r'/Users/hsprde/Documents/GitHub/fbcsp-snn-sgd-mi-classifier'
    relative_directory = r'Dataset'
    
    val_subject = int(sys.argv[1]) if len(sys.argv) > 1 else 1 
    hidden_number_of_Neuron = int(sys.argv[1]) if len(sys.argv) > 2 else 64 
    neuro_population_per_class = int(sys.argv[2]) if len(sys.argv) > 3 else 20
    spiking_prob = float(sys.argv[3]) if len(sys.argv) > 4 else 0.7
    
    data = np.load(f"spike_data/spike_trains_with_labels_val_subject_{val_subject}.npz")

    # Extract and convert to PyTorch tensors
    spike_train_train = torch.tensor(data["train"], dtype=torch.float32).to(device)
    spike_train_test  = torch.tensor(data["test"],  dtype=torch.float32).to(device)
    spike_train_val   = torch.tensor(data["val"],   dtype=torch.float32).to(device)
    
    y_train = torch.tensor(data["y_train"] - 1, dtype=torch.long).to(device)
    y_test  = torch.tensor(data["y_test"]  - 1, dtype=torch.long).to(device)
    y_val   = torch.tensor(data["y_val"]   - 1, dtype=torch.long).to(device)
 
    
    input_size = spike_train_train.shape[2]
    hidden_size = hidden_number_of_Neuron
    output_size = len(np.unique(y_train))
    population_per_class = neuro_population_per_class

    model = SNNClassifier(input_size, hidden_size, output_size, population_per_class).to(device)
    
    best_model, train_losses, train_accuracies, train_incorrect_spike_ratios, test_losses, test_accuracies, test_incorrect_spike_ratios, val_losses, val_accuracies, val_incorrect_spike_ratios = train_with_ideal_spikes_lazy_batchwise(
    model,
    spike_train_train,
    y_train,
    spike_train_test,
    y_test,
    spike_train_val,
    y_val,
    LR=1e-3,
    epochs=2,
    target_spike_probability=spiking_prob)
    

    train_acc, train_cm = evaluate(best_model, spike_train_train, y_train)
    test_acc, test_cm   = evaluate(best_model, spike_train_test, y_test)
    val_acc, val_cm     = evaluate(best_model, spike_train_val, y_val)


    #results_dir = os.path.join(base_directory, "results")
    #os.makedirs(results_dir, exist_ok=True)
    save_path = f'results/model_and_history_LeaveOutSubID{val_subject}.pth'
    #path_to_model_history = os.path.join(results_dir, save_path)

    
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
    }, save_path)
    
    print("Time for one Fold", time.time() - start)
