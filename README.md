# FBCSP-SNN Motor Imagery EEG Classifier

A GPU-accelerated pipeline that combines **Filter-Bank Common Spatial Patterns (FBCSP)** feature extraction with a **Spiking Neural Network (SNN)** classifier for EEG-based Motor Imagery (MI) decoding.

## Architecture

```
Raw EEG  →  Bandpass filter (3 bands)  →  Pairwise CSP projection
         →  Adaptive-threshold spike encoding
         →  2-layer LIF-SNN (snntorch)  →  Population-coded WTA decoding
```

Key design choices:
- **PairwiseCSP**: one set of spatial filters per class pair (generalised eigenvalue problem)
- **Van Rossum loss**: MSE between exponentially-filtered spike trains, computed via FFT convolution
- **AMP training**: `torch.autocast` + `GradScaler` for FP16 on CUDA, with simulated INT8 evaluation
- **`torch.compile`**: fuses CUDA kernels on Linux (Puhti); disabled automatically on Windows

## Supported datasets (via MOABB)

| Name | Classes | Subjects | Channels | Hz |
|---|---|---|---|---|
| `BNCI2014_001` | 4 (left/right hand, feet, tongue) | 9 | 22 | 250 |
| `PhysionetMI` | 4 | 109 | 64 | 160→250 |
| `Cho2017` | 2 (left/right hand) | 52 | 64 | 512→250 |
| `BNCI2015_001` | 2 (right hand, feet) | 12 | 13 | 512→250 |

Data is downloaded automatically on first use to `~/mne_data/`.

## Installation

```bash
# 1. Install PyTorch for your platform (see https://pytorch.org/get-started/locally/)
#    Windows + CUDA 12.x:
pip install torch --index-url https://download.pytorch.org/whl/cu124
#    Puhti HPC:
#    module load pytorch/2.4  (before activating venv)

# 2. Install remaining dependencies
pip install -r requirements.txt
```

## Usage

### Train (10-fold cross-validation)

```bash
# BNCI2014_001 subject 1 — n_classes auto-detected from registry (4)
python main.py train --source moabb --moabb-dataset BNCI2014_001 --subject-id 1

# 2-class dataset — n_classes auto-detected (2); no --n-classes needed
python main.py train --source moabb --moabb-dataset Cho2017 --subject-id 1

# File-based source (legacy .mat files)
python main.py train --source file --data-dir /path/to/dataset --subject-id 1

# Custom hyperparameters
python main.py train --source moabb --moabb-dataset BNCI2014_001 --subject-id 1 \
    --epochs 500 --lr 5e-4 --n-folds 5 --hidden-neurons 128
```

### Inference

```bash
python main.py infer --source moabb --moabb-dataset BNCI2014_001 --subject-id 1 --fold 1
```

### All options

```bash
python main.py train --help
python main.py infer --help
```

## Running on Puhti (CSC HPC)

```bash
# Interactive GPU session via Open OnDemand → Jupyter → Terminal
module load pytorch/2.4
source .venv/bin/activate
python main.py train --source moabb --moabb-dataset BNCI2014_001 --subject-id 1

# Batch job (all 9 subjects in parallel)
sbatch run_puhti.sh
```

See `run_puhti.sh` for the SLURM array-job template.

## Results

Artifacts are written to `Results/Subject_<N>/`:

| File | Contents |
|---|---|
| `snn_model_subject{N}_fold{F}.pth` | Model weights |
| `csp_subject{N}_fold{F}.pkl` | Fitted CSP filters |
| `pipeline_params_subject{N}_fold{F}.json` | Full pipeline config + feature indices |
| `split_info_subject{N}.json` | Train/eval split description (MOABB source) |
| `*.png` | Diagnostic plots (spike raster, confusion matrix, weight histograms, …) |
