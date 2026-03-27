"""High-level training and inference pipelines."""

import json
import pickle
from pathlib import Path
from typing import List

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from fbcsp_snn import DEVICE, setup_logger
from fbcsp_snn.config import Config
from fbcsp_snn.data import load_data
from fbcsp_snn.datasets import load_moabb_subject
from fbcsp_snn.encoding import encode_to_spikes
from fbcsp_snn.evaluation import calculate_feature_importance, evaluate
from fbcsp_snn.model import SNNClassifier
from fbcsp_snn.preprocessing import PairwiseCSP, bandpass_filter
from fbcsp_snn.quantization import quantize_csp, quantize_model
from fbcsp_snn.training import train
from fbcsp_snn.visualization import (
    plot_confusion_matrix,
    plot_discriminative_importance,
    plot_neuron_traces,
    plot_spike_count_difference,
    plot_spike_probability_heatmap,
    plot_spike_propagation,
    plot_weight_histograms,
)

logger = setup_logger(__name__)

# Session identifiers used by load_data
_SESSION_TRAIN = "T"
_SESSION_EVAL = "E"


def _triton_available() -> bool:
    """Return True only if Triton is importable (Linux; not available on Windows)."""
    try:
        import triton  # noqa: F401
        return True
    except ImportError:
        return False


# ── Path helpers ──────────────────────────────────────────────────────────────


def _resolve_data_dir(cfg: Config, base_dir: Path) -> Path:
    return cfg.data_dir or base_dir / "Dataset" / "New_Dataset_2"


def _resolve_results_dir(cfg: Config, base_dir: Path) -> Path:
    results = cfg.results_dir or base_dir / "Results" / f"Subject_{cfg.subject_id}"
    results.mkdir(parents=True, exist_ok=True)
    return results


# ── Serialisation helpers ─────────────────────────────────────────────────────


def _save_fold_artifacts(
    results_dir: Path,
    subject_id: int,
    fold: int,
    model: SNNClassifier,
    csp: PairwiseCSP,
    cfg: Config,
    top_indices: torch.Tensor,
    metrics: dict,
) -> None:
    """Persist model weights, CSP object, pipeline metadata, and fold metrics."""
    torch.save(
        model.state_dict(),
        results_dir / f"snn_model_subject{subject_id}_fold{fold}.pth",
    )
    with open(results_dir / f"csp_subject{subject_id}_fold{fold}.pkl", "wb") as fh:
        pickle.dump(csp, fh)

    payload = {
        "subject_id": subject_id,
        "fold": fold,
        "model_params": {
            "input_size": model.fc1.in_features,
            "hidden_size": model.fc1.out_features,
            "output_size": cfg.n_classes,
            "population_per_class": model.population_per_class,
            "beta": cfg.beta,
            "dropout_prob": cfg.dropout_prob,
        },
        "feature_selection": {
            "percentile_kept": cfg.feature_percentile,
            "top_indices": top_indices.cpu().tolist(),
        },
        "encoding_params": {
            "base_thresh": cfg.base_thresh,
            "adapt_inc": cfg.adapt_inc,
            "decay": cfg.decay,
        },
        "freq_bands": cfg.freq_bands,
        "csp_params": {
            "n_components": csp.n_components,
            "reg_lambda": csp.reg_lambda,
            "selected_classes": csp.selected_classes,
        },
        "metrics": metrics,
    }
    with open(
        results_dir / f"pipeline_params_subject{subject_id}_fold{fold}.json", "w"
    ) as fh:
        json.dump(payload, fh, indent=2)


def _load_fold_artifacts(results_dir: Path, subject_id: int, fold: int, device: torch.device):
    """Load and return (params_dict, csp, model)."""
    params_path = results_dir / f"pipeline_params_subject{subject_id}_fold{fold}.json"
    csp_path = results_dir / f"csp_subject{subject_id}_fold{fold}.pkl"
    model_path = results_dir / f"snn_model_subject{subject_id}_fold{fold}.pth"

    for p in (params_path, csp_path, model_path):
        if not p.exists():
            raise FileNotFoundError(f"Artifact not found: {p}")

    with open(params_path) as fh:
        params = json.load(fh)

    with open(csp_path, "rb") as fh:
        csp: PairwiseCSP = pickle.load(fh)

    mp = params["model_params"]
    model = SNNClassifier(
        input_size=mp["input_size"],
        hidden_size=mp["hidden_size"],
        output_size=mp["output_size"],
        population_per_class=mp["population_per_class"],
        beta=mp["beta"],
        dropout_prob=mp["dropout_prob"],
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return params, csp, model


# ── Preprocessing helpers ─────────────────────────────────────────────────────


def _multiband_filter(X: np.ndarray, freq_bands: list, fs: float = 250.0) -> np.ndarray:
    """Concatenate bandpass-filtered copies of X along the channel axis."""
    return np.concatenate(
        [bandpass_filter(X, low, high, fs=fs) for low, high in freq_bands],
        axis=1,
    )


def _encode_and_prune(
    projected: dict,
    cfg: Config,
    device: torch.device,
    top_indices: torch.Tensor,
) -> torch.Tensor:
    spikes = encode_to_spikes(
        projected,
        base_thresh=cfg.base_thresh,
        adapt_inc=cfg.adapt_inc,
        decay=cfg.decay,
        device=device,
    )
    return spikes[:, :, top_indices]


# ── Training pipeline ─────────────────────────────────────────────────────────


def run_train(
    cfg: Config,
    base_dir: Path,
    device: torch.device = DEVICE,
    fold_id: int = None,
) -> None:
    """Run stratified k-fold cross-validation and persist fold artifacts.

    For each fold the pipeline:

    1. Fits a :class:`~fbcsp_snn.preprocessing.PairwiseCSP` on training data.
    2. Encodes projected signals to spikes.
    3. Selects the top ``feature_percentile`` % of features by discriminative
       importance computed on the training spikes.
    4. Trains an :class:`~fbcsp_snn.model.SNNClassifier`.
    5. Evaluates both FP32 and simulated INT8 pipelines on the held-out test set.
    6. Saves all artifacts and diagnostic plots.
    """
    data_dir = _resolve_data_dir(cfg, base_dir)
    results_dir = _resolve_results_dir(cfg, base_dir)

    logger.info("Subject %d — results → %s", cfg.subject_id, results_dir)

    # ── Load data ─────────────────────────────────────────────────────────────
    if cfg.source == "moabb":
        logger.info(
            "Data source: MOABB / %s  (tmin=%.1f, tmax=%.1f)",
            cfg.moabb_dataset, cfg.tmin, cfg.tmax,
        )
        X_train_all, y_train_all, X_test_all, y_test_raw, split_info = load_moabb_subject(
            cfg.moabb_dataset,
            cfg.subject_id,
            tmin=cfg.tmin,
            tmax=cfg.tmax,
            n_classes=cfg.n_classes,
        )
        with open(results_dir / f"split_info_subject{cfg.subject_id}.json", "w") as fh:
            json.dump(split_info, fh, indent=2)
    else:
        logger.info("Data source: file  (%s)", data_dir)
        X_train_raw, y_train_raw = load_data(data_dir, cfg.subject_id, _SESSION_TRAIN)
        X_test_raw, y_test_raw = load_data(data_dir, cfg.subject_id, _SESSION_EVAL)
        # Keep only labelled MI trials (label > 0 = rest class)
        X_train_all = X_train_raw[y_train_raw > 0]
        y_train_all = y_train_raw[y_train_raw > 0]
        X_test_all = X_test_raw[y_test_raw > 0]
        y_test_raw = y_test_raw[y_test_raw > 0]

    # Bandpass filtering (done once, outside the fold loop)
    X_train_filtered = _multiband_filter(X_train_all, cfg.freq_bands)
    X_test_filtered = _multiband_filter(X_test_all, cfg.freq_bands)
    y_test_tensor = torch.tensor(y_test_raw - 1, dtype=torch.long).to(device, non_blocking=True)

    # ── Cross-validation ──────────────────────────────────────────────────────
    kfold = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=42)

    all_folds = list(enumerate(kfold.split(X_train_filtered, y_train_all), start=1))
    if fold_id is not None:
        if fold_id < 1 or fold_id > cfg.n_folds:
            raise ValueError(f"--fold-id must be between 1 and {cfg.n_folds}, got {fold_id}")
        folds_to_run = [all_folds[fold_id - 1]]
        logger.info("Running single fold %d/%d (parallel mode)", fold_id, cfg.n_folds)
    else:
        folds_to_run = all_folds

    train_accs: List[float] = []
    val_accs: List[float] = []
    test_accs_fp32: List[float] = []
    test_accs_int8: List[float] = []
    total_cm_fp32 = np.zeros((cfg.n_classes, cfg.n_classes), dtype=int)
    total_cm_int8 = np.zeros((cfg.n_classes, cfg.n_classes), dtype=int)

    n_bands = len(cfg.freq_bands)
    n_channels = X_train_all.shape[1]
    # CSP spatial filters are limited by the number of EEG channels;
    # cap csp_components_per_band so datasets with fewer channels don't crash.
    csp_comps_per_band = min(cfg.csp_components_per_band, n_channels)
    if csp_comps_per_band != cfg.csp_components_per_band:
        logger.warning(
            "csp_components_per_band capped from %d to n_channels=%d",
            cfg.csp_components_per_band, n_channels,
        )
    csp_n_components = n_bands * csp_comps_per_band

    for fold_idx, (tr_idx, val_idx) in folds_to_run:
        logger.info("Fold %d/%d", fold_idx, cfg.n_folds)

        X_tr, y_tr = X_train_filtered[tr_idx], y_train_all[tr_idx]
        X_val, y_val = X_train_filtered[val_idx], y_train_all[val_idx]

        # ── CSP ───────────────────────────────────────────────────────────────
        csp = PairwiseCSP(
            n_components=csp_n_components,
            selected_classes=list(range(1, cfg.n_classes + 1)),
            reg_lambda=cfg.lambda_r,
        )
        csp.fit(X_tr, y_tr)

        proj_tr = csp.transform(X_tr)
        proj_val = csp.transform(X_val)
        proj_test = csp.transform(X_test_filtered)

        y_tr_t = torch.tensor(y_tr - 1, dtype=torch.long, device=device)
        y_val_t = torch.tensor(y_val - 1, dtype=torch.long, device=device)

        # ── Spike encoding ────────────────────────────────────────────────────
        spk_tr_full = encode_to_spikes(proj_tr, cfg.base_thresh, cfg.adapt_inc, cfg.decay, device)
        spk_val_full = encode_to_spikes(proj_val, cfg.base_thresh, cfg.adapt_inc, cfg.decay, device)
        spk_test_full = encode_to_spikes(proj_test, cfg.base_thresh, cfg.adapt_inc, cfg.decay, device)

        # ── Feature selection ─────────────────────────────────────────────────
        total_features = spk_tr_full.shape[2]
        n_keep = max(1, int(total_features * cfg.feature_percentile / 100.0))

        if n_keep < total_features:
            importance = calculate_feature_importance(spk_tr_full, y_tr_t)
            top_indices, _ = torch.sort(
                torch.argsort(importance, descending=True)[:n_keep]
            )
            logger.info(
                "Feature selection: keeping %d/%d features (%.0f%%)",
                n_keep, total_features, cfg.feature_percentile,
            )
        else:
            top_indices = torch.arange(total_features, device=device)

        spk_tr = spk_tr_full[:, :, top_indices]
        spk_val = spk_val_full[:, :, top_indices]
        spk_test = spk_test_full[:, :, top_indices]

        # ── Model ─────────────────────────────────────────────────────────────
        model = SNNClassifier(
            input_size=spk_tr.shape[2],
            hidden_size=cfg.hidden_neurons,
            output_size=cfg.n_classes,
            population_per_class=cfg.population_per_class,
            beta=cfg.beta,
            dropout_prob=cfg.dropout_prob,
        ).to(device)

        # torch.compile fuses ops into optimised CUDA kernels (Linux only;
        # requires Triton which is not available on Windows).
        if device.type == "cuda" and _triton_available():
            model = torch.compile(model, mode="reduce-overhead")

        best_model, _ = train(
            model, spk_tr, y_tr_t, spk_val, y_val_t,
            lr=cfg.lr,
            epochs=cfg.epochs,
            weight_decay=cfg.weight_decay,
            spike_prob=cfg.spiking_prob,
            early_stopping_patience=cfg.early_stopping_patience,
            early_stopping_warmup=cfg.early_stopping_warmup,
        )

        # ── Evaluation (FP32) ─────────────────────────────────────────────────
        tr_acc, _ = evaluate(best_model, spk_tr, y_tr_t, device)
        val_acc, _ = evaluate(best_model, spk_val, y_val_t, device)
        test_acc_fp32, cm_fp32 = evaluate(best_model, spk_test, y_test_tensor, device)
        total_cm_fp32 += cm_fp32

        # ── Evaluation (simulated INT8) ───────────────────────────────────────
        q_model = quantize_model(best_model)
        q_csp = quantize_csp(csp)
        spk_test_int8 = encode_to_spikes(
            q_csp.transform(X_test_filtered), cfg.base_thresh, cfg.adapt_inc, cfg.decay, device
        )[:, :, top_indices]
        test_acc_int8, cm_int8 = evaluate(q_model, spk_test_int8, y_test_tensor, device)
        total_cm_int8 += cm_int8

        # ── Diagnostic plots ──────────────────────────────────────────────────
        plot_weight_histograms(best_model, q_model, cfg.subject_id, fold_idx, results_dir)

        sample = spk_val[:, 0:1, :]
        true_lbl = int(y_val_t[0].item())
        best_model.eval()
        with torch.no_grad():
            spk_out, spk_hid, mem_out, mem_hid = best_model(sample)
            class_scores = (
                spk_out.sum(dim=0)
                .view(1, cfg.n_classes, cfg.population_per_class)
                .sum(dim=2)
            )
            pred_lbl = int(class_scores.argmax(dim=1).item())

        plot_spike_propagation(
            sample, spk_hid, spk_out, true_lbl, pred_lbl,
            cfg.subject_id, fold_idx, cfg.population_per_class, results_dir,
        )
        plot_neuron_traces(mem_hid, mem_out, spk_hid, spk_out, cfg.subject_id, fold_idx, results_dir)

        # ── Persist artifacts ─────────────────────────────────────────────────
        _save_fold_artifacts(
            results_dir, cfg.subject_id, fold_idx, best_model, csp, cfg, top_indices,
            metrics={
                "train_acc": tr_acc,
                "val_acc": val_acc,
                "test_acc_fp32": test_acc_fp32,
                "test_acc_int8": test_acc_int8,
                "cm_fp32": cm_fp32.tolist(),
                "cm_int8": cm_int8.tolist(),
            },
        )

        train_accs.append(tr_acc)
        val_accs.append(val_acc)
        test_accs_fp32.append(test_acc_fp32)
        test_accs_int8.append(test_acc_int8)

        logger.info(
            "Fold %d — train=%.4f  val=%.4f  test_fp32=%.4f  test_int8=%.4f",
            fold_idx, tr_acc, val_acc, test_acc_fp32, test_acc_int8,
        )

    # ── Aggregate results (only when all folds ran together) ──────────────────
    if fold_id is None:
        run_aggregate(cfg, base_dir)


# ── Aggregate pipeline ────────────────────────────────────────────────────────


def run_aggregate(cfg: Config, base_dir: Path) -> None:
    """Read per-fold artifacts and produce summary CSV + confusion-matrix plots.

    Works whether folds were run sequentially (``run_train`` with no
    ``fold_id``) or as independent parallel jobs (``--fold-id N``).
    """
    results_dir = _resolve_results_dir(cfg, base_dir)

    test_accs_fp32: List[float] = []
    test_accs_int8: List[float] = []
    total_cm_fp32 = np.zeros((cfg.n_classes, cfg.n_classes), dtype=int)
    total_cm_int8 = np.zeros((cfg.n_classes, cfg.n_classes), dtype=int)
    missing: List[int] = []

    for fold in range(1, cfg.n_folds + 1):
        params_path = results_dir / f"pipeline_params_subject{cfg.subject_id}_fold{fold}.json"
        if not params_path.exists():
            missing.append(fold)
            continue
        with open(params_path) as fh:
            params = json.load(fh)
        m = params.get("metrics", {})
        if not m:
            logger.warning("Fold %d artifact has no metrics (re-run that fold)", fold)
            missing.append(fold)
            continue
        test_accs_fp32.append(m["test_acc_fp32"])
        test_accs_int8.append(m["test_acc_int8"])
        total_cm_fp32 += np.array(m["cm_fp32"], dtype=int)
        total_cm_int8 += np.array(m["cm_int8"], dtype=int)

    if missing:
        logger.warning("Missing / incomplete folds: %s — aggregate is partial", missing)
    if not test_accs_fp32:
        raise RuntimeError(
            f"No fold artifacts found in {results_dir}. Run training first."
        )

    class_names = [f"Class {i + 1}" for i in range(cfg.n_classes)]
    plot_confusion_matrix(total_cm_fp32, class_names, cfg.subject_id, "FP32", results_dir)
    plot_confusion_matrix(total_cm_int8, class_names, cfg.subject_id, "INT8", results_dir)

    summary_path = results_dir / f"accuracies_subject{cfg.subject_id}.csv"
    completed_folds = [f for f in range(1, cfg.n_folds + 1) if f not in missing]
    with open(summary_path, "w") as fh:
        fh.write("fold,fp32_acc,int8_acc\n")
        for fold, fp32, int8 in zip(completed_folds, test_accs_fp32, test_accs_int8):
            fh.write(f"{fold},{fp32},{int8}\n")

    logger.info(
        "Subject %d summary (%d/%d folds) — FP32=%.4f±%.4f  INT8=%.4f±%.4f",
        cfg.subject_id, len(test_accs_fp32), cfg.n_folds,
        float(np.mean(test_accs_fp32)), float(np.std(test_accs_fp32)),
        float(np.mean(test_accs_int8)), float(np.std(test_accs_int8)),
    )
    logger.info("Accuracy CSV → %s", summary_path)


# ── Inference pipeline ────────────────────────────────────────────────────────


def run_infer(
    cfg: Config,
    base_dir: Path,
    fold: int = 1,
    device: torch.device = DEVICE,
) -> None:
    """Load saved artifacts for *fold* and run inference on the evaluation set.

    Generates the full suite of analysis plots for both the test and training
    sets using the saved CSP, feature-selection indices, and SNN model.
    """
    data_dir = _resolve_data_dir(cfg, base_dir)
    results_dir = _resolve_results_dir(cfg, base_dir)

    params, csp, model = _load_fold_artifacts(results_dir, cfg.subject_id, fold, device)
    enc = params["encoding_params"]
    freq_bands = [tuple(b) for b in params["freq_bands"]]

    fs_info = params.get("feature_selection", {})
    percentile_kept: float = fs_info.get("percentile_kept", 100.0)
    top_indices = (
        torch.tensor(fs_info["top_indices"], dtype=torch.long, device=device)
        if percentile_kept < 100.0
        else None
    )

    def _preprocess_and_encode(X: np.ndarray, y_raw: np.ndarray):
        X_filt = _multiband_filter(X, freq_bands)
        proj = csp.transform(X_filt)
        spk = encode_to_spikes(proj, enc["base_thresh"], enc["adapt_inc"], enc["decay"], device)
        if top_indices is not None:
            spk = spk[:, :, top_indices]
        y_t = torch.tensor(y_raw - 1, dtype=torch.long, device=device)
        return spk, y_t

    # ── Load raw data (source-aware) ──────────────────────────────────────────
    if cfg.source == "moabb":
        X_train_raw, y_train_raw, X_test_raw, y_test_raw, _ = load_moabb_subject(
            cfg.moabb_dataset, cfg.subject_id,
            tmin=cfg.tmin, tmax=cfg.tmax, n_classes=cfg.n_classes,
        )
    else:
        X_test_full, y_test_full = load_data(data_dir, cfg.subject_id, _SESSION_EVAL)
        X_test_raw = X_test_full[y_test_full > 0]
        y_test_raw = y_test_full[y_test_full > 0]
        X_train_full, y_train_full = load_data(data_dir, cfg.subject_id, _SESSION_TRAIN)
        X_train_raw = X_train_full[y_train_full > 0]
        y_train_raw = y_train_full[y_train_full > 0]

    # ── Test set ──────────────────────────────────────────────────────────────
    spk_test, y_test_t = _preprocess_and_encode(X_test_raw, y_test_raw)

    test_acc, cm = evaluate(model, spk_test, y_test_t, device)
    logger.info("Test accuracy (fold %d): %.4f", fold, test_acc)
    logger.info("Confusion matrix:\n%s", cm)

    tag = f"test_fold{fold}"
    plot_spike_count_difference(spk_test, y_test_t, cfg.subject_id, fold, results_dir, tag=tag)
    plot_discriminative_importance(spk_test, y_test_t, cfg.subject_id, fold, results_dir, tag=tag)
    plot_spike_probability_heatmap(spk_test, y_test_t, cfg.subject_id, fold, results_dir, tag=tag)

    # ── Training set ──────────────────────────────────────────────────────────
    spk_train, y_train_t = _preprocess_and_encode(X_train_raw, y_train_raw)

    tag_tr = f"train_fold{fold}"
    plot_spike_count_difference(spk_train, y_train_t, cfg.subject_id, fold, results_dir, tag=tag_tr)
    plot_discriminative_importance(spk_train, y_train_t, cfg.subject_id, fold, results_dir, tag=tag_tr)
    plot_spike_probability_heatmap(spk_train, y_train_t, cfg.subject_id, fold, results_dir, tag=tag_tr)
