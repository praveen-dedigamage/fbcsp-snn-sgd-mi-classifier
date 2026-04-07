"""Configuration dataclass and CLI argument parsing."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class Config:
    # Data source
    source: str = "file"                # "file" | "moabb"
    moabb_dataset: str = "BNCI2014_001" # used when source="moabb"
    tmin: float = 0.5                   # epoch start (s) relative to cue
    tmax: float = 3.5                   # epoch end   (s) relative to cue

    # Subject / paths
    subject_id: int = 1
    data_dir: Optional[Path] = None     # resolved at runtime when None (file mode)
    results_dir: Optional[Path] = None  # resolved at runtime when None

    # Preprocessing
    freq_bands: List[Tuple[int, int]] = field(
        default_factory=lambda: [(4, 10), (10, 14), (14, 30)]
    )
    lambda_r: float = 0.0001
    csp_components_per_band: int = 22

    # Spike encoding
    base_thresh: float = 0.001
    adapt_inc: float = 0.6
    decay: float = 0.95

    # Model
    hidden_neurons: int = 64
    population_per_class: int = 20
    n_classes: int = 4
    beta: float = 0.95
    dropout_prob: float = 0.5

    # Training
    spiking_prob: float = 0.7
    lr: float = 1e-3
    epochs: int = 1000
    weight_decay: float = 0.1
    n_folds: int = 10
    early_stopping_patience: int = 100
    early_stopping_warmup: int = 100

    # Feature selection (percentile of features to keep)
    feature_percentile: float = 25.0

    # Robustness / hard-subject improvements
    euclidean_alignment: bool = False     # align each session's mean cov → I before CSP
    artifact_rejection_threshold: float = 0.0  # σ above mean ptp to reject (0 = off)
    ledoit_wolf: bool = False             # Ledoit-Wolf shrinkage instead of sample cov


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    # Data source
    parser.add_argument(
        "--source",
        choices=["file", "moabb"],
        default="file",
        help="'file' loads .mat files from --data-dir; 'moabb' downloads a public dataset",
    )
    parser.add_argument(
        "--moabb-dataset",
        type=str,
        default="BNCI2014_001",
        metavar="NAME",
        help="MOABB dataset name (used when --source=moabb). "
             "Options: BNCI2014_001, PhysionetMI, Cho2017, BNCI2015_001",
    )
    parser.add_argument(
        "--tmin", type=float, default=0.5,
        help="Epoch start in seconds relative to cue (moabb source)",
    )
    parser.add_argument(
        "--tmax", type=float, default=3.5,
        help="Epoch end in seconds relative to cue (moabb source)",
    )
    parser.add_argument("--subject-id", type=int, default=1, metavar="N")
    parser.add_argument("--data-dir", type=Path, default=None, metavar="PATH")
    parser.add_argument("--results-dir", type=Path, default=None, metavar="PATH")
    parser.add_argument(
        "--freq-bands",
        type=str,
        default="[(4,10),(10,14),(14,30)]",
        metavar="BANDS",
        help="Python literal list of (low, high) Hz tuples, e.g. '[(4,10),(10,14)]'",
    )
    parser.add_argument("--lambda-r", type=float, default=0.0001)
    parser.add_argument("--csp-components-per-band", type=int, default=22)
    parser.add_argument("--base-thresh", type=float, default=0.001)
    parser.add_argument("--adapt-inc", type=float, default=0.6)
    parser.add_argument("--decay", type=float, default=0.95)
    parser.add_argument("--hidden-neurons", type=int, default=64)
    parser.add_argument("--population-per-class", type=int, default=20)
    parser.add_argument(
        "--n-classes", type=int, default=None,
        help="Number of MI classes. Auto-detected from the dataset registry when "
             "--source=moabb and not specified; defaults to 4 for file source.",
    )
    parser.add_argument("--beta", type=float, default=0.95)
    parser.add_argument("--dropout-prob", type=float, default=0.5)
    parser.add_argument("--feature-percentile", type=float, default=25.0)

    # ── Hard-subject robustness options ──────────────────────────────────────
    parser.add_argument(
        "--euclidean-alignment",
        action="store_true",
        default=False,
        help="Apply Euclidean Alignment (EA) to each set independently before CSP. "
             "Strongly recommended for subjects with high inter-session variability "
             "(e.g. BCI-IV-2a subjects 2, 5, 6, 9).",
    )
    parser.add_argument(
        "--artifact-rejection-threshold",
        type=float,
        default=0.0,
        metavar="SIGMA",
        help="Reject training trials whose peak-to-peak amplitude exceeds "
             "mean + SIGMA * std across trials. 0 disables rejection. "
             "Typical value: 3.0.",
    )
    parser.add_argument(
        "--ledoit-wolf",
        action="store_true",
        default=False,
        help="Use Ledoit-Wolf shrinkage covariance in CSP instead of the "
             "sample covariance. More stable for low-SNR subjects.",
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments; returns a Namespace with a ``mode`` attribute."""
    root = argparse.ArgumentParser(
        description="FBCSP-SNN Motor Imagery EEG Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = root.add_subparsers(dest="mode", required=True)

    # ── train subcommand ──────────────────────────────────────────────────────
    train_p = sub.add_parser(
        "train",
        help="Run cross-validated training for a subject",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_shared_args(train_p)
    train_p.add_argument("--spiking-prob", type=float, default=0.7)
    train_p.add_argument("--lr", type=float, default=1e-3)
    train_p.add_argument("--epochs", type=int, default=1000)
    train_p.add_argument("--weight-decay", type=float, default=0.1)
    train_p.add_argument("--n-folds", type=int, default=10)
    train_p.add_argument(
        "--early-stopping-patience", type=int, default=100, metavar="N"
    )
    train_p.add_argument(
        "--early-stopping-warmup", type=int, default=100, metavar="N"
    )
    train_p.add_argument(
        "--fold-id", type=int, default=None, metavar="N",
        help="Run only this fold (1-based). Omit to run all folds sequentially. "
             "Use with SLURM --array=1-<n-folds> to run folds as parallel jobs.",
    )

    # ── infer subcommand ──────────────────────────────────────────────────────
    infer_p = sub.add_parser(
        "infer",
        help="Run inference with a previously saved model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_shared_args(infer_p)
    infer_p.add_argument(
        "--fold", type=int, default=1, metavar="N", help="Saved fold to load"
    )
    infer_p.add_argument(
        "--sample-idx",
        type=int,
        default=0,
        metavar="N",
        help="Sample index for single-sample visualizations",
    )

    # ── aggregate subcommand ──────────────────────────────────────────────────
    agg_p = sub.add_parser(
        "aggregate",
        help="Collect per-fold artifacts and produce summary CSV + confusion-matrix plots",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    agg_p.add_argument("--subject-id", type=int, default=1, metavar="N")
    agg_p.add_argument("--results-dir", type=Path, default=None, metavar="PATH")
    agg_p.add_argument("--n-folds", type=int, default=10)
    agg_p.add_argument(
        "--n-classes", type=int, default=None,
        help="Number of classes (auto-detected from saved artifacts when omitted)",
    )

    return root.parse_args()


def config_from_args(args: argparse.Namespace) -> Config:
    """Build a :class:`Config` from parsed CLI arguments."""
    if args.mode == "aggregate":
        # Minimal config — only subject routing and fold count are needed
        n_classes = args.n_classes if args.n_classes is not None else 4
        cfg = Config(
            subject_id=args.subject_id,
            results_dir=args.results_dir,
            n_folds=args.n_folds,
            n_classes=n_classes,
        )
        return cfg

    freq_bands: List[Tuple[int, int]] = ast.literal_eval(args.freq_bands)

    # Resolve n_classes: auto-detect from registry for MOABB sources so that
    # 2-class datasets (Cho2017, BNCI2015_001) work without --n-classes 2.
    if args.n_classes is not None:
        n_classes = args.n_classes
    elif args.source == "moabb":
        from fbcsp_snn.datasets import _REGISTRY
        n_classes = _REGISTRY.get(args.moabb_dataset, {}).get("default_classes", 4)
    else:
        n_classes = 4

    cfg = Config(
        source=args.source,
        moabb_dataset=args.moabb_dataset,
        tmin=args.tmin,
        tmax=args.tmax,
        subject_id=args.subject_id,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        freq_bands=freq_bands,
        lambda_r=args.lambda_r,
        csp_components_per_band=args.csp_components_per_band,
        base_thresh=args.base_thresh,
        adapt_inc=args.adapt_inc,
        decay=args.decay,
        hidden_neurons=args.hidden_neurons,
        population_per_class=args.population_per_class,
        n_classes=n_classes,
        beta=args.beta,
        dropout_prob=args.dropout_prob,
        feature_percentile=args.feature_percentile,
        euclidean_alignment=args.euclidean_alignment,
        artifact_rejection_threshold=args.artifact_rejection_threshold,
        ledoit_wolf=args.ledoit_wolf,
    )

    if args.mode == "train":
        cfg.spiking_prob = args.spiking_prob
        cfg.lr = args.lr
        cfg.epochs = args.epochs
        cfg.weight_decay = args.weight_decay
        cfg.n_folds = args.n_folds
        cfg.early_stopping_patience = args.early_stopping_patience
        cfg.early_stopping_warmup = args.early_stopping_warmup

    return cfg
