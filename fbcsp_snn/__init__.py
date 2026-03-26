"""FBCSP-SNN Motor Imagery EEG Classifier package."""

import logging

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    # Optimise cuDNN kernel selection for fixed input sizes
    torch.backends.cudnn.benchmark = True
    # Allow TF32 on Ampere+ GPUs (free ~2× matmul throughput, negligible precision loss)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    logger.propagate = False
    return logger
