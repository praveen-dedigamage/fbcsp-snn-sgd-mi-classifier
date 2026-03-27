#!/usr/bin/env python3
"""Entry point for the FBCSP-SNN Motor Imagery EEG Classifier.

Usage
-----
Train (10-fold cross-validation for subject 1)::

    python main.py train --subject-id 1

Inference (load fold 3 and run analysis plots)::

    python main.py infer --subject-id 1 --fold 3

Run ``python main.py train --help`` or ``python main.py infer --help`` for the
full list of options.
"""

from pathlib import Path

from fbcsp_snn import DEVICE, setup_logger
from fbcsp_snn.config import config_from_args, parse_args
from fbcsp_snn.pipeline import run_aggregate, run_infer, run_train

logger = setup_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    args = parse_args()
    cfg = config_from_args(args)

    logger.info("Device: %s", DEVICE)
    logger.info("Mode:   %s", args.mode)
    logger.info("Subject: %d", cfg.subject_id)

    if args.mode == "train":
        run_train(cfg, base_dir=BASE_DIR, device=DEVICE, fold_id=getattr(args, "fold_id", None))
    elif args.mode == "infer":
        run_infer(cfg, base_dir=BASE_DIR, fold=args.fold, device=DEVICE)
    elif args.mode == "aggregate":
        run_aggregate(cfg, base_dir=BASE_DIR)


if __name__ == "__main__":
    main()
