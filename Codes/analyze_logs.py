import os
import glob
import re
import pandas as pd
from tqdm import tqdm

def parse_epochs_from_text(text):
    """
    Given the full text of a .log, extract all epoch‐metrics lines of the form:
      Epoch 17 Train Loss: 0.0728, Train Acc: 29.07%, Train Incorrect: 0.7484, Val Loss: 0.0719, Val Acc: 41.34%, Val Incorrect: 0.7474
    Returns a list of dicts, each with keys:
      - epoch (int)
      - train_loss (float)
      - train_acc_pct (float)
      - train_incorrect (float)
      - val_loss (float)
      - val_acc_pct (float)
      - val_incorrect (float)
    """
    epoch_pattern = re.compile(
        r"Epoch\s+(\d+)\s+Train\s+Loss:\s*([\d\.]+),\s*"
        r"Train\s+Acc:\s*([\d\.]+)%,\s*Train\s+Incorrect:\s*([\d\.]+),\s*"
        r"Val\s+Loss:\s*([\d\.]+),\s*Val\s+Acc:\s*([\d\.]+)%,\s*Val\s+Incorrect:\s*([\d\.]+)",
        re.IGNORECASE
    )

    rows = []
    for m in epoch_pattern.finditer(text):
        rows.append({
            "epoch":            int(m.group(1)),
            "train_loss":       float(m.group(2)),
            "train_acc_pct":    float(m.group(3)),
            "train_incorrect":  float(m.group(4)),
            "val_loss":         float(m.group(5)),
            "val_acc_pct":      float(m.group(6)),
            "val_incorrect":    float(m.group(7))
        })
    return rows

def extract_parameters(text):
    """
    Look for a line like:
      [INFO] Loading existing CSP weights for Subject 1, FB: [(4, 8), (8, 13), (13, 30)], Lambda: 0.0005 from /scratch/…
    Return a dict with keys:
      - subject_id   (int)
      - fb_pairs     (str)
      - lambda_val   (float)
      - weights_path (str)
    If no match, return None.
    """
    pattern = re.compile(
        r"Loading existing CSP weights for Subject\s+(\d+),\s*"
        r"FB:\s*(\[[^\]]+\]),\s*Lambda:\s*([\d\.]+)\s*"
        r"from\s+(\S+)",
        re.IGNORECASE
    )
    m = pattern.search(text)
    if not m:
        return None

    return {
        "subject_id":   int(m.group(1)),
        "fb_pairs":     m.group(2),
        "lambda_val":   float(m.group(3)),
        "weights_path": m.group(4)
    }

def extract_last_best_model(text):
    """
    Find all lines like:
      [INFO] Best model updated at epoch 17 with val_acc=0.4585
    Return a dict with:
      - best_epoch   (int)
      - best_val_acc (float)
    If none found, return None.
    """
    pattern = re.compile(
        r"Best model updated at epoch\s+(\d+)\s+with val_acc=([\d\.]+)",
        re.IGNORECASE
    )
    best_epoch = -1
    best_val   = None
    best_pos   = -1

    for m in pattern.finditer(text):
        epoch   = int(m.group(1))
        val_acc = float(m.group(2))
        pos     = m.start()
        if (epoch > best_epoch) or (epoch == best_epoch and pos > best_pos):
            best_epoch = epoch
            best_val   = val_acc
            best_pos   = pos

    if best_epoch < 0:
        return None
    return {"best_epoch": best_epoch, "best_val_acc": best_val}

def process_single_log(path_to_log):
    """
    Open one .log file, extract:
      • hyperparameters (subject_id, fb_pairs, lambda_val, weights_path)
      • all epoch metrics (list of dicts)
      • final best-model info (best_epoch, best_val_acc)
      • metrics at best_epoch (train_loss, train_acc_pct, val_loss, val_acc_pct)
    Returns a dict with:
      {
        filename, filepath,
        subject_id, fb_pairs, lambda_val, weights_path,
        best_epoch, best_val_acc,
        best_train_loss, best_train_acc_pct,
        best_val_loss, best_val_acc_pct
      }
    """
    with open(path_to_log, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # Extract parameters
    params = extract_parameters(text) or {
        "subject_id":   None,
        "fb_pairs":     None,
        "lambda_val":   None,
        "weights_path": None
    }

    # Extract all epoch metrics
    epoch_rows = parse_epochs_from_text(text)

    # Extract last best-model info
    best_info = extract_last_best_model(text) or {"best_epoch": None, "best_val_acc": None}
    be = best_info["best_epoch"]

    # Find metrics for best_epoch
    best_metrics = {
        "best_train_loss":    None,
        "best_train_acc_pct": None,
        "best_val_loss":      None,
        "best_val_acc_pct":   None
    }
    if be is not None:
        for row in epoch_rows:
            if row["epoch"] == be:
                best_metrics["best_train_loss"]    = row["train_loss"]
                best_metrics["best_train_acc_pct"] = row["train_acc_pct"]
                best_metrics["best_val_loss"]      = row["val_loss"]
                best_metrics["best_val_acc_pct"]   = row["val_acc_pct"]
                break

    return {
        "filename":            os.path.basename(path_to_log),
        "filepath":            path_to_log,
        "subject_id":          params["subject_id"],
        "fb_pairs":            params["fb_pairs"],
        "lambda_val":          params["lambda_val"],
        "weights_path":        params["weights_path"],
        "best_epoch":          best_info["best_epoch"],
        "best_val_acc":        best_info["best_val_acc"],
        "best_train_loss":     best_metrics["best_train_loss"],
        "best_train_acc_pct":  best_metrics["best_train_acc_pct"],
        "best_val_loss":       best_metrics["best_val_loss"],
        "best_val_acc_pct":    best_metrics["best_val_acc_pct"]
    }

def build_summary_dataframe(root_dir, recursive=True):
    """
    Walk `root_dir` (and subfolders if recursive=True), find all .log files,
    call process_single_log for each, and return a pandas DataFrame.
    """
    pattern   = "**/*.log" if recursive else "*.log"
    log_paths = glob.glob(os.path.join(root_dir, pattern), recursive=recursive)
    if not log_paths:
        raise RuntimeError(f"No .log files found under: {root_dir}")

    records = []
    for path in tqdm(log_paths, desc="Scanning .log files"):
        records.append(process_single_log(path))

    df = pd.DataFrame(records)
    cols = [
        "filename", "filepath",
        "subject_id", "fb_pairs", "lambda_val", "weights_path",
        "best_epoch", "best_val_acc",
        "best_train_loss", "best_train_acc_pct",
        "best_val_loss", "best_val_acc_pct"
    ]
    return df[cols]

if __name__ == "__main__":
    # Assuming “logs” folder is in the same directory as this script:
    root_dir = "./logs"

    # Build a DataFrame summarizing every .log file
    summary_df = build_summary_dataframe(root_dir)

    print("Full summary (one row per .log file):")
    print(summary_df)

    # Now select, for each subject_id, the run with highest best_val_acc:
    # Method: sort + drop_duplicates
    sorted_df = summary_df.sort_values(
        by=["subject_id", "best_val_acc"],
        ascending=[True, False]
    )
    best_models_df = sorted_df.drop_duplicates(subset="subject_id", keep="first").reset_index(drop=True)

    print("\nBest‐model parameters plus losses/accuracies for each subject:")
    print(best_models_df)
