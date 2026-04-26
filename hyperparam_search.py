"""
hyperparam_search.py
--------------------
Hyperparameter search for the EuroSAT MLP classifier.

Supports two modes:
  --mode grid    – exhaustive grid search over all combinations
  --mode random  – random search over a fixed number of trials

Searched hyperparameters:
  - learning rate         (lr)
  - hidden layer sizes    (hidden1 × hidden2)
  - L2 regularisation     (weight_decay)
  - activation function   (activation)

Results are logged to  search_results.csv  (sortable by val_acc).

Usage
-----
    python hyperparam_search.py --data_dir EuroSAT_RGB --mode grid --epochs 20
    python hyperparam_search.py --data_dir EuroSAT_RGB --mode random --n_trials 20 --epochs 20
"""

import argparse
import csv
import itertools
import os
import random
import time
import numpy as np

from data_loader import load_dataset, DataLoader, CLASS_NAMES, INPUT_DIM, NUM_CLASSES
from model       import MLP
from optimizer   import SGD, cross_entropy_loss
from train       import compute_accuracy, compute_loss


# Search space

GRID = {
    "lr":           [1e-2, 1e-3, 5e-4],
    "hidden1":      [256, 512],
    "hidden2":      [128, 256],
    "weight_decay": [1e-3, 1e-4, 1e-5],
    "activation":   ["relu", "tanh", "sigmoid"],
}

# Random search ranges
RANDOM_RANGES = {
    "lr":           (1e-4, 5e-2),   
    "hidden1":      [256, 512, 1024],
    "hidden2":      [128, 256, 512],
    "weight_decay": (1e-5, 1e-2),   
    "activation":   ["relu", "tanh", "sigmoid"],
}


def _sample_log_uniform(low, high, rng):
    log_low  = np.log10(low)
    log_high = np.log10(high)
    return float(10 ** rng.uniform(log_low, log_high))


def random_config(rng):
    return {
        "lr":           _sample_log_uniform(*RANDOM_RANGES["lr"], rng),
        "hidden1":      int(rng.choice(RANDOM_RANGES["hidden1"])),
        "hidden2":      int(rng.choice(RANDOM_RANGES["hidden2"])),
        "weight_decay": _sample_log_uniform(*RANDOM_RANGES["weight_decay"], rng),
        "activation":   str(rng.choice(RANDOM_RANGES["activation"])),
    }


def run_trial(config: dict,
              data:   dict,
              epochs: int   = 20,
              batch_size: int = 256,
              lr_decay:   float = 0.95,
              momentum:   float = 0.9,
              seed:       int   = 42) -> dict:
    X_train, y_train = data["X_train"], data["y_train"]
    X_val,   y_val   = data["X_val"],   data["y_val"]

    train_loader = DataLoader(X_train, y_train, batch_size=batch_size, seed=seed)

    model = MLP(input_dim=INPUT_DIM,
                hidden1=config["hidden1"],
                hidden2=config["hidden2"],
                num_classes=NUM_CLASSES,
                activation=config["activation"],
                seed=seed)

    opt = SGD(model,
              lr=config["lr"],
              momentum=momentum,
              weight_decay=config["weight_decay"],
              lr_decay=lr_decay)

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        for Xb, yb in train_loader:
            probs = model.forward(Xb)
            grads = model.backward(yb, probs,
                                   weight_decay=config["weight_decay"])
            opt.step(grads)
        opt.decay_lr()

        val_acc = compute_accuracy(model, X_val, y_val)
        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return {"val_acc": best_val_acc}


def search(data_dir:   str   = "EuroSAT_RGB",
           mode:       str   = "grid",
           n_trials:   int   = 20,
           epochs:     int   = 20,
           batch_size: int   = 256,
           output_csv: str   = "outputs/search_results.csv",
           seed:       int   = 42):

    rng  = np.random.default_rng(seed)

    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    data = load_dataset(data_dir, seed=seed)

    if mode == "grid":
        keys   = list(GRID.keys())
        combos = list(itertools.product(*GRID.values()))
        configs = [dict(zip(keys, c)) for c in combos]
        print(f"[Search] Grid search: {len(configs)} combinations")
    else:
        configs = [random_config(rng) for _ in range(n_trials)]
        print(f"[Search] Random search: {n_trials} trials")

    fieldnames = ["trial", "lr", "hidden1", "hidden2",
                  "weight_decay", "activation", "val_acc", "elapsed_s"]

    results = []
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, cfg in enumerate(configs, start=1):
            t0 = time.time()
            print(f"  Trial {i:>3}/{len(configs)}: {cfg} …", end=" ", flush=True)
            result = run_trial(cfg, data,
                               epochs=epochs,
                               batch_size=batch_size,
                               seed=seed)
            elapsed = time.time() - t0
            row = {
                "trial":        i,
                "lr":           f"{cfg['lr']:.2e}",
                "hidden1":      cfg["hidden1"],
                "hidden2":      cfg["hidden2"],
                "weight_decay": f"{cfg['weight_decay']:.2e}",
                "activation":   cfg["activation"],
                "val_acc":      f"{result['val_acc']*100:.2f}",
                "elapsed_s":    f"{elapsed:.1f}",
            }
            writer.writerow(row)
            f.flush()
            results.append((result["val_acc"], cfg))
            print(f"val_acc={result['val_acc']*100:.2f}%  ({elapsed:.1f}s)")

    results.sort(reverse=True, key=lambda x: x[0])
    print("\n" + "=" * 60)
    print(f"Top-5 configurations (out of {len(configs)}):")
    for rank, (acc, cfg) in enumerate(results[:5], start=1):
        print(f"  #{rank}  val_acc={acc*100:.2f}%  {cfg}")
    print(f"\nFull results saved → {output_csv}")

    return results[0][1]   # return best config


def parse_args():
    p = argparse.ArgumentParser(description="Hyperparameter search for EuroSAT MLP")
    p.add_argument("--data_dir",   default="EuroSAT_RGB")
    p.add_argument("--mode",       default="grid",
                   choices=["grid", "random"])
    p.add_argument("--n_trials",   type=int,   default=20,
                   help="Number of trials for random search")
    p.add_argument("--epochs",     type=int,   default=20,
                   help="Epochs per trial (use fewer than full training)")
    p.add_argument("--batch_size", type=int,   default=256)
    p.add_argument("--output_csv", default="outputs/search_results.csv")
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    best = search(data_dir   = args.data_dir,
                  mode       = args.mode,
                  n_trials   = args.n_trials,
                  epochs     = args.epochs,
                  batch_size = args.batch_size,
                  output_csv = args.output_csv,
                  seed       = args.seed)
    print("\nBest config found:")
    for k, v in best.items():
        print(f"  {k}: {v}")
