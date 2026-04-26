"""
train.py
--------
Training loop for the EuroSAT MLP classifier.

Usage
-----
    python train.py [--data_dir EuroSAT_RGB] [--epochs 50] [--batch_size 256]
                    [--lr 1e-3] [--hidden1 512] [--hidden2 256]
                    [--weight_decay 1e-4] [--lr_decay 0.95]
                    [--activation relu] [--save_path best_model.npz]
                    [--seed 42]

Outputs
-------
  best_model.npz      – weights of the epoch with highest val accuracy
  train_history.npz   – loss/accuracy arrays for plotting
"""

import argparse
import os
import time
import numpy as np

from data_loader import load_dataset, DataLoader, CLASS_NAMES, INPUT_DIM, NUM_CLASSES
from model       import MLP
from optimizer   import SGD, cross_entropy_loss


def compute_accuracy(model: MLP, X: np.ndarray, y: np.ndarray,
                     batch_size: int = 512) -> float:
    correct = 0
    n = len(X)
    for start in range(0, n, batch_size):
        Xb = X[start : start + batch_size]
        yb = y[start : start + batch_size]
        preds   = model.predict(Xb)
        correct += (preds == yb).sum()
    return correct / n


def compute_loss(model: MLP, X: np.ndarray, y: np.ndarray,
                 weight_decay: float, batch_size: int = 512) -> float:
    total_loss = 0.0
    n = len(X)
    for start in range(0, n, batch_size):
        Xb    = X[start : start + batch_size]
        yb    = y[start : start + batch_size]
        probs = model.forward(Xb)
        # No L2 penalty here to keep val/test loss unaffected
        total_loss += cross_entropy_loss(probs, yb) * len(yb)
    return total_loss / n


def train(data_dir:     str   = "EuroSAT_RGB",
          epochs:       int   = 50,
          batch_size:   int   = 256,
          lr:           float = 1e-3,
          hidden1:      int   = 512,
          hidden2:      int   = 256,
          weight_decay: float = 1e-4,
          lr_decay:     float = 0.95,
          momentum:     float = 0.9,
          activation:   str   = "relu",
          save_path:    str   = "outputs/best_model.npz",
          seed:         int   = 42,
          verbose:      bool  = True):

    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    data = load_dataset(data_dir, seed=seed)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val,   y_val   = data["X_val"],   data["y_val"]

    train_loader = DataLoader(X_train, y_train, batch_size=batch_size, seed=seed)

    model = MLP(input_dim=INPUT_DIM,
                hidden1=hidden1,
                hidden2=hidden2,
                num_classes=NUM_CLASSES,
                activation=activation,
                seed=seed)

    opt = SGD(model,
              lr=lr,
              momentum=momentum,
              weight_decay=weight_decay,
              lr_decay=lr_decay)

    if verbose:
        print(model)
        print(opt)
        print(f"Training for {epochs} epochs  |  batch_size={batch_size}")
        print("─" * 65)

    history = {
        "train_loss": [],
        "val_loss":   [],
        "val_acc":    [],
        "lr":         [],
    }

    best_val_acc  = -1.0
    best_epoch    = -1

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model._cache = {}  

        epoch_loss = 0.0
        n_batches  = 0
        for Xb, yb in train_loader:
            probs     = model.forward(Xb)
            batch_loss = cross_entropy_loss(probs, yb,
                                            model=model,
                                            weight_decay=weight_decay)
            grads     = model.backward(yb, probs, weight_decay=weight_decay)
            opt.step(grads)
            epoch_loss += batch_loss
            n_batches  += 1

        train_loss = epoch_loss / n_batches

        val_loss = compute_loss(model, X_val, y_val, weight_decay=0.0)
        val_acc  = compute_accuracy(model, X_val, y_val)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(opt.get_lr())

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            model.save_weights(save_path)

        opt.decay_lr()

        elapsed = time.time() - t0
        if verbose:
            marker = " ★" if epoch == best_epoch else ""
            print(f"Epoch [{epoch:>3}/{epochs}]  "
                  f"train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  "
                  f"val_acc={val_acc*100:.2f}%  "
                  f"lr={opt.get_lr():.2e}  "
                  f"({elapsed:.1f}s){marker}")

    if verbose:
        print("─" * 65)
        print(f"Best val accuracy: {best_val_acc*100:.2f}%  (epoch {best_epoch})")
        print(f"Best model saved → {save_path}")

    history_path = save_path.replace(".npz", "_history.npz")
    np.savez(history_path, **{k: np.array(v) for k, v in history.items()})
    if verbose:
        print(f"Training history saved → {history_path}")

    return history


def parse_args():
    p = argparse.ArgumentParser(description="Train EuroSAT MLP classifier")
    p.add_argument("--data_dir",     default="EuroSAT_RGB")
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch_size",   type=int,   default=256)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--hidden1",      type=int,   default=512)
    p.add_argument("--hidden2",      type=int,   default=256)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lr_decay",     type=float, default=0.95)
    p.add_argument("--momentum",     type=float, default=0.9)
    p.add_argument("--activation",   default="relu",
                   choices=["relu", "sigmoid", "tanh"])
    p.add_argument("--save_path",    default="outputs/best_model.npz")
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(data_dir     = args.data_dir,
          epochs       = args.epochs,
          batch_size   = args.batch_size,
          lr           = args.lr,
          hidden1      = args.hidden1,
          hidden2      = args.hidden2,
          weight_decay = args.weight_decay,
          lr_decay     = args.lr_decay,
          momentum     = args.momentum,
          activation   = args.activation,
          save_path    = args.save_path,
          seed         = args.seed)
