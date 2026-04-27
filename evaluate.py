"""
evaluate.py
-----------
加载训练好的模型权重，在测试集上评估分类准确率，并生成混淆矩阵。

用法：
    python evaluate.py [--data_dir EuroSAT_RGB] [--weights outputs/best_model.npz]
                       [--hidden1 512] [--hidden2 256] [--activation relu]
                       [--batch_size 512] [--seed 42]
"""

import argparse
import os
import numpy as np

from data_loader import load_dataset, CLASS_NAMES, INPUT_DIM, NUM_CLASSES
from model       import MLP


def confusion_matrix(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     num_classes: int = NUM_CLASSES) -> np.ndarray:
    """返回 num_classes × num_classes 的混淆矩阵。"""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def print_confusion_matrix(cm: np.ndarray, class_names=CLASS_NAMES):
    max_name = max(len(n) for n in class_names)
    header   = " " * (max_name + 2) + "  ".join(f"{n[:6]:>6}" for n in class_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = f"{class_names[i]:<{max_name}}  " + "  ".join(f"{v:>6}" for v in row)
        print(row_str)


def plot_confusion_matrix(cm: np.ndarray,
                          class_names=CLASS_NAMES,
                          save_path: str = "outputs/confusion_matrix.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)

        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(class_names, fontsize=9)

        # 在格子内标注数值
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm[i, j] > thresh else "black"
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center", fontsize=8, color=color)

        ax.set_ylabel("True label",      fontsize=11)
        ax.set_xlabel("Predicted label", fontsize=11)
        ax.set_title("Confusion Matrix – Test Set", fontsize=13)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[Evaluate] Confusion matrix plot saved → {save_path}")
    except ImportError:
        print("[Evaluate] matplotlib not available – skipping plot.")


def evaluate(data_dir:   str   = "EuroSAT_RGB",
             weights:    str   = "outputs/best_model.npz",
             hidden1:    int   = 512,
             hidden2:    int   = 256,
             activation: str   = "relu",
             batch_size: int   = 512,
             output_dir: str   = "outputs",
             seed:       int   = 42):

    os.makedirs(output_dir, exist_ok=True)

    data   = load_dataset(data_dir, seed=seed)
    X_test = data["X_test"]
    y_test = data["y_test"]
    print(f"[Evaluate] Test set: {len(X_test)} samples")

    model = MLP(input_dim=INPUT_DIM,
                hidden1=hidden1,
                hidden2=hidden2,
                num_classes=NUM_CLASSES,
                activation=activation,
                seed=seed)
    model.load_weights(weights)

    all_preds = []
    for start in range(0, len(X_test), batch_size):
        Xb    = X_test[start : start + batch_size]
        preds = model.predict(Xb)
        all_preds.extend(preds.tolist())
    y_pred = np.array(all_preds)

    acc = (y_pred == y_test).mean()
    print(f"\n[Evaluate] Test Accuracy: {acc * 100:.2f}%\n")

    print("Per-class Accuracy:")
    print("-" * 40)
    for c, name in enumerate(CLASS_NAMES):
        mask    = y_test == c
        c_acc   = (y_pred[mask] == y_test[mask]).mean() if mask.sum() > 0 else 0.0
        print(f"  {name:<25} {c_acc * 100:6.2f}%  (n={mask.sum()})")

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix (rows=true, cols=predicted):")
    print_confusion_matrix(cm)

    # 保存混淆矩阵
    npy_path = os.path.join(output_dir, "confusion_matrix.npy")
    png_path = os.path.join(output_dir, "confusion_matrix.png")
    np.save(npy_path, cm)
    print(f"\n[Evaluate] confusion_matrix.npy saved → {npy_path}")
    plot_confusion_matrix(cm, save_path=png_path)

    return acc, cm, y_pred


def parse_args():
    p = argparse.ArgumentParser(description="EuroSAT MLP 测试集评估")
    p.add_argument("--data_dir",    default="EuroSAT_RGB")
    p.add_argument("--weights",     default="outputs/best_model.npz")
    p.add_argument("--hidden1",     type=int, default=512)
    p.add_argument("--hidden2",     type=int, default=256)
    p.add_argument("--activation",  default="relu",
                   choices=["relu", "sigmoid", "tanh"])
    p.add_argument("--batch_size",  type=int, default=512)
    p.add_argument("--output_dir",  default="outputs")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(data_dir   = args.data_dir,
             weights    = args.weights,
             hidden1    = args.hidden1,
             hidden2    = args.hidden2,
             activation = args.activation,
             batch_size = args.batch_size,
             output_dir = args.output_dir,
             seed       = args.seed)
