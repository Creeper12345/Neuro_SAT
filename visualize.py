"""
visualize.py
------------
All visualisation utilities for the EuroSAT MLP project.

Functions
---------
plot_training_curves   – train/val loss + val accuracy vs. epoch
plot_weight_images     – first-layer weights reshaped to 64×64×3 images
plot_error_examples    – misclassified test images with true/predicted labels

Usage (standalone)
------------------
    python visualize.py --data_dir EuroSAT_RGB --weights best_model.npz
                        --history best_model_history.npz
                        [--hidden1 512] [--hidden2 256] [--activation relu]
                        [--n_weight_imgs 20] [--n_error_imgs 12] [--seed 42]
"""

import argparse
import os
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as _fm

from data_loader import load_dataset, CLASS_NAMES, INPUT_DIM, NUM_CLASSES, IMG_H, IMG_W, IMG_C
from model       import MLP


# ── 1. Training curves ─────────────────────────────────────────────────────────

def plot_training_curves(history_path: str = "outputs/best_model_history.npz",
                         save_path:    str = "outputs/training_curves.png"):
    """
    Plot train loss, val loss, and val accuracy from a saved history .npz.
    """
    h = np.load(history_path)
    epochs = np.arange(1, len(h["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ── Loss curves ────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, h["train_loss"], label="Train Loss",  color="steelblue")
    ax.plot(epochs, h["val_loss"],   label="Val Loss",    color="tomato",  linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Accuracy curve ─────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(epochs, h["val_acc"] * 100, label="Val Accuracy", color="seagreen")
    best_epoch = int(np.argmax(h["val_acc"])) + 1
    best_acc   = float(h["val_acc"].max()) * 100
    ax.axvline(best_epoch, color="orange", linestyle=":", alpha=0.8,
               label=f"Best epoch {best_epoch} ({best_acc:.2f}%)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Validation Accuracy Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Visualize] Training curves saved → {save_path}")


# ── 2a. 第一层隐藏神经元权重图（通用特征探测器）─────────────────────────────

def plot_weight_images(model:     MLP,
                       n_images:  int = 20,
                       save_path: str = "outputs/weight_images.png"):
    """
    将第一层权重矩阵 W1（形状 12288×hidden1）的每一列还原为 64×64×3 图像显示。

    每张图 = 第一层某个隐藏神经元的"偏好输入模式"：
      什么像素组合最能激活它（与类别标签无关）。

    按 L2 范数排序（范数越大 = 该神经元权重越强 = 对输入越敏感），
    取前 n_images 个展示。n_images 默认 20，与 10 个类别无关。
    """
    W1 = model.W1.T    # (hidden1, 12288)，每行是一个神经元的权重向量

    norms = np.linalg.norm(W1, axis=1)
    order = np.argsort(norms)[::-1]
    W1    = W1[order[:n_images]]

    imgs = []
    for w in W1:
        img = w.reshape(IMG_H, IMG_W, IMG_C)
        lo, hi = img.min(), img.max()
        imgs.append((img - lo) / (hi - lo + 1e-8))

    n_cols = min(10, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 1.6, n_rows * 1.9))
    axes = np.array(axes).reshape(n_rows, n_cols)

    for idx in range(n_rows * n_cols):
        ax = axes[idx // n_cols, idx % n_cols]
        if idx < len(imgs):
            ax.imshow(imgs[idx])
            ax.set_title(f"neuron #{order[idx]}", fontsize=5.5)
        ax.axis("off")

    fig.suptitle(
        f"Layer-1 Hidden Neuron Weights  "
        f"(top-{n_images} by L2 norm, out of {model.hidden1} neurons)\n"
        "Each image = preferred input pattern of one neuron  "
        "(class-agnostic low-level feature detectors)",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Visualize] Weight images saved → {save_path}")


# ── 2b. 10类专属"有效权重图"（端到端：输入→各类别）────────────────────────

def plot_class_weight_images(model:     MLP,
                             save_path: str = "outputs/class_weight_images.png"):
    """
    计算从输入层到每个输出类别的"有效权重"，生成 10 张类别专属图像。

    数学原理（线性近似，忽略 ReLU）：
      effective_weight_c = W1 @ W2 @ W3[:, c]
      形状：(12288, 512) @ (512, 256) @ (256,) = (12288,)

    这个向量回答的是：
      "如果不考虑非线性激活，网络认为第 c 类的'理想输入'长什么样？"

    正值（暖色）→ 该像素位置出现此颜色时，网络倾向于判断为这一类；
    负值（冷色）→ 该像素出现此颜色时，网络倾向于排除这一类。

    注意：这是三层线性权重的折叠，ReLU 的非线性被忽略，
    因此图像是一种近似解释而非精确因果。
    """
    # W1: (12288, h1)  W2: (h1, h2)  W3: (h2, 10)
    # effective: (12288, 10)，每列对应一个类别
    effective = model.W1 @ model.W2 @ model.W3   # (12288, 10)

    fig, axes = plt.subplots(2, 5, figsize=(13, 5.5))
    axes = axes.reshape(2, 5)

    for c, class_name in enumerate(CLASS_NAMES):
        ax  = axes[c // 5, c % 5]
        w   = effective[:, c]                        # (12288,)
        img = w.reshape(IMG_H, IMG_W, IMG_C)

        # 对称归一化：以 0 为中心，正值暖色、负值冷色
        abs_max = np.abs(img).max() + 1e-8
        img_sym = (img / abs_max + 1.0) / 2.0       # 映射到 [0, 1]，0.5 = 中性

        ax.imshow(img_sym, vmin=0, vmax=1)
        ax.set_title(f"{class_name}", fontsize=8, fontweight="bold")
        ax.axis("off")

    fig.suptitle(
        "Per-Class Effective Weights:  effective_weight[c] = W1 @ W2 @ W3[:, c]\n"
        "Warm = pixels that push the network toward this class  |  "
        "Cool = pixels that push away  (linear approximation, ReLU ignored)",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()
    print(f"[Visualize] Class weight images saved → {save_path}")


# ── 3. Error analysis ──────────────────────────────────────────────────────────

def plot_error_examples(model:      MLP,
                        data:       dict,
                        n_images:   int   = 12,
                        save_path:  str   = "outputs/error_examples.png",
                        batch_size: int   = 512):
    """
    Find misclassified test images and display them with true / predicted labels.

    Images are loaded directly from the original files (stored in data["test_paths"]),
    completely bypassing denormalization – guarantees correct colors regardless of
    how the normalization was configured.
    """
    X_test     = data["X_test"]
    y_test     = data["y_test"]
    test_paths = data.get("test_paths", None)   # may be absent in old datasets

    # ── run inference ──────────────────────────────────────────────────────
    all_preds = []
    for start in range(0, len(X_test), batch_size):
        Xb    = X_test[start : start + batch_size]
        preds = model.predict(Xb)
        all_preds.extend(preds.tolist())
    y_pred = np.array(all_preds)

    # ── find misclassified indices ─────────────────────────────────────────
    wrong_idx = np.where(y_pred != y_test)[0]
    print(f"[Visualize] Total misclassified: {len(wrong_idx)}/{len(y_test)} "
          f"({len(wrong_idx)/len(y_test)*100:.1f}%)")

    if len(wrong_idx) == 0:
        print("[Visualize] No errors found – perfect classifier!")
        return

    # ── select diverse examples (one per unique true→predicted pair) ───────
    rng         = np.random.default_rng(42)
    chosen      = []
    shown_pairs = set()
    for idx in wrong_idx:
        pair = (int(y_test[idx]), int(y_pred[idx]))
        if pair not in shown_pairs:
            chosen.append(int(idx))
            shown_pairs.add(pair)
        if len(chosen) >= n_images:
            break
    # Fill remaining slots randomly if not enough diverse pairs
    if len(chosen) < n_images:
        chosen_set = set(chosen)
        remaining  = [int(i) for i in wrong_idx if int(i) not in chosen_set]
        if remaining:
            extra = rng.choice(remaining,
                               size=min(n_images - len(chosen), len(remaining)),
                               replace=False)
            chosen.extend(extra.tolist())
    chosen = chosen[:n_images]

    # ── load images from disk (best quality) or fall back to denormalise ───
    def _load_image(data_idx: int) -> np.ndarray:
        """Return a float32 (H, W, 3) array in [0, 1]."""
        if test_paths is not None:
            path = str(test_paths[data_idx])
            img  = Image.open(path).convert("RGB")
            if img.size != (IMG_W, IMG_H):
                img = img.resize((IMG_W, IMG_H), Image.BILINEAR)
            return np.asarray(img, dtype=np.float32) / 255.0
        else:
            # Fallback: per-image min-max normalisation for display
            # (avoids the per-pixel-std artefacts of full denormalisation)
            raw = X_test[data_idx]                         # normalised vector
            img = raw.reshape(IMG_H, IMG_W, IMG_C).copy()
            lo, hi = img.min(), img.max()
            img = (img - lo) / (hi - lo + 1e-8)
            return np.clip(img, 0.0, 1.0)

    # ── build figure ───────────────────────────────────────────────────────
    n_cols = min(4, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.5, n_rows * 4.0))
    # Ensure axes is always a 2-D array
    axes = np.array(axes).reshape(n_rows, n_cols)

    for plot_i, data_i in enumerate(chosen):
        ax        = axes[plot_i // n_cols, plot_i % n_cols]
        img       = _load_image(data_i)
        true_name = CLASS_NAMES[y_test[data_i]]
        pred_name = CLASS_NAMES[y_pred[data_i]]

        # Softmax probabilities for the top-2 classes
        probs     = model.forward(X_test[data_i : data_i + 1])[0]
        top2_idx  = np.argsort(probs)[::-1][:2]
        prob_str  = "  ".join(f"{CLASS_NAMES[j][:6]}:{probs[j]*100:.0f}%"
                              for j in top2_idx)

        ax.imshow(img, interpolation="nearest")
        ax.set_title(
            f"True: {true_name}\nPredicted: {pred_name}\n{prob_str}",
            fontsize=7.5,
            color="crimson",
            pad=3,
        )
        ax.axis("off")

    # Hide unused subplot slots
    for k in range(len(chosen), n_rows * n_cols):
        axes[k // n_cols, k % n_cols].axis("off")

    fig.suptitle(
        f"Error Analysis (Test Set Misclassification Rate {len(wrong_idx)/len(y_test)*100:.1f}%)",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[Visualize] Error examples saved → {save_path}")


# ── 4. Hyperparameter search results ──────────────────────────────────────────

def plot_search_results(csv_path:  str = "outputs/search_results.csv",
                        save_path: str = "outputs/search_results.png"):
    """Bar chart of val_acc for all hyperparameter trials."""
    try:
        import csv
        rows = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        if not rows:
            return

        trials  = [int(r["trial"])      for r in rows]
        val_acc = [float(r["val_acc"])  for r in rows]
        labels  = [f"lr={r['lr']}\nh={r['hidden1']}×{r['hidden2']}\n"
                   f"wd={r['weight_decay']}\n{r['activation']}" for r in rows]

        fig, ax = plt.subplots(figsize=(max(12, len(trials) * 0.8), 5))
        colors = ["#4CAF50" if v == max(val_acc) else "#90CAF9" for v in val_acc]
        ax.bar(trials, val_acc, color=colors)
        ax.set_xticks(trials)
        ax.set_xticklabels([str(t) for t in trials], fontsize=7)
        ax.set_xlabel("Trial")
        ax.set_ylabel("Val Accuracy (%)")
        ax.set_title("Hyperparameter Search Results")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[Visualize] Search results plot saved → {save_path}")
    except Exception as e:
        print(f"[Visualize] Could not plot search results: {e}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Visualise EuroSAT MLP results")
    p.add_argument("--data_dir",      default="EuroSAT_RGB")
    p.add_argument("--weights",       default="outputs/best_model.npz")
    p.add_argument("--history",       default="outputs/best_model_history.npz")
    p.add_argument("--hidden1",       type=int, default=512)
    p.add_argument("--hidden2",       type=int, default=256)
    p.add_argument("--activation",    default="relu",
                   choices=["relu", "sigmoid", "tanh"])
    p.add_argument("--n_weight_imgs", type=int, default=20)
    p.add_argument("--n_error_imgs",  type=int, default=12)
    p.add_argument("--output_dir",    default="outputs")
    p.add_argument("--seed",          type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ── ensure output directory exists ────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    def out(fname):
        return os.path.join(args.output_dir, fname)

    # ── training curves ────────────────────────────────────────────────────
    if os.path.exists(args.history):
        plot_training_curves(args.history, save_path=out("training_curves.png"))
    else:
        print(f"[Visualize] History file not found: {args.history}")

    # ── weight images ──────────────────────────────────────────────────────
    model = MLP(input_dim=INPUT_DIM,
                hidden1=args.hidden1,
                hidden2=args.hidden2,
                num_classes=NUM_CLASSES,
                activation=args.activation,
                seed=args.seed)

    if os.path.exists(args.weights):
        model.load_weights(args.weights)
        plot_weight_images(model, n_images=args.n_weight_imgs,
                           save_path=out("weight_images.png"))
        plot_class_weight_images(model, save_path=out("class_weight_images.png"))
    else:
        print(f"[Visualize] Weights file not found: {args.weights}")
        print("            Skipping weight images & error examples.")

    # ── error analysis ─────────────────────────────────────────────────────
    if os.path.exists(args.weights):
        data = load_dataset(args.data_dir, seed=args.seed)
        plot_error_examples(model, data, n_images=args.n_error_imgs,
                            save_path=out("error_examples.png"))

    # ── search results ─────────────────────────────────────────────────────
    csv_path = out("search_results.csv")
    if os.path.exists(csv_path):
        plot_search_results(csv_path=csv_path,
                            save_path=out("search_results.png"))
