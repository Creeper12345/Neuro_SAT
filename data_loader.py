"""
data_loader.py
--------------
EuroSAT RGB 数据集的加载与预处理模块。
提供 load_dataset() 函数（按 70/15/15 比例分层划分数据集）
和 DataLoader 迭代器类（用于 mini-batch 训练）。
"""

import os
import numpy as np
from PIL import Image

CLASS_NAMES = [
    "AnnualCrop",          # 0
    "Forest",              # 1
    "HerbaceousVegetation",# 2
    "Highway",             # 3
    "Industrial",          # 4
    "Pasture",             # 5
    "PermanentCrop",       # 6
    "Residential",         # 7
    "River",               # 8
    "SeaLake",             # 9
]
NUM_CLASSES = len(CLASS_NAMES)

# EuroSAT images are 64 × 64 × 3
IMG_H, IMG_W, IMG_C = 64, 64, 3
INPUT_DIM = IMG_H * IMG_W * IMG_C   # 12,288


def _load_single(path: str) -> np.ndarray:
    """加载单张图片，返回 float32 格式的一维数组，形状 (12288,)。"""
    img = Image.open(path).convert("RGB")
    if img.size != (IMG_W, IMG_H):
        img = img.resize((IMG_W, IMG_H), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)   # (64, 64, 3)
    return arr.reshape(-1)                     # (12288,)


def load_dataset(data_dir: str, seed: int = 42):
    """
    Split ratios: 70 % train · 15 % val · 15 % test
    """
    rng = np.random.default_rng(seed)

    paths, labels = [], []
    for label_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Class folder not found: {class_dir}")
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".tif")):
                paths.append(os.path.join(class_dir, fname))
                labels.append(label_idx)

    paths  = np.array(paths)
    labels = np.array(labels, dtype=np.int32)
    n      = len(paths)
    print(f"[DataLoader] Found {n} images across {NUM_CLASSES} classes.")

    train_idx, val_idx, test_idx = [], [], []
    for c in range(NUM_CLASSES):
        c_idx = np.where(labels == c)[0]
        rng.shuffle(c_idx)
        n_c   = len(c_idx)
        n_tr  = int(n_c * 0.70)
        n_val = int(n_c * 0.15)
        train_idx.extend(c_idx[:n_tr])
        val_idx.extend(  c_idx[n_tr : n_tr + n_val])
        test_idx.extend( c_idx[n_tr + n_val :])

    # shuffle each split
    for idx_list in [train_idx, val_idx, test_idx]:
        rng.shuffle(idx_list)

    train_idx = np.array(train_idx)
    val_idx   = np.array(val_idx)
    test_idx  = np.array(test_idx)

    print(f"[DataLoader] Split  →  train: {len(train_idx)} | "
          f"val: {len(val_idx)} | test: {len(test_idx)}")

    def _load_split(idx_arr):
        X = np.stack([_load_single(paths[i]) for i in idx_arr])   # (N, 12288)
        y = labels[idx_arr]
        return X, y

    print("[DataLoader] Loading training images …")
    X_train, y_train = _load_split(train_idx)
    print("[DataLoader] Loading validation images …")
    X_val,   y_val   = _load_split(val_idx)
    print("[DataLoader] Loading test images …")
    X_test,  y_test  = _load_split(test_idx)

    # normalize
    mean = X_train.mean(axis=0, keepdims=True)        # (1, 12288)
    std  = X_train.std( axis=0, keepdims=True) + 1e-8

    X_train = (X_train - mean) / std
    X_val   = (X_val   - mean) / std
    X_test  = (X_test  - mean) / std

    print("[DataLoader] Normalisation done  (mean/std computed on train set).")

    test_paths = paths[test_idx]   # original image file paths, same order as X_test

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
        "mean": mean, "std": std,
        "test_paths": test_paths,  
    }


class DataLoader:
    """Iterates over (X, y) in random mini-batches."""

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 batch_size: int = 256, shuffle: bool = True,
                 seed: int = 0):
        self.X          = X
        self.y          = y
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.rng        = np.random.default_rng(seed)
        self.n          = len(X)

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        idx = np.arange(self.n)
        if self.shuffle:
            self.rng.shuffle(idx)
        for start in range(0, self.n, self.batch_size):
            batch_idx = idx[start : start + self.batch_size]
            yield self.X[batch_idx], self.y[batch_idx]


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "EuroSAT_RGB"
    data = load_dataset(data_dir)
    print("X_train shape:", data["X_train"].shape)
    print("y_train unique labels:", np.unique(data["y_train"]))

    loader = DataLoader(data["X_train"], data["y_train"], batch_size=64)
    for i, (xb, yb) in enumerate(loader):
        print(f"  batch {i}: x={xb.shape}  y={yb.shape}")
        if i == 1:
            break
