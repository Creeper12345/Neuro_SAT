"""
visualize_class_neurons.py
--------------------------
针对每个类别找出贡献最大的隐藏神经元，可视化其输入权重模式及通道偏好。

生成四种可视化图：
  1 class_mean_images.png      每类正确预测图像的像素均值
  2 class_key_neurons.png      激活×贡献系数双重加权的关键 H1 神经元权重
  3 class_weight_neurons.png   纯权重贡献系数排序（正/负对比）
  4 class_channels.png         有效权重 W1@W2@W3[:,c] 按 RGB 通道分解

用法：
    python visualize_class_neurons.py --data_dir EuroSAT_RGB \\
        --weights outputs/best_model.npz --hidden1 256 --hidden2 256 --activation relu
"""

import argparse
import os
import sys
import platform
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as _fm


from data_loader import load_dataset, CLASS_NAMES, INPUT_DIM, NUM_CLASSES, IMG_H, IMG_W, IMG_C
from model       import MLP

IMG_SHAPE = (IMG_H, IMG_W, IMG_C)   # (64, 64, 3)

CLASS_ZH = {
    "AnnualCrop":            "年作物 AnnualCrop",
    "Forest":                "森林 Forest",
    "HerbaceousVegetation":  "草本植被 HerbVeg",
    "Highway":               "公路 Highway",
    "Industrial":            "工业区 Industrial",
    "Pasture":               "牧场 Pasture",
    "PermanentCrop":         "永久作物 PermCrop",
    "Residential":           "居住区 Residential",
    "River":                 "河流 River",
    "SeaLake":               "海湖 SeaLake",
}


def weight_to_img(w: np.ndarray, signed: bool = True) -> np.ndarray:
    """12288维权重向量 → [0,1] 的 64×64×3 图像。
    signed=True：对称归一化（0.5=零，亮=正，暗=负）"""
    img = w.reshape(IMG_SHAPE).copy()
    if signed:
        absmax = np.abs(img).max() + 1e-8
        img = (img / absmax + 1.0) / 2.0
    else:
        lo, hi = img.min(), img.max()
        img = (img - lo) / (hi - lo + 1e-8)
    return np.clip(img, 0.0, 1.0)


def add_row_label(ax, text, fontsize=8):
    ax.annotate(
        text,
        xy=(0, 0.5), xycoords="axes fraction",
        xytext=(-8, 0), textcoords="offset points",
        ha="right", va="center",
        fontsize=fontsize, fontweight="bold",
        annotation_clip=False,
    )


def add_colorbar_legend(fig, ax_ref, label_pos="bottom"):
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import Normalize
    cax = fig.add_axes([0.92, 0.1, 0.015, 0.8])
    cb  = ColorbarBase(cax, cmap=plt.cm.RdBu_r,
                       norm=Normalize(vmin=-1, vmax=1),
                       orientation="vertical")
    cb.set_label("← 负权重（抑制）     正权重（激活） →", fontsize=7, rotation=90)
    cb.set_ticks([-1, 0, 1])
    cb.set_ticklabels(["抑制 −", "零 0", "激活 ＋"])


# ══════════════════════════════════════════════════════════════════════════════
# 1 类别均值图像
# ══════════════════════════════════════════════════════════════════════════════

def plot_class_mean_images(model: MLP, data: dict,
                           save_path: str = "outputs/class_mean_images.png"):
    X_test, y_test = data["X_test"], data["y_test"]
    test_paths     = data.get("test_paths", None)

    y_pred       = model.predict(X_test)
    correct_mask = (y_pred == y_test)

    # 每类 2 列：均值图 + 3张随机正例小图
    SAMPLE_COLS = 3
    N_COLS = 1 + SAMPLE_COLS
    fig, axes = plt.subplots(NUM_CLASSES, N_COLS,
                             figsize=(N_COLS * 2.4, NUM_CLASSES * 2.6))
    axes = np.array(axes).reshape(NUM_CLASSES, N_COLS)

    for c, cname in enumerate(CLASS_NAMES):
        mask = correct_mask & (y_test == c)
        idx  = np.where(mask)[0]
        acc_c = mask.sum() / max((y_test == c).sum(), 1) * 100

        if test_paths is not None and len(idx) > 0:
            def load_img(i):
                return np.asarray(
                    Image.open(str(test_paths[i])).convert("RGB")
                        .resize((IMG_W, IMG_H), Image.BILINEAR),
                    dtype=np.float32
                ) / 255.0

            sample_idx = idx[:min(200, len(idx))]
            imgs       = [load_img(i) for i in sample_idx]
            mean_img   = np.stack(imgs).mean(axis=0)
        else:
            mean_img = np.full(IMG_SHAPE, 0.5)

        # 左列：均值图
        ax0 = axes[c, 0]
        ax0.imshow(mean_img)
        ax0.set_title(f"Avg (acc={acc_c:.0f}%)", fontsize=7.5, pad=2)
        ax0.axis("off")

        label = CLASS_ZH.get(cname, cname)
        add_row_label(ax0, label, fontsize=8)

        # 右边 3 列：随机正例
        rng = np.random.default_rng(42)
        picks = rng.choice(idx, size=min(SAMPLE_COLS, len(idx)), replace=False) \
                if len(idx) >= SAMPLE_COLS else idx
        for col in range(SAMPLE_COLS):
            ax = axes[c, col + 1]
            if col < len(picks) and test_paths is not None:
                img = np.asarray(
                    Image.open(str(test_paths[picks[col]])).convert("RGB")
                        .resize((IMG_W, IMG_H), Image.BILINEAR),
                    dtype=np.float32
                ) / 255.0
                ax.imshow(img)
                ax.set_title("样例", fontsize=6.5, pad=1)
            else:
                ax.axis("off")
            ax.axis("off")

    # 列标题
    axes[0, 0].set_title(f"类别均值图像\n(acc=%)", fontsize=8, pad=3)
    for col in range(SAMPLE_COLS):
        axes[0, col + 1].set_title(f"正确预测样例 {col+1}", fontsize=8, pad=3)

    fig.suptitle(
        "1 类别视觉原型 — 正确预测测试图像的像素均值 & 随机样例\n"
        "Class Prototype: Mean of Correctly-Classified Test Samples",
        fontsize=11, y=1.005, fontweight="bold",
    )
    plt.tight_layout(rect=[0.09, 0, 1, 1])
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[1] class mean images → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 2 激活 × 贡献系数双重加权
# ══════════════════════════════════════════════════════════════════════════════

def plot_activation_weighted_neurons(model: MLP, data: dict,
                                     top_k: int = 5,
                                     save_path: str = "outputs/class_key_neurons.png"):
    X_test, y_test = data["X_test"], data["y_test"]

    # 前向传播收集 H1 激活
    print("[2] Collecting H1 activations …")
    h1_acts = np.zeros((len(X_test), model.hidden1), dtype=np.float32)
    for s in range(0, len(X_test), 512):
        model.forward(X_test[s:s+512])
        h1_acts[s:s+512] = model._cache["A1"].astype(np.float32)

    coeff_h1 = model.W2 @ model.W3   # (H1, 10)

    # 每行：行标签列 + top_k 正贡献列 + top_k 负贡献列
    N_COLS = 1 + top_k * 2
    fig, axes = plt.subplots(NUM_CLASSES, N_COLS,
                             figsize=(N_COLS * 1.9, NUM_CLASSES * 2.2))
    axes = np.array(axes).reshape(NUM_CLASSES, N_COLS)

    # 第 0 列：用来放行标签（不画图）
    for c in range(NUM_CLASSES):
        axes[c, 0].axis("off")

    for c, cname in enumerate(CLASS_NAMES):
        mask     = y_test == c
        mean_act = h1_acts[mask].mean(axis=0)       # (H1,)
        coeff_c  = coeff_h1[:, c]
        acc_c    = (model.predict(X_test)[mask] == y_test[mask]).mean() * 100

        # 正贡献 top-k：score = mean_activation × coeff
        pos_score = np.where(coeff_c > 0, mean_act * coeff_c, 0.0)
        pos_top   = np.argsort(pos_score)[::-1][:top_k]
        # 负贡献 top-k：|score| 最大的负向神经元
        neg_score = np.where(coeff_c < 0, mean_act * np.abs(coeff_c), 0.0)
        neg_top   = np.argsort(neg_score)[::-1][:top_k]

        label = CLASS_ZH.get(cname, cname)

        # 行标签列
        axes[c, 0].text(0.5, 0.5, f"{label}\nacc={acc_c:.0f}%",
                        ha="center", va="center", fontsize=8, fontweight="bold",
                        transform=axes[c, 0].transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", fc="#EEF4FF", ec="#99AACC", lw=0.8))

        for col, k in enumerate(pos_top):
            ax  = axes[c, col + 1]
            img = weight_to_img(model.W1[:, k], signed=True)
            ax.imshow(img, vmin=0, vmax=1)
            ax.set_title(f"N{k}\nact={mean_act[k]:.2f}\ncoef=+{coeff_c[k]:.3f}",
                         fontsize=5.5, color="crimson", pad=1)
            for sp in ax.spines.values():
                sp.set_edgecolor("crimson"); sp.set_linewidth(1.5)
            ax.axis("off")

        for col, k in enumerate(neg_top):
            ax  = axes[c, top_k + 1 + col]
            img = weight_to_img(model.W1[:, k], signed=True)
            ax.imshow(img, vmin=0, vmax=1)
            ax.set_title(f"N{k}\nact={mean_act[k]:.2f}\ncoef={coeff_c[k]:.3f}",
                         fontsize=5.5, color="steelblue", pad=1)
            for sp in ax.spines.values():
                sp.set_edgecolor("steelblue"); sp.set_linewidth(1.5)
            ax.axis("off")

    # 列标题行（第 0 行）
    axes[0, 0].set_title("类别\n(acc%)", fontsize=8, pad=2)
    for col in range(top_k):
        axes[0, col + 1].set_title(f"正贡献 #{col+1}\n(红框)", fontsize=7, color="crimson", pad=2)
    for col in range(top_k):
        axes[0, top_k + 1 + col].set_title(f"负贡献 #{col+1}\n(蓝框)", fontsize=7, color="steelblue", pad=2)

    # 中间分隔线
    mid_x = (top_k + 1) / N_COLS
    fig.add_artist(plt.Line2D([mid_x, mid_x], [0, 1],
                              transform=fig.transFigure,
                              color="gray", lw=0.8, linestyle="--"))

    legend_patches = [
        mpatches.Patch(color="crimson",   label="正贡献：激活→推向该类  score = act × coeff (coeff>0)"),
        mpatches.Patch(color="steelblue", label="负贡献：激活→抑制该类  score = act × |coeff| (coeff<0)"),
        mpatches.Patch(color="#cccccc",   label="图像像素：亮=正权重(兴奋)，暗=负权重(抑制)，中灰=接近零"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=1,
               fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.03))

    fig.suptitle(
        "2 激活×贡献系数双重加权关键 H1 神经元权重可视化\n"
        "score = mean_activation[k] × coeff[k,c]   |   coeff = (W2 @ W3[:,c])[k]",
        fontsize=10, y=1.002, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[2] activation-weighted neuron vis → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 3 纯权重贡献（正/负对比，不依赖数据）
# ══════════════════════════════════════════════════════════════════════════════

def plot_weight_only_neurons(model: MLP,
                             top_k: int = 5,
                             save_path: str = "outputs/class_weight_neurons.png"):
    coeff_h1 = model.W2 @ model.W3   # (H1, 10)

    N_COLS = 1 + top_k * 2
    fig, axes = plt.subplots(NUM_CLASSES, N_COLS,
                             figsize=(N_COLS * 1.9, NUM_CLASSES * 2.2))
    axes = np.array(axes).reshape(NUM_CLASSES, N_COLS)

    for c in range(NUM_CLASSES):
        axes[c, 0].axis("off")

    for c, cname in enumerate(CLASS_NAMES):
        coeff_c = coeff_h1[:, c]
        pos_top = np.argsort(coeff_c)[::-1][:top_k]
        neg_top = np.argsort(coeff_c)[:top_k]
        label   = CLASS_ZH.get(cname, cname)

        axes[c, 0].text(0.5, 0.5, label,
                        ha="center", va="center", fontsize=8, fontweight="bold",
                        transform=axes[c, 0].transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", fc="#EEF4FF", ec="#99AACC", lw=0.8))

        for col, k in enumerate(pos_top):
            ax  = axes[c, col + 1]
            img = weight_to_img(model.W1[:, k], signed=True)
            ax.imshow(img, vmin=0, vmax=1)
            ax.set_title(f"N{k}\n+{coeff_c[k]:.3f}",
                         fontsize=6, color="crimson", pad=1)
            for sp in ax.spines.values():
                sp.set_edgecolor("crimson"); sp.set_linewidth(1.5)
            ax.axis("off")

        for col, k in enumerate(neg_top):
            ax  = axes[c, top_k + 1 + col]
            img = weight_to_img(model.W1[:, k], signed=True)
            ax.imshow(img, vmin=0, vmax=1)
            ax.set_title(f"N{k}\n{coeff_c[k]:.3f}",
                         fontsize=6, color="steelblue", pad=1)
            for sp in ax.spines.values():
                sp.set_edgecolor("steelblue"); sp.set_linewidth(1.5)
            ax.axis("off")

    axes[0, 0].set_title("类别", fontsize=8, pad=2)
    for col in range(top_k):
        axes[0, col + 1].set_title(f"正贡献 Top{col+1}", fontsize=7, color="crimson", pad=2)
    for col in range(top_k):
        axes[0, top_k + 1 + col].set_title(f"负贡献 Top{col+1}", fontsize=7, color="steelblue", pad=2)

    mid_x = (top_k + 1) / N_COLS
    fig.add_artist(plt.Line2D([mid_x, mid_x], [0, 1],
                              transform=fig.transFigure,
                              color="gray", lw=0.8, linestyle="--"))

    legend_patches = [
        mpatches.Patch(color="crimson",   label="正贡献神经元（红框）：该神经元被激活时 → 提高该类概率"),
        mpatches.Patch(color="steelblue", label="负贡献神经元（蓝框）：该神经元被激活时 → 降低该类概率"),
        mpatches.Patch(color="#f5e0e0",   label="图像亮部 = 权重正值区域（若输入匹配则激活该神经元）"),
        mpatches.Patch(color="#d0e0f0",   label="图像暗部 = 权重负值区域（输入匹配则抑制该神经元）"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=2,
               fontsize=7.5, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        "3 纯权重贡献系数关键 H1 神经元（不依赖数据）\n"
        "coeff[k, c] = (W2 @ W3[:, c])[k]   —   该神经元对类别 c 的线性贡献强度",
        fontsize=10, y=1.002, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[3] weight-only neuron vis → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 4 有效权重分 RGB 通道（颜色偏好分析）
# ══════════════════════════════════════════════════════════════════════════════

def plot_class_effective_channels(model: MLP,
                                  save_path: str = "outputs/class_channels.png"):
    effective = model.W1 @ model.W2 @ model.W3   # (12288, 10)

    # 每类：RGB合并 | R通道 | G通道 | B通道 | 通道均值柱状图
    N_COLS = 5
    fig, axes = plt.subplots(NUM_CLASSES, N_COLS,
                             figsize=(N_COLS * 2.5, NUM_CLASSES * 2.4))
    axes = np.array(axes).reshape(NUM_CLASSES, N_COLS)

    cmaps = [plt.cm.RdGy_r, plt.cm.RdYlGn, plt.cm.RdBu]
    ch_names = ["R", "G", "B"]
    ch_colors = ["tomato", "seagreen", "steelblue"]

    for c, cname in enumerate(CLASS_NAMES):
        w   = effective[:, c]
        img = w.reshape(IMG_SHAPE)
        absmax = np.abs(img).max() + 1e-8
        rgb_show = np.clip((img / absmax + 1.0) / 2.0, 0, 1)
        label = CLASS_ZH.get(cname, cname)

        # 列 0：RGB 合并
        ax0 = axes[c, 0]
        ax0.imshow(rgb_show)
        ax0.set_title("RGB 合并", fontsize=7.5, pad=2)
        ax0.axis("off")
        add_row_label(ax0, label, fontsize=8)

        # 列 1-3：单通道
        for ch, (cmap, ch_name) in enumerate(zip(cmaps, ch_names)):
            ch_data  = img[:, :, ch]
            am_ch    = np.abs(ch_data).max() + 1e-8
            ax       = axes[c, ch + 1]
            im       = ax.imshow(ch_data, cmap=cmap, vmin=-am_ch, vmax=am_ch)
            mean_val = ch_data.mean()
            sign_str = f"+{mean_val:.4f}" if mean_val >= 0 else f"{mean_val:.4f}"
            ax.set_title(f"{ch_name} 通道\n均值 {sign_str}", fontsize=7, pad=2)
            ax.axis("off")

        # 列 4：三通道均值柱状图（颜色偏好摘要）
        ax_bar = axes[c, 4]
        ch_means = [img[:, :, ch].mean() for ch in range(3)]
        bars = ax_bar.barh(ch_names, ch_means,
                           color=[c if v >= 0 else "#aaaaaa"
                                  for c, v in zip(ch_colors, ch_means)],
                           edgecolor="white", height=0.5)
        ax_bar.axvline(0, color="black", lw=0.8)
        ax_bar.set_xlim(-max(0.001, max(abs(v) for v in ch_means)) * 1.3,
                         max(0.001, max(abs(v) for v in ch_means)) * 1.3)
        ax_bar.set_title("通道偏好\n（正=兴奋）", fontsize=7, pad=2)
        ax_bar.tick_params(labelsize=6.5)
        ax_bar.set_xlabel("权重均值", fontsize=6)
        for spine in ["top", "right"]:
            ax_bar.spines[spine].set_visible(False)

    # 列标题
    col_titles = ["RGB 合并\n（对称归一化）",
                  "R 通道\n（红色=正权重）",
                  "G 通道\n（绿色=正权重）",
                  "B 通道\n（蓝色=正权重）",
                  "通道均值\n（颜色偏好摘要）"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=8, pad=3, fontweight="bold")

    # 图例
    legend_patches = [
        mpatches.Patch(color="tomato",    label="R 通道正权重：输入红色像素 → 激活"),
        mpatches.Patch(color="seagreen",  label="G 通道正权重：输入绿色像素 → 激活"),
        mpatches.Patch(color="steelblue", label="B 通道正权重：输入蓝色像素 → 激活"),
        mpatches.Patch(color="#aaaaaa",   label="负权重（灰色）：对应颜色出现时 → 抑制该类"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "4 类别有效权重分 RGB 通道解析    effective[c] = W1 @ W2 @ W3[:, c]\n"
        "正值(暖色) = 该颜色像素激活该类，负值(冷色) = 该颜色像素抑制该类\n"
        "柱状图摘要可直观判断每类对 RGB 三通道的偏好方向",
        fontsize=10, y=1.002, fontweight="bold",
    )
    plt.tight_layout(rect=[0.08, 0.04, 1, 1])
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[4] class channel analysis → {save_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="类别神经元可视化")
    p.add_argument("--data_dir",   default="EuroSAT_RGB")
    p.add_argument("--weights",    default="outputs/best_model_1.npz")
    p.add_argument("--hidden1",    type=int,   default=256)
    p.add_argument("--hidden2",    type=int,   default=256)
    p.add_argument("--activation", default="relu")
    p.add_argument("--top_k",      type=int,   default=5)
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    def out(fname):
        return os.path.join(args.output_dir, fname)

    # 加载模型
    model = MLP(input_dim=INPUT_DIM, hidden1=args.hidden1, hidden2=args.hidden2,
                num_classes=NUM_CLASSES, activation=args.activation, seed=args.seed)
    model.load_weights(args.weights)
    print(f"[ClassNeuron] Model loaded ← {args.weights}\n")

    # 加载数据
    data = load_dataset(args.data_dir, seed=args.seed)
    print()

    # 1 类别均值图像
    plot_class_mean_images(model, data, save_path=out("class_mean_images.png"))

    # 2 激活×贡献双重加权
    plot_activation_weighted_neurons(model, data,
                                     top_k=args.top_k,
                                     save_path=out("class_key_neurons.png"))

    # 3 纯权重贡献
    plot_weight_only_neurons(model, top_k=args.top_k,
                             save_path=out("class_weight_neurons.png"))

    # 4 有效权重分通道
    plot_class_effective_channels(model, save_path=out("class_channels.png"))

    print("\n[ClassNeuron] All done. Files saved to:", args.output_dir)
