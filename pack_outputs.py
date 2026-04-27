"""
pack_outputs.py
---------------
将 outputs/ 目录下的实验结果文件打包为 zip 压缩包，便于提交。

用法：
    python pack_outputs.py [--output_dir outputs] [--zip_name hw1_outputs.zip]
"""

import argparse
import glob
import os
import zipfile
from datetime import datetime


FIXED_FILES = [
    # 可视化图像
    "training_curves.png",
    "weight_images.png",
    "error_examples.png",
    "confusion_matrix.png",
    "search_results.png",
    # 数据文件
    "confusion_matrix.npy",
    "search_results.csv",
]

# glob 模式匹配
GLOB_PATTERNS = [
    "best_model*.npz",
    "*_history.npz",
]


def collect_files(output_dir: str) -> list:
    """收集 output_dir 下所有需要打包的文件，返回存在的文件路径列表。"""
    found, missing = [], []

    # 固定文件名
    for fname in FIXED_FILES:
        path = os.path.join(output_dir, fname)
        if os.path.isfile(path):
            found.append(path)
        else:
            missing.append(fname)

    # glob 模式
    for pattern in GLOB_PATTERNS:
        matches = sorted(glob.glob(os.path.join(output_dir, pattern)))
        if matches:
            found.extend(matches)
        else:
            missing.append(pattern)

    # 去重
    seen, unique = set(), []
    for p in found:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    return unique, missing


def pack(output_dir: str = "outputs", zip_name: str = None):

    if not os.path.isdir(output_dir):
        print(f"[Pack] 错误：目录不存在 → {output_dir}")
        return

    # 自动生成带时间戳的文件名
    if zip_name is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_name = f"hw1_outputs_{ts}.zip"

    files, missing = collect_files(output_dir)

    if not files:
        print(f"[Pack] 在 {output_dir} 中未找到任何可打包文件，退出。")
        return

    zip_path = os.path.join(output_dir, zip_name)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fpath in files:
            arcname = os.path.basename(fpath) 
            zf.write(fpath, arcname)
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  + {arcname:<40} ({size_kb:>8.1f} KB)")

    zip_size_mb = os.path.getsize(zip_path) / 1024 / 1024
    print()
    print(f"[Pack] ✓ 打包完成 → {zip_path}  ({zip_size_mb:.2f} MB)")
    print(f"[Pack]   共打包 {len(files)} 个文件")

    if missing:
        print()
        print("[Pack] 以下文件不存在，已跳过：")
        for m in missing:
            print(f"  - {m}")

    return zip_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="打包 EuroSAT MLP 实验可视化输出")
    p.add_argument("--output_dir", default="outputs",
                   help="存放生成文件的目录（默认: outputs）")
    p.add_argument("--zip_name",   default=None,
                   help="输出 zip 文件名（默认: hw1_outputs_<时间戳>.zip）")
    args = p.parse_args()

    pack(output_dir=args.output_dir, zip_name=args.zip_name)
