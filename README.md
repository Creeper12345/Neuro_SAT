# HW1：从零构建三层神经网络实现 EuroSAT 地表覆盖分类

本项目手工实现了一个三层多层感知机（MLP）分类器，**完全基于 NumPy**（不使用 PyTorch、TensorFlow、JAX 等自动微分框架），在 EuroSAT 遥感图像数据集上完成 10 类土地覆盖分类任务。

---

## 数据集

- **EuroSAT RGB**：27,000 张 64×64 彩色卫星图像，共 10 个类别：
  `AnnualCrop`、`Forest`、`HerbaceousVegetation`、`Highway`、`Industrial`、`Pasture`、`PermanentCrop`、`Residential`、`River`、`SeaLake`
- 数据集划分：训练集 70% / 验证集 15% / 测试集 15%（分层抽样）
- 数据集请自行下载并放置于项目根目录下的 `EuroSAT_RGB/` 文件夹中（该文件夹不纳入版本控制）

---

## 项目结构

```
.
├── data_loader.py         # 数据加载、预处理、mini-batch 迭代器
├── model.py               # MLP 模型定义 + 手动反向传播
├── optimizer.py           # SGD 优化器、学习率衰减、交叉熵损失
├── train.py               # 训练主循环，自动保存最优权重
├── evaluate.py            # 测试集准确率 + 混淆矩阵
├── hyperparam_search.py   # 网格搜索 / 随机搜索超参数
├── visualize.py           # 训练曲线、权重可视化、错例分析
├── pack_outputs.py        # 将 outputs/ 目录打包为 zip
├── requirements.txt       # 依赖包列表
├── README.md
└── outputs/               # 所有生成文件均保存在此目录（自动创建）
    ├── best_model.npz
    ├── best_model_history.npz
    ├── confusion_matrix.png / .npy
    ├── search_results.csv / .png
    ├── training_curves.png
    ├── weight_images.png
    └── error_examples.png
```

---

## 模型架构

```
输入层  (12 288)  ← 64×64×3 展平
  ↓  全连接 + 激活（ReLU / Sigmoid / Tanh）
隐藏层1 (512)
  ↓  全连接 + 激活（ReLU / Sigmoid / Tanh）
隐藏层2 (256)
  ↓  全连接
输出层  (10)   ← Softmax → 交叉熵损失
```

- 权重初始化：Xavier / Glorot 均匀分布
- 优化器：带动量的 SGD（momentum=0.9）
- 正则化：L2 权重衰减（Weight Decay）
- 学习率调度：指数衰减（每 epoch 乘以衰减因子）
- 模型保存：验证集准确率最高时自动保存权重

---

## 环境依赖

```bash
pip install -r requirements.txt
```

依赖项：
- `numpy >= 1.24.0`
- `Pillow >= 9.0.0`
- `matplotlib >= 3.6.0`

> 本项目**不依赖**任何深度学习框架（PyTorch / TensorFlow / JAX），所有前向传播与反向传播均手动实现。

---

## 运行方法

### 1. 训练模型

```bash
python train.py \
  --data_dir EuroSAT_RGB \
  --epochs 50 \
  --batch_size 256 \
  --lr 1e-3 \
  --hidden1 512 \
  --hidden2 256 \
  --weight_decay 1e-4 \
  --lr_decay 0.95 \
  --activation relu \
  --save_path outputs/best_model.npz
```

训练完成后，最优权重将保存至 `outputs/best_model.npz`，训练历史保存至 `outputs/best_model_history.npz`（`outputs/` 目录若不存在会自动创建）。

**主要参数说明：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data_dir` | EuroSAT_RGB 数据集路径 | `EuroSAT_RGB` |
| `--epochs` | 训练轮数 | `50` |
| `--batch_size` | mini-batch 大小 | `256` |
| `--lr` | 初始学习率 | `1e-3` |
| `--hidden1` | 第一隐藏层神经元数 | `512` |
| `--hidden2` | 第二隐藏层神经元数 | `256` |
| `--weight_decay` | L2 正则化强度 λ | `1e-4` |
| `--lr_decay` | 每 epoch 学习率衰减因子 | `0.95` |
| `--activation` | 激活函数：`relu` / `sigmoid` / `tanh` | `relu` |
| `--save_path` | 最优模型权重保存路径 | `outputs/best_model.npz` |

---

### 2. 测试评估

```bash
python evaluate.py \
  --data_dir EuroSAT_RGB \
  --weights outputs/best_model.npz \
  --hidden1 512 \
  --hidden2 256 \
  --activation relu
```

输出（均保存至 `outputs/`）：
- 总体分类准确率（Accuracy）
- 各类别准确率
- 10×10 混淆矩阵（打印 + 保存为 `outputs/confusion_matrix.png` 和 `.npy`）

---

### 3. 超参数搜索

```bash
# 网格搜索（穷举所有组合）
python hyperparam_search.py \
  --data_dir EuroSAT_RGB \
  --mode grid \
  --epochs 20

# 随机搜索（指定试验次数）
python hyperparam_search.py \
  --data_dir EuroSAT_RGB \
  --mode random \
  --n_trials 20 \
  --epochs 20
```

结果保存至 `outputs/search_results.csv`，包含每组超参数的验证集准确率。

**搜索的超参数范围：**
- 学习率：`[1e-2, 1e-3, 5e-4]`
- 隐藏层大小：`hidden1 ∈ {256, 512}`，`hidden2 ∈ {128, 256}`
- L2 正则化：`[1e-3, 1e-4, 1e-5]`
- 激活函数：`relu`、`tanh`、`sigmoid`

---

### 4. 可视化

```bash
python visualize.py \
  --data_dir EuroSAT_RGB \
  --weights outputs/best_model.npz \
  --history outputs/best_model_history.npz \
  --n_weight_imgs 20 \
  --n_error_imgs 12
```

生成（均保存至 `outputs/`）：
- `training_curves.png`：训练集/验证集 Loss 曲线 + 验证集 Accuracy 曲线
- `weight_images.png`：第一层权重恢复为 64×64×3 图像的可视化
- `error_examples.png`：测试集错误分类样例（附真实标签与预测标签，中文支持）
- `search_results.png`：超参数搜索结果对比图（如已运行搜索）

---

## 模型权重下载

训练好的最优模型权重已上传至 Google Drive：

**[点击下载 best_model.npz](待填写)**

> 下载后将 `best_model.npz` 放置于 `outputs/` 目录，即可直接运行 `evaluate.py` 和 `visualize.py`。

---

## GitHub 仓库

[https://github.com/Creeper12345/NEuro_SAT](https://github.com/Creeper12345/NEuro_SAT)

---

## 注意事项

- 本项目为个人作业，所有代码均为独立实现
- 反向传播梯度已通过数值梯度检验（误差 < 1e-10）验证正确性
- 由于使用纯 NumPy 实现，建议在多核 CPU 服务器上运行以加快矩阵运算速度
