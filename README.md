# EuroSAT 遥感图像分类 — 三层 MLP（纯 NumPy 实现）

GitHub 仓库：[https://github.com/Creeper12345/NEuro_SAT](https://github.com/Creeper12345/NEuro_SAT)

## 环境安装

```bash
pip install -r requirements.txt
```


## 数据集

代码需要 EuroSAT RGB 数据集，**该文件夹不包含在仓库中**，需自行下载后放在项目根目录，结构如下：

```
EuroSAT_RGB/
├── AnnualCrop/
├── Forest/
├── HerbaceousVegetation/
├── Highway/
├── Industrial/
├── Pasture/
├── PermanentCrop/
├── Residential/
├── River/
└── SeaLake/
```

## 代码文件说明与运行顺序

| 文件 | 说明 |
|------|------|
| `data_loader.py` | 数据加载与预处理模块 |
| `model.py` | MLP 模型定义与手动反向传播 |
| `optimizer.py` | SGD 优化器与交叉熵损失 |
| `train.py` | 训练主循环 |
| `evaluate.py` | 测试集评估与混淆矩阵 |
| `hyperparam_search.py` | 超参数搜索 |
| `visualize.py` | 训练曲线、权重图像、错误样例可视化 |
| `visualize_class_neurons.py` | 类别神经元可视化 |
| `pack_outputs.py` | 打包 outputs/ 目录 |

### 第一步：训练模型

```bash
python train.py \
  --data_dir EuroSAT_RGB \
  --epochs 50 \
  --batch_size 256 \
  --lr 1e-3 \
  --hidden1 256 \
  --hidden2 256 \
  --weight_decay 1e-4 \
  --lr_decay 0.95 \
  --activation relu \
  --save_path outputs/best_model.npz
```

### 第二步：测试评估

```bash
python evaluate.py \
  --data_dir EuroSAT_RGB \
  --weights outputs/best_model.npz \
  --hidden1 256 \
  --hidden2 256 \
  --activation relu
```

> 也可直接使用预训练权重：从 [Google Drive](https://drive.google.com/drive/folders/1LIUSWQcB7LQmbXyX4mYbj_dEdDgCBTRw) 下载后放至 `outputs/` 目录。注意权重名称略有不同。

### 第三步：超参数搜索

```bash
# 网格搜索
python hyperparam_search.py --data_dir EuroSAT_RGB --mode grid --epochs 20

# 随机搜索
python hyperparam_search.py --data_dir EuroSAT_RGB --mode random --n_trials 20 --epochs 20
```

### 第四步：可视化

```bash
python visualize.py \
  --data_dir EuroSAT_RGB \
  --weights outputs/best_model.npz \
  --history outputs/best_model_history.npz \
  --hidden1 256 \
  --hidden2 256 \
  --activation relu
```

### 第五步：类别神经元可视化

```bash
python visualize_class_neurons.py \
  --data_dir EuroSAT_RGB \
  --weights outputs/best_model.npz \
  --hidden1 256 \
  --hidden2 256 \
  --activation relu
```

### 第六步：打包输出文件

```bash
python pack_outputs.py --output_dir outputs
```
