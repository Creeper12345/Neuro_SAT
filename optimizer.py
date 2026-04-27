"""
optimizer.py
------------
SGD 优化器（支持动量和学习率衰减）及交叉熵损失函数。

包含：
    cross_entropy_loss  – 计算交叉熵损失（可选 L2 正则化项）
    SGD                 – 带动量的 SGD，支持每 epoch 学习率衰减
"""

import numpy as np
from model import MLP, softmax


# 交叉熵损失

def cross_entropy_loss(probs: np.ndarray,
                       y:     np.ndarray,
                       model: MLP = None,
                       weight_decay: float = 0.0) -> float:
    """
    计算交叉熵损失。

    参数：
        probs        : (N, C)  模型输出的 softmax 概率
        y            : (N,)    真实类别标签（整数）
        model        : MLP     若提供，则在损失中加入 W1/W2/W3 的 L2 惩罚
        weight_decay : float   L2 系数 λ（0 表示不正则化）
    """
    N = len(y)
    # 截断防止 log(0)
    log_probs = np.log(np.clip(probs[np.arange(N), y], 1e-12, 1.0))
    ce_loss   = -log_probs.mean()

    if weight_decay > 0.0 and model is not None:
        l2 = (np.sum(model.W1 ** 2) +
              np.sum(model.W2 ** 2) +
              np.sum(model.W3 ** 2))
        ce_loss += 0.5 * weight_decay * l2

    return float(ce_loss)


# SGD 优化器
class SGD:
    """
    带动量的随机梯度下降（SGD）。

    参数：
        model        : MLP    要更新参数的模型
        lr           : float  初始学习率
        momentum     : float  动量系数（0 = 无动量的普通 SGD）
        weight_decay : float  L2 正则化系数 λ
        lr_decay     : float  每 epoch 学习率乘法衰减因子（如 0.95 = 每轮降 5%）
        lr_min       : float  学习率下限，防止衰减过小
    """

    def __init__(self,
                 model:        MLP,
                 lr:           float = 1e-3,
                 momentum:     float = 0.9,
                 weight_decay: float = 1e-4,
                 lr_decay:     float = 0.95,
                 lr_min:       float = 1e-6):

        self.model        = model
        self.lr           = lr
        self.momentum     = momentum
        self.weight_decay = weight_decay
        self.lr_decay     = lr_decay
        self.lr_min       = lr_min
        self._step        = 0

        self._velocity = {
            "W1": np.zeros_like(model.W1),
            "b1": np.zeros_like(model.b1),
            "W2": np.zeros_like(model.W2),
            "b2": np.zeros_like(model.b2),
            "W3": np.zeros_like(model.W3),
            "b3": np.zeros_like(model.b3),
        }

    def step(self, grads: dict):
        lr = self.lr
        mu = self.momentum

        param_map = {
            "W1": "dW1", "b1": "db1",
            "W2": "dW2", "b2": "db2",
            "W3": "dW3", "b3": "db3",
        }

        for param_name, grad_name in param_map.items():
            g = grads[grad_name]
            v = self._velocity[param_name]

            # 动量更新：v ← μv + (1-μ)g
            self._velocity[param_name] = mu * v + (1.0 - mu) * g

            param = getattr(self.model, param_name)
            param -= lr * self._velocity[param_name]
            setattr(self.model, param_name, param)

        self._step += 1

    def zero_grad(self):
        """占位方法：本实现中梯度每次反向传播时重新计算，无需清零。"""
        pass

    def decay_lr(self):
        """每个 epoch 结束时调用，对学习率做乘法衰减。"""
        self.lr = max(self.lr * self.lr_decay, self.lr_min)

    def get_lr(self) -> float:
        return self.lr

    def __repr__(self):
        return (f"SGD(lr={self.lr:.2e}, momentum={self.momentum}, "
                f"weight_decay={self.weight_decay}, lr_decay={self.lr_decay})")


if __name__ == "__main__":
    model = MLP(input_dim=16, hidden1=8, hidden2=4, num_classes=3)
    opt   = SGD(model, lr=0.01, momentum=0.9, weight_decay=1e-4, lr_decay=0.9)

    X = np.random.randn(10, 16).astype(np.float32)
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

    probs = model.forward(X)
    loss  = cross_entropy_loss(probs, y, model=model, weight_decay=1e-4)
    print(f"Initial loss: {loss:.4f}")

    grads = model.backward(y, probs, weight_decay=1e-4)
    opt.step(grads)
    opt.decay_lr()
    print(f"LR after decay: {opt.get_lr():.6f}")

    probs2 = model.forward(X)
    loss2  = cross_entropy_loss(probs2, y, model=model, weight_decay=1e-4)
    print(f"Loss after one step: {loss2:.4f}  (should be ≤ {loss:.4f})")
