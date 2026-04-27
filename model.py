"""
model.py
--------
基于 NumPy 手动实现的三层 MLP 分类器。

网络结构：
    输入层 (12288) → 全连接+激活 → 隐藏层1 → 全连接+激活 → 隐藏层2 → 全连接 → Softmax (10类)

所有梯度均手动推导，不依赖任何自动微分框架。
"""

import numpy as np


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)

def relu_grad(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(z.dtype)

def sigmoid(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0,
                    1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))

def sigmoid_grad(z: np.ndarray) -> np.ndarray:
    s = sigmoid(z)
    return s * (1.0 - s)

def tanh_act(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)

def tanh_grad(z: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(z) ** 2


ACTIVATIONS = {
    "relu":    (relu,     relu_grad),
    "sigmoid": (sigmoid,  sigmoid_grad),
    "tanh":    (tanh_act, tanh_grad),
}


def softmax(z: np.ndarray) -> np.ndarray:
    """按行计算数值稳定的 softmax。"""
    z_shifted = z - z.max(axis=1, keepdims=True)
    exp_z     = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


class MLP:
    """
    三层多层感知机（MLP）：
        第1层：Linear(input_dim → hidden1) + 激活函数
        第2层：Linear(hidden1  → hidden2)  + 激活函数
        第3层：Linear(hidden2  → num_classes) + Softmax

    参数说明：
        input_dim   : 输入维度（EuroSAT 为 12288）
        hidden1     : 第一隐藏层神经元数（默认 512）
        hidden2     : 第二隐藏层神经元数（默认 256）
        num_classes : 输出类别数（默认 10）
        activation  : 激活函数，支持 'relu' / 'sigmoid' / 'tanh'
        seed        : 权重初始化随机种子
    """

    def __init__(self,
                 input_dim:   int = 12_288,
                 hidden1:     int = 512,
                 hidden2:     int = 256,
                 num_classes: int = 10,
                 activation:  str = "relu",
                 seed:        int = 0):

        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}'. "
                             f"Choose from {list(ACTIVATIONS)}.")

        self.input_dim   = input_dim
        self.hidden1     = hidden1
        self.hidden2     = hidden2
        self.num_classes = num_classes
        self.activation  = activation
        self._act_fn, self._act_grad = ACTIVATIONS[activation]

        rng = np.random.default_rng(seed)
        self._init_weights(rng)

        self._cache: dict = {}

    # 权重初始化（Xavier / Glorot 均匀分布）
    def _init_weights(self, rng: np.random.Generator):
        def xavier(fan_in, fan_out):
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float32)

        self.W1 = xavier(self.input_dim, self.hidden1)
        self.b1 = np.zeros((1, self.hidden1),     dtype=np.float32)

        self.W2 = xavier(self.hidden1, self.hidden2)
        self.b2 = np.zeros((1, self.hidden2),     dtype=np.float32)

        self.W3 = xavier(self.hidden2, self.num_classes)
        self.b3 = np.zeros((1, self.num_classes), dtype=np.float32)


    def forward(self, X: np.ndarray) -> np.ndarray:
        # 第1层
        Z1 = X @ self.W1 + self.b1           # (N, hidden1)
        A1 = self._act_fn(Z1)                # (N, hidden1)

        # 第2层
        Z2 = A1 @ self.W2 + self.b2          # (N, hidden2)
        A2 = self._act_fn(Z2)                # (N, hidden2)

        # 输出层
        Z3 = A2 @ self.W3 + self.b3          # (N, num_classes)

        # 保存中间值，供反向传播使用
        self._cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3}

        return softmax(Z3)                   # (N, num_classes)


    def backward(self, y: np.ndarray,
                 probs: np.ndarray,
                 weight_decay: float = 0.0) -> dict:
        X  = self._cache["X"]
        Z1 = self._cache["Z1"]
        A1 = self._cache["A1"]
        Z2 = self._cache["Z2"]
        A2 = self._cache["A2"]
        N  = X.shape[0]

        # 输出层梯度：d(CE)/dZ3 = probs - one_hot(y)
        one_hot  = np.zeros_like(probs)
        one_hot[np.arange(N), y] = 1.0
        dZ3 = (probs - one_hot) / N              # (N, num_classes)

        dW3 = A2.T @ dZ3                         # (hidden2, num_classes)
        db3 = dZ3.sum(axis=0, keepdims=True)     # (1, num_classes)
        dA2 = dZ3 @ self.W3.T                    # (N, hidden2)

        # 第2层梯度
        dZ2 = dA2 * self._act_grad(Z2)           # (N, hidden2)
        dW2 = A1.T @ dZ2                         # (hidden1, hidden2)
        db2 = dZ2.sum(axis=0, keepdims=True)     # (1, hidden2)
        dA1 = dZ2 @ self.W2.T                    # (N, hidden1)

        # 第1层梯度
        dZ1 = dA1 * self._act_grad(Z1)           # (N, hidden1)
        dW1 = X.T @ dZ1                          # (input_dim, hidden1)
        db1 = dZ1.sum(axis=0, keepdims=True)     # (1, hidden1)

        # L2 正则化梯度
        if weight_decay > 0.0:
            dW1 += weight_decay * self.W1
            dW2 += weight_decay * self.W2
            dW3 += weight_decay * self.W3

        return {"dW1": dW1, "db1": db1,
                "dW2": dW2, "db2": db2,
                "dW3": dW3, "db3": db3}


    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.forward(X)
        return probs.argmax(axis=1)


    def save_weights(self, path: str):
        """将所有参数保存为 .npz 文件。"""
        np.savez(path,
                 W1=self.W1, b1=self.b1,
                 W2=self.W2, b2=self.b2,
                 W3=self.W3, b3=self.b3)
        print(f"[MLP] Weights saved → {path}")

    def load_weights(self, path: str):
        """从 .npz 文件中加载参数。"""
        data = np.load(path)
        self.W1 = data["W1"]
        self.b1 = data["b1"]
        self.W2 = data["W2"]
        self.b2 = data["b2"]
        self.W3 = data["W3"]
        self.b3 = data["b3"]
        print(f"[MLP] Weights loaded ← {path}")

    def __repr__(self):
        return (f"MLP(input={self.input_dim} → h1={self.hidden1} → "
                f"h2={self.hidden2} → out={self.num_classes}, "
                f"act={self.activation})")


if __name__ == "__main__":
    model = MLP(input_dim=12_288, hidden1=512, hidden2=256, num_classes=10)
    print(model)

    X_dummy = np.random.randn(8, 12_288).astype(np.float32)
    y_dummy = np.array([0, 1, 2, 3, 4, 5, 6, 7])

    probs = model.forward(X_dummy)
    print("probs shape:", probs.shape)
    print("probs sum (should be 1):", probs.sum(axis=1))

    grads = model.backward(y_dummy, probs, weight_decay=1e-4)
    for k, v in grads.items():
        print(f"  {k}: {v.shape}")
