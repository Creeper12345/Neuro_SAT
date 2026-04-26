"""
optimizer.py
------------
SGD optimiser with learning-rate decay + cross-entropy loss with L2 regularisation.

Classes
-------
SGD            – vanilla SGD (+ optional momentum) with step-based LR decay
CrossEntropyLoss – computes loss value (backward pass lives in model.py)
"""

import numpy as np
from model import MLP, softmax


# cross-entropy loss

def cross_entropy_loss(probs: np.ndarray,
                       y:     np.ndarray,
                       model: MLP = None,
                       weight_decay: float = 0.0) -> float:
    """
    Compute mean cross-entropy loss  + optional L2 regularisation term.

    Parameters
    ----------
    probs        : (N, C)  softmax probabilities (output of model.forward)
    y            : (N,)    integer ground-truth labels
    model        : MLP     – if provided, add L2 penalty on W1, W2, W3
    weight_decay : float   – λ for L2 penalty  (0 = disabled)

    Returns
    -------
    loss : float
    """
    N = len(y)
    # Clamp to avoid log(0)
    log_probs = np.log(np.clip(probs[np.arange(N), y], 1e-12, 1.0))
    ce_loss   = -log_probs.mean()

    if weight_decay > 0.0 and model is not None:
        l2 = (np.sum(model.W1 ** 2) +
              np.sum(model.W2 ** 2) +
              np.sum(model.W3 ** 2))
        ce_loss += 0.5 * weight_decay * l2

    return float(ce_loss)


# SGD optimiser
class SGD:
    """
    Stochastic Gradient Descent with optional momentum and step-based LR decay.

    Parameters
    ----------
    model        : MLP    – the model whose parameters will be updated
    lr           : float  – initial learning rate
    momentum     : float  – momentum coefficient (0 = vanilla SGD)
    weight_decay : float  – L2 regularisation coefficient λ
    lr_decay     : float  – multiplicative LR decay factor applied each epoch
                            e.g. 0.95 reduces LR by 5 % every epoch
    lr_min       : float  – floor on the learning rate
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

            # Momentum update: v ← μv + (1-μ)g  
            self._velocity[param_name] = mu * v + (1.0 - mu) * g

            param = getattr(self.model, param_name)
            param -= lr * self._velocity[param_name]
            setattr(self.model, param_name, param)

        self._step += 1

    def zero_grad(self):
        """No-op: gradients are recomputed fresh each backward call."""
        pass

    # ── epoch-level LR decay ───────────────────────────────────────────────
    def decay_lr(self):
        """Call once per epoch to apply multiplicative LR decay."""
        self.lr = max(self.lr * self.lr_decay, self.lr_min)

    def get_lr(self) -> float:
        return self.lr

    def __repr__(self):
        return (f"SGD(lr={self.lr:.2e}, momentum={self.momentum}, "
                f"weight_decay={self.weight_decay}, lr_decay={self.lr_decay})")


# ── quick sanity check ─────────────────────────────────────────────────────────
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
