"""Online learning models implemented from scratch."""

import math
from typing import List, Optional


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, x))))


class OnlineLogisticRegression:
    """
    Stochastic gradient-descent logistic regression with optional L2 penalty.

    Supports partial_fit(x, y) for true online / streaming use.
    """

    def __init__(
        self,
        n_features: int,
        lr: float = 0.05,
        l2: float = 1e-4,
        decay: float = 1.0,
    ) -> None:
        self.n_features = n_features
        self.lr = lr
        self.l2 = l2
        self.decay = decay          # multiply lr by this each step (cosine-like schedule via caller)
        self.weights: List[float] = [0.0] * n_features
        self.bias: float = 0.0
        self._step = 0

    def _logit(self, x: List[float]) -> float:
        return sum(w * xi for w, xi in zip(self.weights, x)) + self.bias

    def predict_proba(self, x: List[float]) -> float:
        return _sigmoid(self._logit(x))

    def predict(self, x: List[float]) -> int:
        return 1 if self.predict_proba(x) >= 0.5 else 0

    def partial_fit(self, x: List[float], y: int) -> float:
        """Update on a single example; returns the predicted probability before update."""
        p = self.predict_proba(x)
        err = p - y
        lr = self.lr * (self.decay ** self._step)
        for i, xi in enumerate(x):
            self.weights[i] -= lr * (err * xi + self.l2 * self.weights[i])
        self.bias -= lr * err
        self._step += 1
        return p


class Perceptron:
    """Classic online Perceptron (Rosenblatt 1958) with optional passive-aggressive update."""

    def __init__(self, n_features: int, passive_aggressive: bool = True, C: float = 1.0) -> None:
        self.n_features = n_features
        self.passive_aggressive = passive_aggressive
        self.C = C
        self.weights: List[float] = [0.0] * n_features
        self.bias: float = 0.0

    def _margin(self, x: List[float], y: int) -> float:
        yy = 1.0 if y == 1 else -1.0
        raw = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return yy * raw

    def predict_proba(self, x: List[float]) -> float:
        raw = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return _sigmoid(raw)

    def predict(self, x: List[float]) -> int:
        raw = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return 1 if raw >= 0 else 0

    def partial_fit(self, x: List[float], y: int) -> float:
        p = self.predict_proba(x)
        margin = self._margin(x, y)
        if margin >= 1.0:
            return p
        yy = 1.0 if y == 1 else -1.0
        if self.passive_aggressive:
            norm_sq = sum(xi ** 2 for xi in x) + 1.0
            loss = max(0.0, 1.0 - margin)
            tau = min(self.C, loss / norm_sq)
        else:
            tau = 1.0
        for i, xi in enumerate(x):
            self.weights[i] += tau * yy * xi
        self.bias += tau * yy
        return p
