"""Rolling metrics for online evaluation."""

import math
from collections import deque
from typing import Deque, List, Optional, Tuple


class RollingAccuracy:
    """Sliding-window accuracy over the last `window` predictions."""

    def __init__(self, window: int = 100) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.window = window
        self._buf: Deque[int] = deque(maxlen=window)
        self._correct: int = 0  # running sum; keeps .value O(1) instead of O(window)

    def update(self, pred: int, label: int) -> None:
        correct = int(pred == label)
        if len(self._buf) == self.window:
            self._correct -= self._buf[0]  # subtract element about to be evicted
        self._buf.append(correct)
        self._correct += correct

    @property
    def value(self) -> Optional[float]:
        if not self._buf:
            return None
        return self._correct / len(self._buf)

    @property
    def count(self) -> int:
        return len(self._buf)


class RollingLogLoss:
    """Sliding-window log-loss."""

    def __init__(self, window: int = 100, eps: float = 1e-7) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.window = window
        self.eps = eps
        self._buf: Deque[float] = deque(maxlen=window)
        self._sum: float = 0.0  # running sum; keeps .value O(1) instead of O(window)

    def update(self, prob: float, label: int) -> None:
        p = max(self.eps, min(1 - self.eps, prob))
        loss = -(label * math.log(p) + (1 - label) * math.log(1 - p))
        if len(self._buf) == self.window:
            self._sum -= self._buf[0]  # subtract element about to be evicted
        self._buf.append(loss)
        self._sum += loss

    @property
    def value(self) -> Optional[float]:
        if not self._buf:
            return None
        return self._sum / len(self._buf)


class MetricsTracker:
    """Collects per-step snapshots for report generation."""

    def __init__(self, window: int = 100) -> None:
        self._acc = RollingAccuracy(window)
        self._loss = RollingLogLoss(window)
        self.snapshots: List[Tuple[int, float, float]] = []   # (step, acc, loss)
        self._snapshot_every = max(1, window // 10)

    def update(self, step: int, prob: float, pred: int, label: int) -> None:
        self._acc.update(pred, label)
        self._loss.update(prob, label)
        if step % self._snapshot_every == 0:
            acc = self._acc.value or 0.0
            loss = self._loss.value or 0.0
            self.snapshots.append((step, acc, loss))

    @property
    def current_accuracy(self) -> Optional[float]:
        return self._acc.value

    @property
    def current_log_loss(self) -> Optional[float]:
        return self._loss.value
