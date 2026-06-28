"""Concept-drift detectors."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DriftEvent:
    step: int
    kind: str       # "warning" or "drift"
    detector: str


class PageHinkley:
    """
    Page-Hinkley test for detecting a persistent shift in the mean of a sequence.

    Raises a drift alarm when the cumulative deviation from the running mean
    exceeds a threshold delta scaled by lambda.

    Parameters
    ----------
    delta : float
        Minimum magnitude of change to detect (sensitivity).
    lambda_ : float
        Threshold for alarm.
    alpha : float
        Forgetting factor for the running mean (1.0 = no forgetting).
    """

    def __init__(self, delta: float = 0.005, lambda_: float = 50.0, alpha: float = 1.0) -> None:
        self.delta = delta
        self.lambda_ = lambda_
        self.alpha = alpha
        self._reset()

    def _reset(self) -> None:
        self._sum = 0.0
        self._mean = 0.0
        self._n = 0
        self._min_sum = 0.0

    def update(self, value: float) -> bool:
        """Return True if drift is detected."""
        self._n += 1
        self._mean = self.alpha * self._mean + (1 - self.alpha) * value if self._n > 1 else value
        self._sum += value - self._mean - self.delta
        self._min_sum = min(self._min_sum, self._sum)
        return (self._sum - self._min_sum) > self.lambda_

    def reset(self) -> None:
        self._reset()


class DDM:
    """
    Drift Detection Method (Gama et al., 2004) using error-rate statistics.

    Monitors mean and std of binary error (0/1) and raises warnings/alarms
    when they exceed calibrated multiples of the minimum observed values.

    Parameters
    ----------
    warning_level : float
        Std-dev multiplier for warning state.
    drift_level : float
        Std-dev multiplier for drift alarm.
    min_samples : int
        Warmup samples before detection is active.
    """

    def __init__(
        self, warning_level: float = 2.0, drift_level: float = 3.0, min_samples: int = 30
    ) -> None:
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.min_samples = min_samples
        self._reset()

    def _reset(self) -> None:
        self._n = 0
        self._err_sum = 0.0
        self._min_p = float("inf")
        self._min_std = float("inf")
        self.in_warning = False

    def _stats(self):
        p = self._err_sum / self._n
        std = (p * (1 - p) / self._n) ** 0.5
        return p, std

    def update(self, error: int) -> str:
        """
        Feed binary error (0=correct, 1=wrong).

        Returns 'drift', 'warning', or 'ok'.
        """
        self._n += 1
        self._err_sum += error
        if self._n < self.min_samples:
            return "ok"

        p, std = self._stats()
        if p + std < self._min_p + self._min_std:
            self._min_p = p
            self._min_std = std

        level = p + std
        min_level = self._min_p + self._min_std

        if level > min_level + self.drift_level * self._min_std:
            self._reset()
            return "drift"
        if level > min_level + self.warning_level * self._min_std:
            self.in_warning = True
            return "warning"

        self.in_warning = False
        return "ok"


class EWMADrift:
    """
    EWMA (exponentially weighted moving average) drift detector.

    Signals drift when the smoothed error rate deviates beyond a
    z-score threshold from its minimum baseline.
    """

    def __init__(self, alpha: float = 0.1, z_thresh: float = 3.5, min_samples: int = 30) -> None:
        self.alpha = alpha
        self.z_thresh = z_thresh
        self.min_samples = min_samples
        self._ewma = 0.0
        self._ewma_var = 0.0
        self._n = 0
        self._baseline: Optional[float] = None

    def update(self, error: int) -> str:
        self._n += 1
        if self._n == 1:
            self._ewma = float(error)
            self._ewma_var = 0.0
            return "ok"
        self._ewma_var = (1 - self.alpha) * (self._ewma_var + self.alpha * (error - self._ewma) ** 2)
        self._ewma = (1 - self.alpha) * self._ewma + self.alpha * error

        if self._n < self.min_samples:
            return "ok"

        if self._baseline is None:
            self._baseline = self._ewma

        std = self._ewma_var ** 0.5 or 1e-9
        if (self._ewma - self._baseline) / std > self.z_thresh:
            self._baseline = self._ewma
            return "drift"
        return "ok"
