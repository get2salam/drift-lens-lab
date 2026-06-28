"""Deterministic synthetic binary-classification streams with drift."""

import math
import random
from dataclasses import dataclass
from typing import Generator, Literal, Tuple

DriftKind = Literal["none", "sudden", "gradual", "recurring"]


@dataclass
class StreamConfig:
    n_features: int = 10
    drift_kind: DriftKind = "sudden"
    drift_at: int = 500
    drift_width: int = 50       # only used for gradual drift
    recur_period: int = 200     # only used for recurring drift
    seed: int = 42


class _RNG:
    """Minimal deterministic RNG (xorshift64) to avoid numpy dependency."""

    def __init__(self, seed: int) -> None:
        self._state = seed ^ 0xDEADBEEF or 1

    def _next(self) -> int:
        x = self._state
        x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
        x ^= (x >> 7) & 0xFFFFFFFFFFFFFFFF
        x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
        self._state = x & 0xFFFFFFFFFFFFFFFF
        return self._state

    def uniform(self) -> float:
        return (self._next() & 0xFFFFFFFF) / 0xFFFFFFFF

    def gauss(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        # Box-Muller
        u1 = max(self.uniform(), 1e-15)
        u2 = self.uniform()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        return mu + sigma * z

    def choice(self, seq):
        return seq[self._next() % len(seq)]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, x))))


def _sample(rng: _RNG, weights, bias: float) -> Tuple[list, int]:
    """Draw one (features, label) pair from a linear model + noise."""
    x = [rng.gauss() for _ in weights]
    logit = sum(w * xi for w, xi in zip(weights, x)) + bias
    prob = _sigmoid(logit + rng.gauss(0, 0.3))
    label = 1 if rng.uniform() < prob else 0
    return x, label


def _make_weights(rng: _RNG, n: int):
    return [rng.gauss(0, 1.5) for _ in range(n)]


def stream(cfg: StreamConfig) -> Generator[Tuple[int, list, int], None, None]:
    """
    Yield (step, features, label) indefinitely.

    Drift changes the underlying class-boundary weights at the configured step.
    """
    rng = _RNG(cfg.seed)
    n = cfg.n_features

    weights_a = _make_weights(rng, n)
    weights_b = _make_weights(rng, n)
    bias = rng.gauss(0, 0.5)

    step = 0
    while True:
        if cfg.drift_kind == "none":
            w = weights_a

        elif cfg.drift_kind == "sudden":
            w = weights_b if step >= cfg.drift_at else weights_a

        elif cfg.drift_kind == "gradual":
            if step < cfg.drift_at:
                alpha = 0.0
            elif step < cfg.drift_at + cfg.drift_width:
                alpha = (step - cfg.drift_at) / cfg.drift_width
            else:
                alpha = 1.0
            w = [a + alpha * (b - a) for a, b in zip(weights_a, weights_b)]

        elif cfg.drift_kind == "recurring":
            cycle = (step // cfg.recur_period) % 2
            w = weights_a if cycle == 0 else weights_b

        else:
            raise ValueError(f"Unknown drift kind: {cfg.drift_kind!r}")

        x, label = _sample(rng, w, bias)
        yield step, x, label
        step += 1
