"""Score a detector's reported alarms against a known ground-truth drift point."""

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class DetectorScore:
    """Result of scoring a detector's alarms against a known drift point."""

    detected: bool
    detection_delay: Optional[int]
    false_alarms: int
    total_alarms: int

    @property
    def precision(self) -> Optional[float]:
        """Fraction of alarms that were the true-positive detection (None if no alarms fired)."""
        if self.total_alarms == 0:
            return None
        true_positives = 1 if self.detected else 0
        return true_positives / self.total_alarms


def score_detector(
    drift_events: List[Tuple[int, str]],
    drift_at: Optional[int],
    max_delay: int = 200,
    event_kind: str = "drift",
) -> DetectorScore:
    """
    Score a detector's reported events against a single known drift point.

    An `event_kind` event at or after `drift_at` and within `max_delay` steps
    counts as the true-positive detection; the earliest such event's distance
    from `drift_at` is the detection delay. Any `event_kind` event strictly
    before `drift_at`, or fired after the detection window closes
    (`drift_at + max_delay`) with no prior detection, counts as a false alarm
    since it cannot be attributed to the known drift.

    If `drift_at` is None (no drift injected into the stream), every
    `event_kind` event is a false alarm and `detected` is always False.
    """
    alarms = [step for step, kind in drift_events if kind == event_kind]
    total_alarms = len(alarms)

    if drift_at is None:
        return DetectorScore(
            detected=False, detection_delay=None,
            false_alarms=total_alarms, total_alarms=total_alarms,
        )

    window_end = drift_at + max_delay
    false_alarms = sum(1 for s in alarms if s < drift_at or s > window_end)
    in_window = [s for s in alarms if drift_at <= s <= window_end]

    if in_window:
        first = min(in_window)
        return DetectorScore(
            detected=True, detection_delay=first - drift_at,
            false_alarms=false_alarms, total_alarms=total_alarms,
        )

    return DetectorScore(
        detected=False, detection_delay=None,
        false_alarms=false_alarms, total_alarms=total_alarms,
    )
