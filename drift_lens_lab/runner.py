"""High-level convenience wrapper for running experiments programmatically."""

from typing import List, Optional, Tuple

from .detectors import DDM, EWMADrift, PageHinkley
from .metrics import MetricsTracker
from .models import OnlineLogisticRegression, Perceptron
from .reports import ExperimentResult, write_report
from .streams import StreamConfig, stream


def run(
    steps: int = 500,
    drift_kind: str = "sudden",
    drift_at: Optional[int] = None,
    drift_width: int = 50,
    n_features: int = 10,
    seed: int = 42,
    model: str = "logreg",
    detector: str = "ddm",
    window: int = 100,
    report: Optional[str] = None,
) -> ExperimentResult:
    """
    Run a full drift experiment and return an ExperimentResult.

    Parameters mirror the CLI flags; ``report`` is a file path (HTML or .md).
    """
    from .cli import run_experiment
    return run_experiment(
        steps=steps,
        drift_kind=drift_kind,
        drift_at=drift_at if drift_at is not None else steps // 2,
        drift_width=drift_width,
        n_features=n_features,
        seed=seed,
        model_name=model,
        detector_name=detector,
        window=window,
        report_path=report,
    )
