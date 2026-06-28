"""Command-line interface for DriftLens Lab experiments."""

import argparse
import sys
from typing import List, Optional, Tuple

from .detectors import DDM, EWMADrift, PageHinkley
from .metrics import MetricsTracker
from .models import OnlineLogisticRegression, Perceptron
from .reports import ExperimentResult, write_report
from .streams import StreamConfig, stream


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="drift-lens-lab",
        description="Simulate concept drift, train an online classifier, detect drift, write report.",
    )
    p.add_argument("--steps", type=int, default=500, help="Number of stream steps (default: 500)")
    p.add_argument(
        "--drift",
        choices=["none", "sudden", "gradual", "recurring"],
        default="sudden",
        metavar="KIND",
        help="Drift kind: none | sudden | gradual | recurring (default: sudden)",
    )
    p.add_argument("--drift-at", type=int, default=None, metavar="STEP",
                   help="Step at which drift begins (default: steps//2)")
    p.add_argument("--drift-width", type=int, default=50, metavar="N",
                   help="Width of gradual drift transition (default: 50)")
    p.add_argument("--features", type=int, default=10, metavar="N",
                   help="Number of input features (default: 10)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument(
        "--model",
        choices=["logreg", "perceptron"],
        default="logreg",
        help="Online model: logreg | perceptron (default: logreg)",
    )
    p.add_argument(
        "--detector",
        choices=["ddm", "ph", "ewma"],
        default="ddm",
        help="Drift detector: ddm | ph | ewma (default: ddm)",
    )
    p.add_argument("--window", type=int, default=100, help="Rolling metric window (default: 100)")
    p.add_argument("--report", metavar="PATH", default=None,
                   help="Write HTML (or .md) report to PATH")
    return p


def run_experiment(
    steps: int,
    drift_kind: str,
    drift_at: int,
    drift_width: int,
    n_features: int,
    seed: int,
    model_name: str,
    detector_name: str,
    window: int,
    report_path: Optional[str],
) -> ExperimentResult:
    cfg = StreamConfig(
        n_features=n_features,
        drift_kind=drift_kind,
        drift_at=drift_at,
        drift_width=drift_width,
        seed=seed,
    )

    if model_name == "logreg":
        model = OnlineLogisticRegression(n_features=n_features)
    else:
        model = Perceptron(n_features=n_features)

    if detector_name == "ddm":
        detector = DDM()
    elif detector_name == "ph":
        detector = PageHinkley()
    else:
        detector = EWMADrift()  # ewma branch

    tracker = MetricsTracker(window=window)
    drift_events: List[Tuple[int, str]] = []

    src = stream(cfg)
    for _ in range(steps):
        step, x, label = next(src)
        prob = model.partial_fit(x, label)
        pred = 1 if prob >= 0.5 else 0
        tracker.update(step, prob, pred, label)

        error = int(pred != label)
        if detector_name == "ddm":
            status = detector.update(error)
            if status in ("drift", "warning"):
                drift_events.append((step, status))
        elif detector_name == "ph":
            if detector.update(float(error)):
                drift_events.append((step, "drift"))
                detector.reset()
        else:
            status = detector.update(error)
            if status in ("drift", "warning"):
                drift_events.append((step, status))

    result = ExperimentResult(
        steps=steps,
        drift_kind=drift_kind,
        drift_at=drift_at,
        seed=seed,
        model_name=model_name,
        detector_name=detector_name,
        drift_events=drift_events,
        snapshots=tracker.snapshots,
        final_accuracy=tracker.current_accuracy,
        final_log_loss=tracker.current_log_loss,
    )

    if report_path:
        write_report(result, report_path)

    return result


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    drift_at = args.drift_at if args.drift_at is not None else args.steps // 2

    result = run_experiment(
        steps=args.steps,
        drift_kind=args.drift,
        drift_at=drift_at,
        drift_width=args.drift_width,
        n_features=args.features,
        seed=args.seed,
        model_name=args.model,
        detector_name=args.detector,
        window=args.window,
        report_path=args.report,
    )

    acc = f"{result.final_accuracy:.4f}" if result.final_accuracy is not None else "n/a"
    loss = f"{result.final_log_loss:.4f}" if result.final_log_loss is not None else "n/a"
    n_drift = sum(1 for _, k in result.drift_events if k == "drift")
    n_warn = sum(1 for _, k in result.drift_events if k == "warning")

    print(f"Steps        : {result.steps}")
    print(f"Drift kind   : {result.drift_kind}  (at step {result.drift_at})")
    print(f"Model        : {result.model_name}")
    print(f"Detector     : {result.detector_name}")
    print(f"Accuracy     : {acc}")
    print(f"Log-loss     : {loss}")
    print(f"Drift alarms : {n_drift}  warnings: {n_warn}")
    if args.report:
        print(f"Report       : {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
