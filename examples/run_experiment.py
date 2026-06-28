"""
Reproducible example: sudden concept drift at step 150, 300 total steps.

Run from the repo root:
    python examples/run_experiment.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from drift_lens_lab.cli import run_experiment
from drift_lens_lab.reports import to_markdown

result = run_experiment(
    steps=300,
    drift_kind="sudden",
    drift_at=150,
    drift_width=50,
    n_features=10,
    seed=42,
    model_name="logreg",
    detector_name="ddm",
    window=100,
    report_path=None,
)

print(to_markdown(result))
print(f"\nDrift events : {result.drift_events}")
print(f"Final accuracy: {result.final_accuracy:.4f}")
print(f"Final log-loss: {result.final_log_loss:.4f}")
