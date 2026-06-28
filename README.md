# DriftLens Lab

> A tiny, dependency-free ML lab for simulating concept drift, training online classifiers from scratch, detecting drift, and generating readable monitoring reports.

[![CI](https://github.com/get2salam/drift-lens-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/get2salam/drift-lens-lab/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## What is concept drift?

In production ML systems, the statistical relationship between features and labels often changes over time — this is **concept drift**. A fraud model trained last year may silently degrade as attack patterns evolve. DriftLens Lab lets you simulate these scenarios in a controlled setting, observe how online learners respond, and evaluate detector sensitivity without needing real data or cloud infrastructure.

---

## Features

| Module | What it does |
|--------|-------------|
| `streams.py` | Deterministic synthetic streams with **sudden**, **gradual**, and **recurring** drift |
| `models.py` | Online **Logistic Regression** and **Perceptron** trained sample-by-sample |
| `detectors.py` | **DDM** (Gama 2004), **Page-Hinkley**, and **EWMA** drift detectors |
| `metrics.py` | Sliding-window rolling accuracy and log-loss |
| `reports.py` | Self-contained **HTML** and **Markdown** reports with inline SVG charts |
| `cli.py` | One-command experiment runner |

Zero runtime dependencies — pure Python standard library.

---

## Install

```bash
git clone https://github.com/get2salam/drift-lens-lab.git
cd drift-lens-lab
pip install -e .          # optional: makes `drift-lens-lab` available as a CLI alias
```

---

## Quick start

```bash
# Sudden drift at step 150, 300 total steps, HTML report
python -m drift_lens_lab --steps 300 --drift sudden --drift-at 150 --seed 42 --report out/report.html
```

Sample output:

```
Steps        : 300
Drift kind   : sudden  (at step 150)
Model        : logreg
Detector     : ddm
Accuracy     : 0.7600
Log-loss     : 0.4969
Drift alarms : 0  warnings: 0
Report       : out/report.html
```

The log-loss sparkline in the report typically shows a characteristic dip-and-recovery around the drift point, even when the online model adapts quickly.

---

## CLI options

```
python -m drift_lens_lab [OPTIONS]

  --steps N          Stream length (default: 500)
  --drift KIND       none | sudden | gradual | recurring (default: sudden)
  --drift-at STEP    Step at which drift begins (default: steps//2)
  --drift-width N    Transition length for gradual drift (default: 50)
  --features N       Number of input features (default: 10)
  --seed N           Random seed (default: 42)
  --model MODEL      logreg | perceptron (default: logreg)
  --detector DET     ddm | ph | ewma (default: ddm)
  --window N         Rolling metric window (default: 100)
  --report PATH      Write HTML or Markdown report to PATH
```

---

## Run tests

```bash
python -m unittest discover -s tests -v
```

All 39 tests are deterministic and offline — no network, no downloads.

---

## Example

```bash
python examples/run_experiment.py
```

---

## Project layout

```
drift_lens_lab/
├── __init__.py
├── __main__.py       # python -m drift_lens_lab
├── streams.py        # synthetic drift streams
├── models.py         # online LR and Perceptron
├── detectors.py      # DDM, Page-Hinkley, EWMA
├── metrics.py        # rolling accuracy and log-loss
├── reports.py        # HTML/Markdown report generator
└── cli.py            # argparse CLI
tests/                # 39 unit tests
examples/             # reproducible example script
```

---

## Why online learning?

Classic batch ML requires retraining from scratch when data shifts. Online learners update incrementally after each sample, making them natural fits for streaming pipelines. Drift detectors add an observability layer — triggering alerts or model resets when the error rate statistics deviate from baseline.

---

## License

MIT © Abdul Salam
