"""CLI smoke and integration tests."""
import os
import tempfile
import unittest
from drift_lens_lab.cli import main, run_experiment


class TestCLISmoke(unittest.TestCase):
    def test_default_run(self):
        ret = main(["--steps", "50", "--seed", "42"])
        self.assertEqual(ret, 0)

    def test_gradual_drift(self):
        ret = main(["--steps", "100", "--drift", "gradual", "--seed", "1"])
        self.assertEqual(ret, 0)

    def test_recurring_drift(self):
        ret = main(["--steps", "100", "--drift", "recurring", "--seed", "3"])
        self.assertEqual(ret, 0)

    def test_perceptron_model(self):
        ret = main(["--steps", "50", "--model", "perceptron", "--seed", "5"])
        self.assertEqual(ret, 0)

    def test_ph_detector(self):
        ret = main(["--steps", "50", "--detector", "ph", "--seed", "7"])
        self.assertEqual(ret, 0)

    def test_ewma_detector(self):
        ret = main(["--steps", "50", "--detector", "ewma", "--seed", "9"])
        self.assertEqual(ret, 0)

    def test_report_html_written(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "out", "report.html")
            ret = main(["--steps", "100", "--seed", "42", "--report", out])
            self.assertEqual(ret, 0)
            self.assertTrue(os.path.exists(out))

    def test_report_md_written(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "report.md")
            ret = main(["--steps", "50", "--seed", "42", "--report", out])
            self.assertEqual(ret, 0)
            self.assertTrue(os.path.exists(out))


class TestRunExperimentDeterminism(unittest.TestCase):
    def _run(self):
        return run_experiment(
            steps=200, drift_kind="sudden", drift_at=100,
            drift_width=50, n_features=10, seed=42,
            model_name="logreg", detector_name="ddm",
            window=100, report_path=None,
        )

    def test_reproducible(self):
        r1 = self._run()
        r2 = self._run()
        self.assertAlmostEqual(r1.final_accuracy, r2.final_accuracy, places=10)
        self.assertEqual(r1.drift_events, r2.drift_events)


class TestRunExperimentValidation(unittest.TestCase):
    """run_experiment must raise ValueError for nonsensical inputs."""

    def _run(self, **overrides):
        defaults = dict(
            steps=50, drift_kind="sudden", drift_at=25,
            drift_width=10, n_features=5, seed=0,
            model_name="logreg", detector_name="ddm",
            window=20, report_path=None,
        )
        defaults.update(overrides)
        return run_experiment(**defaults)

    def test_rejects_zero_steps(self):
        with self.assertRaises(ValueError, msg="steps=0 should raise"):
            self._run(steps=0)

    def test_rejects_negative_steps(self):
        with self.assertRaises(ValueError):
            self._run(steps=-10)

    def test_rejects_zero_features(self):
        with self.assertRaises(ValueError):
            self._run(n_features=0)

    def test_rejects_zero_window(self):
        with self.assertRaises(ValueError):
            self._run(window=0)

    def test_rejects_negative_drift_at(self):
        with self.assertRaises(ValueError):
            self._run(drift_at=-1)


if __name__ == "__main__":
    unittest.main()
