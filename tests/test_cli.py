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


if __name__ == "__main__":
    unittest.main()
