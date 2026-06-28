"""Tests for drift detectors."""
import unittest
from drift_lens_lab.detectors import DDM, EWMADrift, PageHinkley


class TestPageHinkley(unittest.TestCase):
    def test_no_drift_stable(self):
        ph = PageHinkley(delta=0.005, lambda_=50.0)
        alarms = [ph.update(0.3) for _ in range(200)]
        # Stable sequence: no drift expected
        self.assertFalse(any(alarms))

    def test_detects_sudden_shift(self):
        ph = PageHinkley(delta=0.005, lambda_=5.0)
        # Feed stable data then spike
        for _ in range(50):
            ph.update(0.1)
        alarms = [ph.update(0.9) for _ in range(50)]
        self.assertTrue(any(alarms))

    def test_reset_clears_state(self):
        ph = PageHinkley(lambda_=5.0)
        for _ in range(50):
            ph.update(0.9)
        ph.reset()
        self.assertEqual(ph._n, 0)
        self.assertEqual(ph._sum, 0.0)


class TestDDM(unittest.TestCase):
    def test_warmup_no_alarm(self):
        ddm = DDM(min_samples=30)
        results = [ddm.update(0) for _ in range(29)]
        self.assertTrue(all(r == "ok" for r in results))

    def test_stable_zero_error_ok(self):
        ddm = DDM()
        results = [ddm.update(0) for _ in range(200)]
        # Perfect classifier should not trigger drift
        self.assertFalse(any(r == "drift" for r in results))

    def test_detects_error_surge(self):
        ddm = DDM(warning_level=2.0, drift_level=3.0, min_samples=30)
        # Establish baseline with low error
        for _ in range(100):
            ddm.update(0)
        # Surge to high error
        results = [ddm.update(1) for _ in range(100)]
        self.assertTrue(any(r in ("drift", "warning") for r in results))


class TestEWMADrift(unittest.TestCase):
    def test_stable_returns_ok(self):
        e = EWMADrift(min_samples=30)
        results = [e.update(0) for _ in range(200)]
        self.assertFalse(any(r == "drift" for r in results))

    def test_detects_sudden_error_increase(self):
        e = EWMADrift(alpha=0.3, z_thresh=2.0, min_samples=10)
        for _ in range(50):
            e.update(0)
        results = [e.update(1) for _ in range(50)]
        self.assertTrue(any(r == "drift" for r in results))


if __name__ == "__main__":
    unittest.main()
