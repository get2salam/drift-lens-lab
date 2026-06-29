"""Tests for rolling metrics."""
import math
import unittest
from drift_lens_lab.metrics import MetricsTracker, RollingAccuracy, RollingLogLoss


class TestRollingAccuracy(unittest.TestCase):
    def test_empty_is_none(self):
        r = RollingAccuracy(window=10)
        self.assertIsNone(r.value)

    def test_perfect_predictions(self):
        r = RollingAccuracy(window=20)
        for _ in range(20):
            r.update(1, 1)
        self.assertAlmostEqual(r.value, 1.0)

    def test_all_wrong(self):
        r = RollingAccuracy(window=20)
        for _ in range(20):
            r.update(0, 1)
        self.assertAlmostEqual(r.value, 0.0)

    def test_window_slides(self):
        r = RollingAccuracy(window=10)
        for _ in range(10):
            r.update(0, 1)   # all wrong
        for _ in range(10):
            r.update(1, 1)   # all correct
        self.assertAlmostEqual(r.value, 1.0)


class TestRollingLogLoss(unittest.TestCase):
    def test_perfect_prob_low_loss(self):
        r = RollingLogLoss(window=20)
        for _ in range(20):
            r.update(0.9999, 1)
        self.assertLess(r.value, 0.01)

    def test_worst_prob_high_loss(self):
        r = RollingLogLoss(window=20)
        for _ in range(20):
            r.update(0.0001, 1)
        self.assertGreater(r.value, 5.0)


class TestMetricsTracker(unittest.TestCase):
    def test_snapshots_taken(self):
        t = MetricsTracker(window=100)
        for i in range(100):
            t.update(i, 0.7, 1, 1)
        self.assertGreater(len(t.snapshots), 0)

    def test_current_accuracy_in_range(self):
        t = MetricsTracker(window=50)
        for i in range(50):
            t.update(i, 0.6, 1, 1)
        self.assertIsNotNone(t.current_accuracy)
        self.assertGreaterEqual(t.current_accuracy, 0.0)
        self.assertLessEqual(t.current_accuracy, 1.0)


class TestRollingAccuracyRunningSum(unittest.TestCase):
    """Invariant: running _correct must equal sum(buf) for every update."""

    def test_matches_recompute_through_window_slide(self):
        r = RollingAccuracy(window=10)
        import random
        rng = random.Random(0)
        for _ in range(30):
            r.update(rng.randint(0, 1), rng.randint(0, 1))
            self.assertEqual(r._correct, sum(r._buf))
            self.assertAlmostEqual(r.value, sum(r._buf) / len(r._buf), places=12)

    def test_zero_window_raises(self):
        with self.assertRaises(ValueError):
            RollingAccuracy(window=0)

    def test_negative_window_raises(self):
        with self.assertRaises(ValueError):
            RollingAccuracy(window=-1)


class TestRollingLogLossRunningSum(unittest.TestCase):
    """Invariant: running _sum must equal sum(buf) for every update."""

    def test_matches_recompute_through_window_slide(self):
        r = RollingLogLoss(window=10)
        import random
        rng = random.Random(1)
        for _ in range(30):
            r.update(rng.random(), rng.randint(0, 1))
            self.assertAlmostEqual(r._sum, sum(r._buf), places=10)
            self.assertAlmostEqual(r.value, sum(r._buf) / len(r._buf), places=10)

    def test_zero_window_raises(self):
        with self.assertRaises(ValueError):
            RollingLogLoss(window=0)


if __name__ == "__main__":
    unittest.main()
