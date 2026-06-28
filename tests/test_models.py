"""Tests for online models."""
import unittest
from drift_lens_lab.models import OnlineLogisticRegression, Perceptron


class TestLogisticRegression(unittest.TestCase):
    def test_initial_predict_neutral(self):
        m = OnlineLogisticRegression(n_features=2)
        p = m.predict_proba([0.0, 0.0])
        self.assertAlmostEqual(p, 0.5, places=5)

    def test_learns_all_ones(self):
        m = OnlineLogisticRegression(n_features=3, lr=0.2)
        for _ in range(200):
            m.partial_fit([1.0, 1.0, 1.0], 1)
        self.assertGreater(m.predict_proba([1.0, 1.0, 1.0]), 0.8)

    def test_learns_all_zeros(self):
        m = OnlineLogisticRegression(n_features=3, lr=0.2)
        for _ in range(200):
            m.partial_fit([1.0, 1.0, 1.0], 0)
        self.assertLess(m.predict_proba([1.0, 1.0, 1.0]), 0.2)

    def test_predict_binary(self):
        m = OnlineLogisticRegression(n_features=2)
        self.assertIn(m.predict([1.0, -1.0]), {0, 1})

    def test_returns_prob_before_update(self):
        m = OnlineLogisticRegression(n_features=2)
        p0 = m.predict_proba([1.0, 0.0])
        p_ret = m.partial_fit([1.0, 0.0], 1)
        self.assertAlmostEqual(p0, p_ret, places=10)


class TestPerceptron(unittest.TestCase):
    def test_learns_separable(self):
        m = Perceptron(n_features=2)
        for _ in range(300):
            m.partial_fit([1.0, 0.0], 1)
            m.partial_fit([-1.0, 0.0], 0)
        self.assertEqual(m.predict([1.0, 0.0]), 1)
        self.assertEqual(m.predict([-1.0, 0.0]), 0)

    def test_pa_update_non_separable(self):
        m = Perceptron(n_features=2, passive_aggressive=True, C=0.5)
        for _ in range(100):
            m.partial_fit([0.0, 0.0], 1)
        # Should not crash and weights should remain finite
        for w in m.weights:
            self.assertTrue(-1e6 < w < 1e6)


if __name__ == "__main__":
    unittest.main()
