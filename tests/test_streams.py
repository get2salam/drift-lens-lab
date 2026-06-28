"""Tests for stream generators."""
import unittest
from drift_lens_lab.streams import StreamConfig, stream


class TestStreamDeterminism(unittest.TestCase):
    def _collect(self, kind, steps=50, **kw):
        cfg = StreamConfig(drift_kind=kind, seed=42, **kw)
        src = stream(cfg)
        return [next(src) for _ in range(steps)]

    def test_same_seed_same_output(self):
        a = self._collect("none")
        b = self._collect("none")
        self.assertEqual(a, b)

    def test_different_seeds_differ(self):
        cfg1 = StreamConfig(seed=1)
        cfg2 = StreamConfig(seed=99)
        s1 = next(stream(cfg1))
        s2 = next(stream(cfg2))
        self.assertNotEqual(s1[1], s2[1])

    def test_sudden_drift_changes_output(self):
        cfg_pre = StreamConfig(drift_kind="sudden", drift_at=1000, seed=7)
        cfg_post = StreamConfig(drift_kind="sudden", drift_at=0, seed=7)
        pre = [next(stream(cfg_pre))[2] for _ in range(100)]
        post = [next(stream(cfg_post))[2] for _ in range(100)]
        # After drift the label sequence should differ from before drift
        self.assertNotEqual(pre, post)

    def test_labels_binary(self):
        data = self._collect("gradual", steps=200)
        labels = {row[2] for row in data}
        self.assertTrue(labels.issubset({0, 1}))

    def test_step_increments(self):
        data = self._collect("none", steps=10)
        steps = [row[0] for row in data]
        self.assertEqual(steps, list(range(10)))

    def test_feature_length(self):
        cfg = StreamConfig(n_features=5, seed=1)
        _, x, _ = next(stream(cfg))
        self.assertEqual(len(x), 5)

    def test_invalid_drift_kind(self):
        cfg = StreamConfig(drift_kind="invalid")
        with self.assertRaises(ValueError):
            next(stream(cfg))


if __name__ == "__main__":
    unittest.main()
