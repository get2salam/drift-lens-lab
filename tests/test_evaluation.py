"""Tests for detector evaluation scoring."""
import unittest
from drift_lens_lab.cli import run_experiment
from drift_lens_lab.evaluation import score_detector


class TestScoreDetector(unittest.TestCase):
    def test_immediate_detection_has_zero_delay(self):
        score = score_detector([(150, "drift")], drift_at=150)
        self.assertTrue(score.detected)
        self.assertEqual(score.detection_delay, 0)
        self.assertEqual(score.false_alarms, 0)
        self.assertEqual(score.total_alarms, 1)

    def test_delayed_detection_within_window(self):
        score = score_detector([(180, "drift")], drift_at=150, max_delay=100)
        self.assertTrue(score.detected)
        self.assertEqual(score.detection_delay, 30)

    def test_missed_detection_outside_window(self):
        score = score_detector([(400, "drift")], drift_at=150, max_delay=100)
        self.assertFalse(score.detected)
        self.assertIsNone(score.detection_delay)
        self.assertEqual(score.false_alarms, 1)

    def test_alarm_before_drift_is_false_alarm(self):
        score = score_detector([(10, "drift"), (155, "drift")], drift_at=150, max_delay=100)
        self.assertTrue(score.detected)
        self.assertEqual(score.detection_delay, 5)
        self.assertEqual(score.false_alarms, 1)
        self.assertEqual(score.total_alarms, 2)

    def test_earliest_in_window_alarm_wins(self):
        score = score_detector(
            [(200, "drift"), (160, "drift")], drift_at=150, max_delay=100,
        )
        self.assertEqual(score.detection_delay, 10)

    def test_no_drift_at_means_everything_is_a_false_alarm(self):
        score = score_detector([(20, "drift"), (300, "drift")], drift_at=None)
        self.assertFalse(score.detected)
        self.assertIsNone(score.detection_delay)
        self.assertEqual(score.false_alarms, 2)

    def test_ignores_non_matching_event_kind(self):
        score = score_detector([(150, "warning")], drift_at=150, event_kind="drift")
        self.assertFalse(score.detected)
        self.assertEqual(score.total_alarms, 0)
        self.assertEqual(score.false_alarms, 0)

    def test_no_alarms_is_not_detected_and_not_a_false_alarm(self):
        score = score_detector([], drift_at=150)
        self.assertFalse(score.detected)
        self.assertEqual(score.total_alarms, 0)
        self.assertEqual(score.false_alarms, 0)


class TestDetectorScorePrecision(unittest.TestCase):
    def test_precision_none_with_no_alarms(self):
        score = score_detector([], drift_at=150)
        self.assertIsNone(score.precision)

    def test_precision_full_with_single_true_positive(self):
        score = score_detector([(150, "drift")], drift_at=150)
        self.assertAlmostEqual(score.precision, 1.0)

    def test_precision_zero_with_only_false_alarms(self):
        score = score_detector([(10, "drift"), (20, "drift")], drift_at=150, max_delay=5)
        self.assertAlmostEqual(score.precision, 0.0)


class TestScoreDetectorIntegration(unittest.TestCase):
    """Score a real end-to-end experiment run, not just synthetic event lists."""

    def test_sudden_drift_ddm_scores_are_well_formed(self):
        result = run_experiment(
            steps=400, drift_kind="sudden", drift_at=200,
            drift_width=50, n_features=10, seed=42,
            model_name="logreg", detector_name="ddm",
            window=100, report_path=None,
        )
        score = score_detector(result.drift_events, result.drift_at)
        self.assertIsInstance(score.detected, bool)
        self.assertGreaterEqual(score.false_alarms, 0)
        self.assertGreaterEqual(score.total_alarms, 0)
        if score.detected:
            self.assertGreaterEqual(score.detection_delay, 0)

    def test_no_drift_stream_never_reports_a_true_detection(self):
        result = run_experiment(
            steps=200, drift_kind="none", drift_at=100,
            drift_width=50, n_features=5, seed=1,
            model_name="logreg", detector_name="ddm",
            window=50, report_path=None,
        )
        score = score_detector(result.drift_events, result.drift_at, max_delay=0)
        self.assertFalse(score.detected)


if __name__ == "__main__":
    unittest.main()
