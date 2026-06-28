"""Tests for report generation."""
import os
import tempfile
import unittest
from drift_lens_lab.reports import ExperimentResult, to_html, to_markdown, write_report


def _make_result(**kw):
    defaults = dict(
        steps=100,
        drift_kind="sudden",
        drift_at=50,
        seed=42,
        model_name="logreg",
        detector_name="ddm",
        drift_events=[(55, "warning"), (70, "drift")],
        snapshots=[(i, 0.5 + i * 0.001, 0.7 - i * 0.001) for i in range(20)],
        final_accuracy=0.72,
        final_log_loss=0.58,
    )
    defaults.update(kw)
    return ExperimentResult(**defaults)


class TestMarkdown(unittest.TestCase):
    def test_contains_headers(self):
        md = to_markdown(_make_result())
        self.assertIn("# DriftLens Lab", md)
        self.assertIn("## Drift Events", md)

    def test_contains_drift_event(self):
        md = to_markdown(_make_result())
        self.assertIn("70", md)

    def test_no_events_placeholder(self):
        md = to_markdown(_make_result(drift_events=[]))
        self.assertIn("No drift events detected", md)


class TestHTML(unittest.TestCase):
    def test_is_valid_html_skeleton(self):
        h = to_html(_make_result())
        self.assertTrue(h.startswith("<!DOCTYPE html>"))
        self.assertIn("</html>", h)

    def test_contains_metrics(self):
        h = to_html(_make_result())
        self.assertIn("0.7200", h)

    def test_svg_present(self):
        h = to_html(_make_result())
        self.assertIn("<svg", h)


class TestWriteReport(unittest.TestCase):
    def test_write_html(self):
        result = _make_result()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sub", "report.html")
            write_report(result, path)
            self.assertTrue(os.path.exists(path))
            with open(path) as f:
                content = f.read()
            self.assertIn("<!DOCTYPE html>", content)

    def test_write_markdown(self):
        result = _make_result()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "report.md")
            write_report(result, path)
            self.assertTrue(os.path.exists(path))
            with open(path) as f:
                content = f.read()
            self.assertIn("# DriftLens Lab", content)


if __name__ == "__main__":
    unittest.main()
