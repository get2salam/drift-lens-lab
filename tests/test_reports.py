"""Tests for report generation."""
import os
import tempfile
import unittest
from drift_lens_lab.reports import (
    ExperimentResult,
    _md_escape_cell,
    _svg_line,
    to_html,
    to_markdown,
    write_report,
)


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


class TestSVGColorSecurity(unittest.TestCase):
    """_svg_line must reject color values that could inject content into SVG/HTML."""

    _VALS = [0.1, 0.5, 0.9]

    def test_hex_color_accepted(self):
        svg = _svg_line(self._VALS, color="#4f8ef7")
        self.assertIn("stroke=", svg)

    def test_short_hex_accepted(self):
        _svg_line(self._VALS, color="#abc")

    def test_named_color_accepted(self):
        _svg_line(self._VALS, color="steelblue")

    def test_rgb_accepted(self):
        _svg_line(self._VALS, color="rgb(79, 142, 247)")

    def test_rgba_accepted(self):
        _svg_line(self._VALS, color="rgba(79, 142, 247, 0.8)")

    def test_injection_with_closing_quote_raises(self):
        with self.assertRaises(ValueError):
            _svg_line(self._VALS, color='red"/><script>alert(1)</script>')

    def test_injection_with_angle_bracket_raises(self):
        with self.assertRaises(ValueError):
            _svg_line(self._VALS, color="<script>")

    def test_injection_with_semicolon_raises(self):
        with self.assertRaises(ValueError):
            _svg_line(self._VALS, color="red; background: url(evil)")

    def test_safe_color_appears_correctly_in_output(self):
        svg = _svg_line(self._VALS, color="#4f8ef7")
        self.assertIn('stroke="#4f8ef7"', svg)


class TestHTMLSecurity(unittest.TestCase):
    """HTML report must not render user-controlled strings as raw markup."""

    def test_xss_in_drift_kind_is_escaped(self):
        h = to_html(_make_result(drift_kind='<script>alert(1)</script>'))
        self.assertNotIn("<script>alert(1)</script>", h)
        self.assertIn("&lt;script&gt;", h)

    def test_xss_in_model_name_is_escaped(self):
        h = to_html(_make_result(model_name='"><img src=x onerror=alert(1)>'))
        # Raw <img tag must not appear; html.escape() must have converted it
        self.assertNotIn("<img src=x", h)
        self.assertIn("&lt;img", h)

    def test_xss_in_detector_name_is_escaped(self):
        h = to_html(_make_result(detector_name="<b onmouseover=alert(1)>ddm</b>"))
        self.assertNotIn("<b onmouseover", h)

    def test_xss_in_drift_event_kind_is_escaped(self):
        payload = '<img src=x onerror=alert(1)>'
        h = to_html(_make_result(drift_events=[(10, payload)]))
        self.assertNotIn("<img src=x", h)
        self.assertIn("&lt;img", h)


class TestMarkdownSecurity(unittest.TestCase):
    """Markdown reports must neutralize pipe injection in table cells."""

    def test_md_escape_cell_escapes_pipe(self):
        self.assertEqual(_md_escape_cell("foo|bar"), r"foo\|bar")

    def test_md_escape_cell_escapes_backslash(self):
        self.assertEqual(_md_escape_cell("foo\\bar"), r"foo\\bar")

    def test_md_escape_cell_passthrough_safe(self):
        self.assertEqual(_md_escape_cell("logreg"), "logreg")

    def test_pipe_in_model_name_is_escaped_in_markdown(self):
        md = to_markdown(_make_result(model_name="log|reg"))
        self.assertNotIn("| log|reg |", md)
        self.assertIn(r"log\|reg", md)

    def test_pipe_in_detector_name_is_escaped_in_markdown(self):
        md = to_markdown(_make_result(detector_name="d|dm"))
        self.assertNotIn("| d|dm |", md)
        self.assertIn(r"d\|dm", md)

    def test_pipe_in_drift_kind_is_escaped_in_markdown(self):
        md = to_markdown(_make_result(drift_kind="sudden|gradual"))
        self.assertNotIn("| `sudden|gradual` |", md)
        self.assertIn(r"sudden\|gradual", md)


if __name__ == "__main__":
    unittest.main()
