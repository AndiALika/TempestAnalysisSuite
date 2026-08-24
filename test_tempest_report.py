"""
test_tempest_report.py
=====================
Validation suite for ``tempest_report`` (stdlib unittest only).
"""

import os
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tempest_report as rep


class TestCsvAndSession(unittest.TestCase):
    def test_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.csv")
            rep.export_csv(p, ["a", "b"], [[1, 2], [3, 4]])
            with open(p, encoding="utf-8") as f:
                text = f.read()
        self.assertIn("a,b", text)
        self.assertIn("3,4", text)

    def test_session_roundtrip(self):
        obj = {"room": [10, 8], "devices": [{"type": "CRT Monitor", "x": 1}]}
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.json")
            rep.save_session(p, obj)
            back = rep.load_session(p)
        self.assertEqual(obj, back)


class TestFigure(unittest.TestCase):
    def test_save_png(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2], [0, 1, 4])
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "f.png")
            rep.save_figure(fig, p)
            self.assertTrue(os.path.exists(p))
            self.assertGreater(os.path.getsize(p), 0)
        plt.close(fig)


@unittest.skipUnless(rep.HAVE_REPORTLAB, "reportlab not installed")
class TestPdf(unittest.TestCase):
    def test_build_pdf(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "r.pdf")
            rep.build_pdf_report(
                p, "TEMPEST Report", "Test subtitle",
                meta={"Room": "10 x 8 m", "Wall SE": "20 dB"},
                tables=[("Devices", ["#", "Type"], [[1, "CRT Monitor"]])],
                notes=["Representative model — not a calibrated measurement."])
            self.assertTrue(os.path.exists(p))
            self.assertGreater(os.path.getsize(p), 500)   # a real PDF


if __name__ == "__main__":
    unittest.main(verbosity=2)
