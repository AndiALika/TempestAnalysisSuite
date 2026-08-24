"""
tempest_report.py
=================
Export / reporting helpers for the TEMPEST Analysis Suite: PNG figures, CSV data,
JSON session files, and multi-section PDF reports (via reportlab).

GUI-free and unit-tested, like the other core modules.  A thesis needs
reproducible figures and tabular data, so every analysis screen can now emit
publication-ready artefacts instead of throw-away message boxes.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, Image)
    HAVE_REPORTLAB = True
except Exception:                       # pragma: no cover - optional dependency
    HAVE_REPORTLAB = False


# ─────────────────────────────────────────────────────────────────────────────
#  Figures & tabular data
# ─────────────────────────────────────────────────────────────────────────────
def save_figure(fig, path, dpi=150):
    """Save a Matplotlib figure as PNG, preserving the dark theme background."""
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return path


def export_csv(path, headers, rows):
    """Write ``rows`` (iterable of iterables) with a ``headers`` line to CSV."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  Session persistence
# ─────────────────────────────────────────────────────────────────────────────
def save_session(path, obj):
    """Persist a JSON-serialisable session object (e.g. a room layout)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    return path


def load_session(path):
    """Load a session object previously written by :func:`save_session`."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
#  PDF reports
# ─────────────────────────────────────────────────────────────────────────────
def build_pdf_report(path, title, subtitle="", meta=None, tables=None,
                     image_path=None, notes=None):
    """Build a multi-section PDF report.

    Parameters
    ----------
    path : str            output .pdf path
    title, subtitle : str
    meta : dict | None            key/value summary rendered as a 2-column table
    tables : list[tuple] | None   list of ``(heading, headers, rows)``
    image_path : str | None       a PNG to embed (e.g. the zone map)
    notes : list[str] | None      trailing paragraphs (methodology, disclaimers)
    """
    if not HAVE_REPORTLAB:              # pragma: no cover
        raise RuntimeError("reportlab is not installed; cannot build PDF report.")

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(path, pagesize=A4,
                            topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                            leftMargin=1.8 * cm, rightMargin=1.8 * cm)
    story = [Paragraph(title, styles["Title"])]
    if subtitle:
        story.append(Paragraph(subtitle, styles["Heading3"]))
    story.append(Spacer(1, 0.4 * cm))

    def _table(rows, col_header=False):
        t = Table(rows, hAlign="LEFT")
        style = [("FONTSIZE", (0, 0), (-1, -1), 9),
                 ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                 ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                 ("LEFTPADDING", (0, 0), (-1, -1), 5),
                 ("RIGHTPADDING", (0, 0), (-1, -1), 5)]
        if col_header:
            style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#21262d")),
                      ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                      ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
        t.setStyle(TableStyle(style))
        return t

    if meta:
        story.append(Paragraph("Summary", styles["Heading2"]))
        story.append(_table([[str(k), str(v)] for k, v in meta.items()]))
        story.append(Spacer(1, 0.4 * cm))

    if image_path and os.path.exists(image_path):
        story.append(Paragraph("Zone Map", styles["Heading2"]))
        story.append(Image(image_path, width=15 * cm, height=11 * cm))
        story.append(Spacer(1, 0.4 * cm))

    for heading, headers, rows in (tables or []):
        story.append(Paragraph(heading, styles["Heading2"]))
        story.append(_table([list(headers)] + [list(r) for r in rows],
                            col_header=True))
        story.append(Spacer(1, 0.4 * cm))

    for note in (notes or []):
        story.append(Paragraph(note, styles["Normal"]))
        story.append(Spacer(1, 0.2 * cm))

    doc.build(story)
    return path


__all__ = [
    "HAVE_REPORTLAB",
    "save_figure", "export_csv", "save_session", "load_session",
    "build_pdf_report",
]
