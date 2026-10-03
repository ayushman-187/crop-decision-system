"""report.py - downloadable CSV and PDF report of one analysis (used by the Streamlit app)."""
from __future__ import annotations
import io
from datetime import date
import numpy as np
import pandas as pd

from .data import FEATURES, LABEL, UNITS


def _latin(s) -> str:
    """reportlab's built-in fonts are Latin-1: replace the few non-Latin-1 symbols we use."""
    return (str(s).replace("≈", "~").replace("≥", ">=").replace("≤", "<=").replace("→", "->")
            .replace("–", "-").replace("—", "-").replace("±", "+/-").encode("latin-1", "replace").decode("latin-1"))


def build_csv(inputs: pd.Series, ranking: pd.DataFrame) -> bytes:
    """One row per ranked crop; the field's inputs are repeated as columns so the file is self-contained."""
    out = ranking.copy()
    for f in FEATURES:
        out[f"input_{f}"] = inputs[f]
    return out.to_csv(index=False).encode("utf-8")


def build_pdf(inputs: pd.Series, trust_tbl: pd.DataFrame, info: dict, topsis_tbl: pd.DataFrame, preset: str,
              explanation: str, advice: dict, disclaimer: str, figure_png: bytes | None = None) -> bytes:
    """A 1-2 page PDF: inputs, ranking, trust message, TOPSIS ranking, explanation, what-if advice, disclaimer."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    ss = getSampleStyleSheet(); buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.4 * cm, bottomMargin=1.4 * cm)
    H = lambda t: Paragraph(_latin(t), ss["Heading2"]); P = lambda t: Paragraph(_latin(t), ss["BodyText"])

    def table(df, widths=None):
        data = [list(map(_latin, df.columns))] + [[_latin(v) for v in r] for r in df.values.tolist()]
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2e7d32")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                               ("FONTSIZE", (0, 0), (-1, -1), 8), ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                               ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f8e9")])]))
        return t

    el = [Paragraph("Crop Suitability &amp; Agronomic Recommendation Report", ss["Title"]), P(f"Generated {date.today().isoformat()}"), Spacer(1, 6)]
    el += [H("1. Field inputs")]
    rows = [[LABEL[f].capitalize(), "not provided" if pd.isna(inputs[f]) else f"{inputs[f]:g} {UNITS[f]}".strip()] for f in FEATURES]
    el += [table(pd.DataFrame(rows, columns=["Parameter", "Value"]), [6 * cm, 5 * cm])]
    el += [H("2. Suitability ranking"), table(trust_tbl[["rank", "crop", "SI", "prob", "SI_p10", "SI_p90"]].rename(
        columns={"SI": "Suitability Index", "prob": "Model prob.", "SI_p10": "SI low (P10)", "SI_p90": "SI high (P90)"}).round(2)),
           Spacer(1, 4), P(f"<b>Trust:</b> {info['message']}")]
    el += [H(f"3. Priority-weighted ranking (TOPSIS, preset: {preset})")]
    t = topsis_tbl.reset_index().rename(columns={"index": "crop"}).round(3).dropna(axis=1, how="all")   # drop empty profit column
    el += [table(t[[c for c in ["rank", "crop", "TOPSIS", "SI", "water", "risk", "profit"] if c in t.columns]])]
    el += [H("4. Why the top crop?"), P(explanation)]
    if figure_png:
        el += [Image(io.BytesIO(figure_png), width=17 * cm, height=5.9 * cm)]
    el += [H("5. What-if advice (minimum change)")]
    for crop, lines in advice.items():
        el += [P(f"<b>{crop.capitalize()}:</b> " + (" ".join(lines) if lines else "no controllable change reaches the target."))]
    el += [Spacer(1, 6), P(f"<i>{disclaimer}</i>")]
    doc.build(el)
    return buf.getvalue()
