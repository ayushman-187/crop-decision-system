"""Draws the 1-slide architecture diagram -> docs/architecture_slide.png.  Run: python steps/make_architecture_diagram.py"""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent.parent / "docs" / "architecture_slide.png"
fig, ax = plt.subplots(figsize=(15, 8.4)); ax.set_xlim(0, 150); ax.set_ylim(0, 84); ax.axis("off")
C = dict(inp="#e3f2fd", eng="#e8f5e9", core="#fff8e1", out="#fce4ec", ui="#ede7f6")

def box(x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=color, ec="#455a64", lw=1.4))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=10.5, fontweight="bold")
    ax.text(x + w / 2, y + h - 6.0, body, ha="center", va="top", fontsize=8.2, color="#263238", linespacing=1.35)

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16, lw=1.8, color="#37474f"))

ax.text(75, 81, "Crop Suitability Ranking & Agronomic Recommendation Decision System", ha="center", fontsize=15, fontweight="bold")
ax.text(75, 77.3, "Hybrid (rules + machine learning) decision support: rank  ·  trust  ·  explain  ·  what-if", ha="center", fontsize=10.5, color="#455a64")

box(2, 44, 26, 26, "1  INPUTS", "Climate: temperature,\nhumidity, rainfall\nSoil (optional): N, P, K, pH\nIrrigation (mm)\nPriorities + weights\nOptional profit index\n\nMissing values allowed", C["inp"])
box(35, 54, 30, 16, "2a  RULES ENGINE (fuzzy)", "Trapezoid membership per\ncrop x feature (p10-p90 core)\nweighted GEOMETRIC mean\n(Liebig minimum)", C["eng"])
box(35, 33, 30, 16, "2b  ML ENGINE", "Random Forest (300 trees)\nforest trained on available\ncolumns when inputs missing", C["eng"])
box(72, 40, 28, 28, "3  HYBRID SUITABILITY INDEX", "SI = a·100·P_ml + (1-a)·Fuzzy\n\na, fuzzy width, sharpness tuned\non inner CV with clean AND\nnoisy validation copies\n\nsmallest a within 0.1% of best\n-> rules stay active", C["core"])
box(107, 54, 41, 16, "4  TRUST", "confidence = entropy + margin\nborderline flag (gap < 0.20)\ntree disagreement, P10-P90 of SI\nunder 10% input error", C["out"])
box(107, 34, 41, 16, "5  MULTI-CRITERIA RANKING", "hard gates -> TOPSIS over\nSI · water fit · risk · profit\nuser-set weights + stability check", C["out"])
box(35, 8, 30, 20, "6  EXPLAIN", "SHAP drivers (per prediction)\n+ fuzzy fit per factor\n-> plain-language reason\n+ limiting factor", C["out"])
box(72, 8, 28, 20, "7  WHAT-IF ENGINE", "cheapest N / P / K / pH change\nthat lifts SI by +X points\nkg/ha, urea / DAP / MOP, lime\nfragility check + disclaimer", C["out"])
box(107, 8, 41, 20, "8  OUTPUTS  (Streamlit app)", "Top-5 table with confidence bars\nradar chart · SHAP chart\nlive what-if sliders\nPDF + CSV report\n(low-resource: 2 MB model, offline)", C["ui"])

arrow(28, 62, 35, 62); arrow(28, 52, 35, 44); arrow(65, 62, 72, 58); arrow(65, 41, 72, 48)
arrow(100, 58, 107, 62); arrow(100, 50, 107, 44); arrow(86, 40, 86, 28)
arrow(53, 33, 53, 28); arrow(127, 54, 127, 50); arrow(127, 34, 127, 28); arrow(100, 18, 107, 18)
ax.text(75, 2.5, "Honest scope: dataset = Kaggle Crop Recommendation (22 crops, 2200 rows, balanced, synthetic-like, India-centric). "
        "NPK relative -> kg/ha advice is indicative only.", ha="center", fontsize=8.5, style="italic", color="#b71c1c")
plt.tight_layout(); plt.savefig(OUT, dpi=140); print("saved", OUT)
