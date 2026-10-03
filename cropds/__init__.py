"""cropds - Crop Suitability Ranking and Agronomic Recommendation Decision System.

Module map (read in this order):
    data.py         constants, dataset loading, input handling, example fields
    scoring.py      knowledge-based fuzzy suitability engine (rules) + Suitability Index blend
    model.py        ML models + the Hybrid model (ML + rules), save/load
    uncertainty.py  confidence, borderline flag, per-crop uncertainty
    rank.py         multi-criteria ranking (hard gates + TOPSIS + weight sensitivity)
    explain.py      SHAP + plain-language reasons
    whatif.py       minimum-change agronomic advice
    evaluate.py     metrics, calibration, ablation, robustness, fairness
    report.py       CSV / PDF report builders
"""
