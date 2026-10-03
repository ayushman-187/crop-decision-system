"""rank.py - multi-criteria ranking (TOPSIS) over several decision criteria.

Criteria
  SI      Suitability Index (benefit)  hybrid ML + rules, 0-100
  water   water fit (benefit)          fuzzy rainfall membership (+ optional irrigation)
  risk    risk (cost)                  1 - pessimistic SI / 100 (model uncertainty + input/season error)
  profit  profit index (benefit)       OPTIONAL, user supplied - the dataset contains no prices

Design: stage 1 HARD GATES remove agronomically poor crops (so no weight setting can recommend them),
        stage 2 TOPSIS trades the survivors off. The USER chooses the weights.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .scoring import trapezoid
from .uncertainty import pessimistic_si

PRESETS = {                      #                     SI    water  risk   profit
    "balanced":      dict(SI=0.40, water=0.20, risk=0.20, profit=0.20),
    "risk_averse":   dict(SI=0.30, water=0.15, risk=0.45, profit=0.10),
    "profit_driven": dict(SI=0.25, water=0.10, risk=0.15, profit=0.50),
    "water_scarce":  dict(SI=0.25, water=0.45, risk=0.20, profit=0.10),
}
CRITERIA = ["SI", "water", "risk", "profit"]
BENEFIT = {"SI": True, "water": True, "risk": False, "profit": True}
TIE_TOL = 0.05                   # a criterion whose spread is < 5% of its scale is treated as a tie
ELIGIBLE_MIN_SI = 3.0
GATE_SI_FRACTION = 0.30          # must reach >= 30% of the best crop's SI (and >= 9) ...
GATE_WATER = 0.20                # ... and rainfall(+irrigation) must be at least marginally workable


def water_fit(model, X: pd.DataFrame, irrigation_mm=0.0):
    """Fuzzy rainfall membership per crop (0-1). Irrigation adds to effective rainfall."""
    r = float(X["rainfall"].iloc[0]) + irrigation_mm
    return np.array([float(trapezoid(r, *model.fuzzy_.params_[c]["rainfall"])) for c in model.classes_])


def risk_score(model, X: pd.DataFrame):
    """Risk 0-1 = 1 - pessimistic SI / 100."""
    return np.clip(1 - pessimistic_si(model, X) / 100.0, 0, 1)


def build_matrix(model, X, irrigation_mm=0.0, profit_index: dict | None = None, shortlist=8):
    """Decision matrix (crops x criteria) of the crops that pass the hard gates."""
    SI = model.suitability_index(X)[0]
    wf, rk = water_fit(model, X, irrigation_mm), risk_score(model, X)
    gate_si = max(ELIGIBLE_MIN_SI * 3, GATE_SI_FRACTION * SI.max())
    keep = [i for i in np.argsort(-SI)[:shortlist] if SI[i] >= gate_si and wf[i] >= GATE_WATER]
    if not keep:
        keep = [int(np.argmax(SI))]
    pr = [(profit_index or {}).get(model.classes_[i], np.nan) for i in keep]
    return pd.DataFrame(dict(SI=SI[keep], water=wf[keep], risk=rk[keep], profit=pr), index=model.classes_[keep])


def topsis(D: pd.DataFrame, weights: dict) -> pd.Series:
    """TOPSIS with min-max normalisation (vector norm barely separates risk values that all sit near 0.9-1.0).
    Criteria with weight 0, or without data for every crop, are dropped. Returns closeness 0-1 (higher = better)."""
    w = {c: weights.get(c, 0.0) for c in CRITERIA if weights.get(c, 0.0) > 0 and D[c].notna().all()}
    tot = sum(w.values()); cols = list(w)
    M = D[cols].values.astype(float)
    rng_ = M.max(0) - M.min(0)
    scale = np.array([100.0 if c == "SI" else 1.0 for c in cols])
    rng_ = np.where(rng_ / scale < TIE_TOL, 0.0, rng_)                         # near-identical criterion -> tie
    M = np.where(rng_ > 1e-9, (M - M.min(0)) / np.where(rng_ > 1e-9, rng_, 1), 0.5)
    M = M * np.array([w[c] / tot for c in cols])
    ben = np.array([BENEFIT[c] for c in cols])
    ideal = np.where(ben, M.max(0), M.min(0)); anti = np.where(ben, M.min(0), M.max(0))
    d_pos = np.sqrt(((M - ideal) ** 2).sum(1)); d_neg = np.sqrt(((M - anti) ** 2).sum(1))
    return pd.Series(d_neg / (d_pos + d_neg + 1e-12), index=D.index, name="topsis")


def rank(D, weights):
    out = D.copy(); out["TOPSIS"] = topsis(D, weights)
    return out.sort_values("TOPSIS", ascending=False).assign(rank=lambda t: range(1, len(t) + 1))


def weight_sensitivity(D, weights, n=2000, spread=30, seed=0):
    """How stable is rank 1? Draw random weight vectors (Dirichlet) around the user's weights and report how often
    each crop wins -> 'recommendation robustness'."""
    rng = np.random.default_rng(seed)
    cols = [c for c in CRITERIA if weights.get(c, 0) > 0 and D[c].notna().all()]
    base = np.array([weights[c] for c in cols]); base = base / base.sum()
    wins = pd.Series(0, index=D.index, dtype=float)
    for w in rng.dirichlet(base * spread, size=n):
        wins[topsis(D, dict(zip(cols, w))).idxmax()] += 1
    return (wins / n).sort_values(ascending=False)
