"""scoring.py - knowledge-based fuzzy suitability engine (the "rules" half of the hybrid).

For every (crop, feature) a TRAPEZOIDAL membership function is learned:

    1.0 |        ______________
        |       /              \\
        |      /                \\
    0.0 |_____/                  \\_____
          a    b              c    d

  core    [b, c] = p10..p90 of the crop's observed values  -> "ideal range"  (membership 1.0)
  support [a, d] = core widened by `width` x core-span      -> membership decays linearly to 0
  EXPERT_OVERRIDES replaces the data-driven core by agronomic (e.g. FAO EcoCrop) ranges. Ships EMPTY on
  purpose: fill it from a source you can cite before claiming expert knowledge.

Crop score = 100 x weighted GEOMETRIC mean of feature memberships  (Liebig's law of the minimum:
one bad factor drags the whole crop down). `agg="mean"` gives the arithmetic version for the ablation study.

Missing inputs (NaN) are simply skipped and the remaining weights are re-normalised, so the engine still
works when a farmer has no soil-test report.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif

from .data import FEATURES, LABEL

# {"rice": {"ph": (5.0, 7.5)}}  -> core-range override; cite the source in your report
EXPERT_OVERRIDES: dict = {}


def trapezoid(x, a, b, c, d):
    """Vectorised trapezoidal membership in [0, 1]. NaN in -> NaN out."""
    x = np.asarray(x, dtype=float)
    up = np.where(b > a, (x - a) / (b - a + 1e-12), (x >= b).astype(float))
    down = np.where(d > c, (d - x) / (d - c + 1e-12), (x <= c).astype(float))
    return np.clip(np.minimum(np.minimum(up, down), 1.0), 0.0, 1.0)


class FuzzyCropScorer:
    def __init__(self, width=0.5, gamma=1.0, core=(10, 90), weight_power=0.5, agg="geometric"):
        self.width, self.gamma, self.core = width, gamma, core
        self.weight_power, self.agg = weight_power, agg

    # ------------------------------------------------------------------ learn the ranges
    def fit(self, X: pd.DataFrame, y):
        X = X[FEATURES].reset_index(drop=True)
        y = pd.Series(np.asarray(y)).reset_index(drop=True)
        self.classes_ = np.array(sorted(y.unique()))
        lo_q, hi_q = self.core
        self.params_ = {}                                       # crop -> feature -> (a, b, c, d)
        for crop in self.classes_:
            sub = X[y == crop]
            self.params_[crop] = {}
            for f in FEATURES:
                b, c = np.percentile(sub[f], [lo_q, hi_q])
                if crop in EXPERT_OVERRIDES and f in EXPERT_OVERRIDES[crop]:
                    b, c = EXPERT_OVERRIDES[crop][f]
                span = max(c - b, 0.05 * X[f].std())            # avoid zero-width cores
                self.params_[crop][f] = (b - self.width * span, b, c, c + self.width * span)
        # feature weights = how well the feature separates crops (ANOVA F), softened by a power
        F, _ = f_classif(X, y)
        w = np.power(np.nan_to_num(F) + 1.0, self.weight_power)
        self.weights_ = w / w.sum()
        return self

    # ------------------------------------------------------------------ scoring
    def memberships(self, X: pd.DataFrame):
        """-> array (n_samples, n_crops, n_features), values in [0,1] (NaN where an input is missing)."""
        X = X[FEATURES]
        M = np.zeros((len(X), len(self.classes_), len(FEATURES)))
        for i, crop in enumerate(self.classes_):
            for j, f in enumerate(FEATURES):
                M[:, i, j] = trapezoid(X[f].values, *self.params_[crop][f])
        return M

    def score(self, X: pd.DataFrame):
        """-> (n_samples, n_crops) suitability 0-100. Missing features are ignored (weights re-normalised)."""
        M = self.memberships(X)
        avail = ~np.isnan(M)
        M = np.clip(np.where(avail, M, 1.0), 1e-3, 1.0)         # floor keeps log finite
        w = self.weights_[None, None, :] * avail
        w = w / np.maximum(w.sum(axis=2, keepdims=True), 1e-12)
        if self.agg == "geometric":
            return 100.0 * np.exp((np.log(M) * w).sum(axis=2))
        return 100.0 * (M * w).sum(axis=2)

    def predict_proba(self, X):
        s = self.score(X) ** self.gamma + 1e-12
        return s / s.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------ explainability
    def explain(self, row: pd.Series, crop: str) -> pd.DataFrame:
        """Per-feature breakdown for ONE field and ONE crop."""
        out = []
        for j, f in enumerate(FEATURES):
            a, b, c, d = self.params_[crop][f]
            v = float(row[f])
            if np.isnan(v):
                out.append(dict(feature=f, value=np.nan, ideal=(round(b, 2), round(c, 2)), membership=np.nan,
                                status="not provided", weight=round(float(self.weights_[j]), 3)))
                continue
            m = float(trapezoid(v, a, b, c, d))
            status = ("ideal" if m >= 0.999 else "acceptable" if m >= 0.5 else
                      "marginal" if m > 0.05 else "unsuitable")
            direction = "" if m >= 0.999 else (" (too low)" if v < b else " (too high)")
            out.append(dict(feature=f, value=v, ideal=(round(b, 2), round(c, 2)), membership=round(m, 3),
                            status=status + direction, weight=round(float(self.weights_[j]), 3)))
        return pd.DataFrame(out)


def suitability_index(P_ml: np.ndarray, fuzzy_score: np.ndarray, alpha: float) -> np.ndarray:
    """The hybrid Suitability Index (0-100):   SI = alpha * 100 * P_ml  +  (1 - alpha) * fuzzy_score."""
    return alpha * 100.0 * P_ml + (1.0 - alpha) * fuzzy_score
