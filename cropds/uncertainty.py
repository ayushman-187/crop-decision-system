"""uncertainty.py - how much should the user trust the ranking?

Field level : entropy (how spread the probability is) + margin (gap between rank 1 and rank 2)
Crop level  : disagreement of the forest's trees about that crop + sensitivity of its SI to input error
Borderline  : rank 1 and rank 2 too close  ->  "two crops are nearly equal, consider both"
All functions accept fields with missing inputs (NaN): they automatically use the forest for the available columns.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .data import FEATURES, perturb_inputs
from .scoring import suitability_index

BORDERLINE_MARGIN = 0.20          # p1 - p2 below this -> borderline (validated in evaluate.py / step 5)


def entropy_norm(P):
    """Normalised Shannon entropy in [0,1] (0 = certain, 1 = completely undecided)."""
    P = np.clip(P, 1e-12, 1)
    return -(P * np.log(P)).sum(axis=1) / np.log(P.shape[1])


def field_confidence(P):
    """Confidence 0-100 = 50% (1 - entropy) + 50% margin between the best two crops. Returns (conf, entropy, margin)."""
    Hn = entropy_norm(P)
    top2 = np.sort(P, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    return 100 * (0.5 * (1 - Hn) + 0.5 * margin), Hn, margin


def _tree_probs(model, X):
    """per-tree probabilities (trees, n, crops) using the forest that matches the available columns."""
    cols = model.active_cols(X.iloc[0])
    rf = model.rf_for(cols)
    Xv = np.asarray(X[cols], dtype=np.float32)
    return np.stack([t.predict_proba(Xv) for t in rf.estimators_])


def tree_std(model, X):
    """Std of per-tree probabilities for every crop -> (n, n_crops). Epistemic uncertainty."""
    return _tree_probs(model, X).std(axis=0)


def si_spread(model, X, sigma=0.10, n=60, seed=0):
    """Monte-Carlo SI of each crop under input error. Returns (p10, p50, p90), each (n_samples, n_crops)."""
    rng = np.random.default_rng(seed)
    noisy = perturb_inputs(X, model.std_, sigma, n, rng)
    SI = model.suitability_index(noisy).reshape(n, len(X), -1)
    return np.percentile(SI, 10, axis=0), np.percentile(SI, 50, axis=0), np.percentile(SI, 90, axis=0)


def pessimistic_si(model, X, sigma=0.10, n=60, q=10, seed=1):
    """'Downside' suitability of every crop for ONE field: the q-th percentile, over simulated input error, of a
    lower-confidence-bound SI:  SI_lcb = alpha*100*max(P_ml - tree_std, 0) + (1-alpha)*fuzzy.
    Model uncertainty (tree disagreement) AND input/season error both pull it down."""
    rng = np.random.default_rng(seed)
    noisy = perturb_inputs(X, model.std_, sigma, n, rng)
    probs = _tree_probs(model, noisy)
    lcb = np.clip(probs.mean(0) - probs.std(0), 0, 1)
    si = suitability_index(lcb, model.fuzzy_.score(noisy), model.alpha_)
    return np.percentile(si, q, axis=0)


def trust_report(model, X: pd.DataFrame, top_k=5):
    """Trust table + summary for ONE field (X has exactly 1 row)."""
    P = model.predict_proba(X); SI = model.suitability_index(X)[0]
    conf, Hn, margin = field_confidence(P)
    ts = tree_std(model, X)[0]
    p10, _, p90 = si_spread(model, X)
    order = np.argsort(-SI)[:top_k]
    tbl = pd.DataFrame([dict(rank=r, crop=model.classes_[i], SI=round(float(SI[i]), 1), prob=round(float(P[0, i]), 3),
                             tree_disagreement=round(float(ts[i]), 3), SI_p10=round(float(p10[0, i]), 1),
                             SI_p90=round(float(p90[0, i]), 1)) for r, i in enumerate(order, 1)])
    borderline = bool(margin[0] < BORDERLINE_MARGIN)
    band = "HIGH" if conf[0] >= 70 else "MEDIUM" if conf[0] >= 40 else "LOW"
    msg = (f"Confidence {conf[0]:.0f}/100 ({band}). " +
           (f"BORDERLINE: {tbl.crop[0]} and {tbl.crop[1]} are nearly equal (gap {margin[0]*100:.0f} pts) - consider both."
            if borderline else f"{tbl.crop[0]} clearly leads {tbl.crop[1]} by {margin[0]*100:.0f} pts."))
    return tbl, dict(confidence=float(conf[0]), entropy=float(Hn[0]), margin=float(margin[0]),
                     borderline=borderline, band=band, message=msg)
