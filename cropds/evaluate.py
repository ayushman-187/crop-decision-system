"""evaluate.py - Step 8: metrics, calibration, ablation, robustness and fairness analysis.

All experiments use STRATIFIED 5-fold cross-validation (optionally repeated). Anything that is tuned (alpha, fuzzy
width, gamma) is tuned inside the training fold only, so no number below leaks test data.

Public functions
    run_main(repeats)   LR / RF / XGB / rules-only / hybrid: accuracy, F1, calibration, noise, bias, missing inputs, per-crop
    run_ablation()      what each design choice of the hybrid contributes
Both return plain dicts / DataFrames; steps/step08_evaluation.py turns them into plots and the markdown report.
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from .data import FEATURES, TARGET, CROP_GROUP, load_dataset, perturb_inputs, clip_physical
from .model import HybridCropModel, make_lr, make_rf, make_xgb

NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4]
BIAS_SCENARIOS = {                                   # systematic sensor / forecast errors
    "rainfall -20%": lambda X: X.assign(rainfall=X.rainfall * 0.8),
    "rainfall +20%": lambda X: X.assign(rainfall=X.rainfall * 1.2),
    "temperature +3C": lambda X: X.assign(temperature=X.temperature + 3),
    "humidity -10 pts": lambda X: X.assign(humidity=X.humidity - 10),
    "pH +0.5": lambda X: X.assign(ph=X.ph + 0.5),
}
MISSING_SCENARIOS = {                                # inputs a farmer may simply not have
    "no soil test (N,P,K unknown)": ["N", "P", "K"],
    "no pH": ["ph"],
    "no humidity": ["humidity"],
}


# ----------------------------------------------------------------------------- metric helpers
def top_k_acc(S, y_idx, k):
    """Fraction of samples whose true crop is among the k highest-scored crops."""
    topk = np.argsort(-S, axis=1)[:, :k]
    return float((topk == y_idx[:, None]).any(axis=1).mean())


def ece(P, y_idx, bins=15):
    """Expected calibration error of the top-label confidence."""
    conf, pred = P.max(1), P.argmax(1)
    edges = np.linspace(0, 1, bins + 1); total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y_idx[m]).mean() - conf[m].mean())
    return float(total)


def brier(P, y_idx):
    """Multi-class Brier score (lower is better)."""
    Y = np.eye(P.shape[1])[y_idx]
    return float(((P - Y) ** 2).sum(1).mean())


def reliability_curve(P, y_idx, bins=10):
    conf, pred = P.max(1), P.argmax(1)
    edges = np.linspace(0, 1, bins + 1); rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() >= 5:
            rows.append((conf[m].mean(), (pred[m] == y_idx[m]).mean(), int(m.sum())))
    return pd.DataFrame(rows, columns=["confidence", "accuracy", "n"])


def metrics(S, P, y_idx):
    """S = scores used for ranking, P = probabilities used for log-loss / calibration."""
    pred = S.argmax(1)
    Pc = np.clip(P, 1e-9, 1); Pc = Pc / Pc.sum(1, keepdims=True)
    return dict(top1=float((pred == y_idx).mean()), top3=top_k_acc(S, y_idx, 3),
                macro_f1=float(f1_score(y_idx, pred, average="macro")),
                logloss=float(log_loss(y_idx, Pc, labels=range(P.shape[1]))),
                brier=brier(Pc, y_idx), ece=ece(Pc, y_idx))


# ----------------------------------------------------------------------------- one fold, all models
class FoldModels:
    """Fits every model on one training fold and exposes a uniform .predict(name, X) -> (scores, probs)."""
    NAMES = ["Logistic Regression", "Random Forest", "XGBoost", "Rules only (fuzzy)", "Hybrid"]

    def __init__(self, Xtr, ytr, classes, seed=42, with_xgb=True):
        self.classes = classes
        yi = np.searchsorted(classes, ytr.values)
        self.mean_ = Xtr.mean()
        self.lr = make_lr().fit(Xtr, yi)
        self.rf = make_rf(seed).fit(Xtr, yi)
        self.xgb = make_xgb(seed).fit(Xtr, yi) if with_xgb else None
        self.hyb = HybridCropModel(seed=seed).fit(Xtr, ytr)

    def predict(self, name, X):
        Xi = X[FEATURES].fillna(self.mean_)                      # baselines cannot handle NaN -> mean imputation
        if name == "Logistic Regression": P = self.lr.predict_proba(Xi); return P, P
        if name == "Random Forest":       P = self.rf.predict_proba(Xi); return P, P
        if name == "XGBoost":             P = self.xgb.predict_proba(Xi); return P, P
        if name == "Rules only (fuzzy)":  return self.hyb.fuzzy_.score(X), self.hyb.fuzzy_.predict_proba(X)
        if name == "Hybrid":              return self.hyb.suitability_index(X), self.hyb.predict_proba(X)
        raise KeyError(name)


# ----------------------------------------------------------------------------- MAIN experiment
def run_main(repeats=2, seed=42, verbose=True, with_xgb=True):
    df = load_dataset(); X, y = df[FEATURES], df[TARGET]
    classes = np.array(sorted(y.unique())); yi = np.searchsorted(classes, y.values)
    names = [n for n in FoldModels.NAMES if with_xgb or n != "XGBoost"]
    rng = np.random.default_rng(seed)
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=seed).split(X, y))

    rows_clean, rows_noise, rows_bias, rows_missing = [], [], [], []
    oofP = {n: np.zeros((len(y), len(classes))) for n in names}            # repeat 0 only
    oofS = {n: np.zeros((len(y), len(classes))) for n in names}
    oof_noisy_pred = {n: np.zeros(len(y), dtype=int) for n in ["Random Forest", "Hybrid"]}   # 20% noise, repeat 0
    alphas = []
    t0 = time.time()
    for k, (tr, te) in enumerate(splits):
        fm = FoldModels(X.iloc[tr], y.iloc[tr], classes, seed, with_xgb)
        alphas.append(fm.hyb.alpha_)
        Xte, yte = X.iloc[te].reset_index(drop=True), yi[te]
        # ---- clean metrics + out-of-fold probabilities (repeat 0)
        for n in names:
            S, P = fm.predict(n, Xte)
            rows_clean.append(dict(fold=k, model=n, **metrics(S, P, yte)))
            if k < 5:
                oofP[n][te], oofS[n][te] = P, S
        # ---- Gaussian input noise
        for s in NOISE_LEVELS[1:]:
            Xn = perturb_inputs(Xte, fm.hyb.std_, s, 1, rng)
            for n in names:
                S, P = fm.predict(n, Xn)
                rows_noise.append(dict(fold=k, model=n, sigma=s, top1=float((S.argmax(1) == yte).mean()),
                                       top3=top_k_acc(S, yte, 3)))
                if s == 0.2 and k < 5 and n in oof_noisy_pred:
                    oof_noisy_pred[n][te] = S.argmax(1)
        # ---- systematic bias
        for name, fn in BIAS_SCENARIOS.items():
            Xb = clip_physical(fn(Xte.copy()))
            for n in names:
                S, _ = fm.predict(n, Xb)
                rows_bias.append(dict(fold=k, model=n, scenario=name, top1=float((S.argmax(1) == yte).mean())))
        # ---- missing inputs (hybrid uses forests trained on the available columns)
        for name, drop in MISSING_SCENARIOS.items():
            Xm = Xte.copy(); Xm[drop] = np.nan
            keep = [f for f in FEATURES if f not in drop]
            subset_rf = fm.hyb.rf_for(keep)
            for n in names:
                S, _ = fm.predict(n, Xm)
                rows_missing.append(dict(fold=k, model=n + (" (mean-imputed)" if n in ("Random Forest", "Logistic Regression", "XGBoost") else ""),
                                         scenario=name, top1=float((S.argmax(1) == yte).mean())))
            rows_missing.append(dict(fold=k, model="Random Forest (trained on available cols)", scenario=name,
                                     top1=float((subset_rf.predict_proba(Xm[keep]).argmax(1) == yte).mean())))
        if verbose:
            print(f"  fold {k+1}/{len(splits)} done ({time.time()-t0:.0f}s) alpha={fm.hyb.alpha_:.1f}", flush=True)

    clean = pd.DataFrame(rows_clean)
    agg = clean.groupby("model")[["top1", "top3", "macro_f1", "logloss", "brier", "ece"]].agg(["mean", "std"]).reindex(names)
    noise = pd.concat([clean.assign(sigma=0.0)[["fold", "model", "sigma", "top1", "top3"]], pd.DataFrame(rows_noise)])
    bias = pd.DataFrame(rows_bias); missing = pd.DataFrame(rows_missing)

    # ---- significance: hybrid vs RF on pooled out-of-fold predictions of repeat 0 (McNemar exact test)
    ph = oofS["Hybrid"].argmax(1) == yi; pr = oofS["Random Forest"].argmax(1) == yi
    b, c = int((ph & ~pr).sum()), int((~ph & pr).sum())
    mcnemar_p = binomtest(b, b + c, 0.5).pvalue if (b + c) > 0 else 1.0
    boot = np.random.default_rng(0)
    diffs = [(ph[i].mean() - pr[i].mean()) for i in (boot.integers(0, len(yi), len(yi)) for _ in range(2000))]
    sig = dict(hybrid_only_correct=b, rf_only_correct=c, mcnemar_p=float(mcnemar_p),
               diff_mean=float(np.mean(diffs)), diff_ci95=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))])

    # ---- calibration curves from out-of-fold probabilities
    calib = {n: reliability_curve(oofP[n], yi) for n in names if n != "Rules only (fuzzy)"}

    # ---- fairness / limitations: per crop and per agronomic group
    pred_h = oofS["Hybrid"].argmax(1)
    per_crop = pd.DataFrame({"crop": classes[yi], "ok_h": pred_h == yi, "ok_rf": oofS["Random Forest"].argmax(1) == yi,
                             "ok_h_noisy": oof_noisy_pred["Hybrid"] == yi, "ok_rf_noisy": oof_noisy_pred["Random Forest"] == yi})
    per_crop["group"] = per_crop.crop.map(CROP_GROUP)
    by_crop = per_crop.groupby("crop")[["ok_h", "ok_rf", "ok_h_noisy", "ok_rf_noisy"]].mean()
    by_group = per_crop.groupby("group")[["ok_h", "ok_rf", "ok_h_noisy", "ok_rf_noisy"]].mean()
    by_group["n_crops"] = per_crop.groupby("group").crop.nunique()
    # confusion pairs
    cm = pd.crosstab(pd.Series(classes[yi], name="true"), pd.Series(classes[pred_h], name="pred"))
    pairs = [(a, b2, int(cm.loc[a, b2])) for a in cm.index for b2 in cm.columns if a != b2 and cm.loc[a, b2] > 0]
    pairs.sort(key=lambda t: -t[2])

    return dict(names=names, clean=clean, agg=agg, noise=noise, bias=bias, missing=missing, calib=calib,
                significance=sig, by_crop=by_crop, by_group=by_group, confusions=pairs[:10], alphas=alphas,
                oofP=oofP, oofS=oofS, y_idx=yi, classes=classes)


# ----------------------------------------------------------------------------- ABLATION
ABLATION_VARIANTS = {
    "ML only (Random Forest)":                  None,
    "Rules only (fuzzy)":                       None,
    "Hybrid - FULL":                            dict(),
    "Hybrid - w/o noise-aware tuning":          dict(noise_sigma=0.0),
    "Hybrid - fixed alpha = 0.5":               dict(fixed_alpha=0.5),
    "Hybrid - arithmetic mean (no Liebig min)": dict(fuzzy_agg="mean"),
    "Hybrid - equal feature weights":           dict(fuzzy_weight_power=0.0),
}


def run_ablation(seed=42, verbose=True):
    """Remove / change ONE design choice at a time and re-measure (5-fold CV, shared noisy test copies)."""
    df = load_dataset(); X, y = df[FEATURES], df[TARGET]
    classes = np.array(sorted(y.unique())); yi = np.searchsorted(classes, y.values)
    rng = np.random.default_rng(seed + 1)
    rows, t0 = [], time.time()
    for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)):
        Xte, yte = X.iloc[te].reset_index(drop=True), yi[te]
        views = {0.0: Xte}
        std = X.iloc[tr].std()
        for s in (0.2, 0.3):
            views[s] = perturb_inputs(Xte, std, s, 1, rng)
        rf = make_rf(seed).fit(X.iloc[tr], yi[tr])
        for name, kw in ABLATION_VARIANTS.items():
            if name.startswith("ML only"):
                score = lambda V: rf.predict_proba(V)
                P_of = score
            elif name.startswith("Rules only"):
                hyb = HybridCropModel(seed=seed).fit(X.iloc[tr], y.iloc[tr])
                score = lambda V, h=hyb: h.fuzzy_.score(V); P_of = lambda V, h=hyb: h.fuzzy_.predict_proba(V)
            else:
                hyb = HybridCropModel(seed=seed, **kw).fit(X.iloc[tr], y.iloc[tr])
                score = lambda V, h=hyb: h.suitability_index(V); P_of = lambda V, h=hyb: h.predict_proba(V)
            r = dict(fold=k, variant=name)
            S0 = score(views[0.0]); m = metrics(S0, P_of(views[0.0]), yte)
            r.update(top1=m["top1"], top3=m["top3"], macro_f1=m["macro_f1"], logloss=m["logloss"])
            for s in (0.2, 0.3):
                Sn = score(views[s]); r[f"top1_noise{int(s*100)}"] = float((Sn.argmax(1) == yte).mean())
                r[f"top3_noise{int(s*100)}"] = top_k_acc(Sn, yte, 3)
            if name == "Hybrid - FULL":
                r["alpha"] = hyb.alpha_
            rows.append(r)
        if verbose:
            print(f"  ablation fold {k+1}/5 ({time.time()-t0:.0f}s)", flush=True)
    d = pd.DataFrame(rows)
    return d.groupby("variant").agg(["mean", "std"]).reindex(list(ABLATION_VARIANTS)), d
