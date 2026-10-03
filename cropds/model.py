"""model.py - machine-learning models and the Hybrid model (ML + fuzzy rules).

HybridCropModel
    SI(crop) = alpha * 100 * P_ml(crop) + (1 - alpha) * FuzzyScore(crop)

    * alpha, fuzzy width and fuzzy sharpness (gamma) are tuned on INNER cross-validation folds of the training
      data only (no leakage into the test fold).
    * tuning looks at clean AND noise-corrupted validation copies (`noise_sigma`) so the blend favours settings
      that survive cheap-sensor error ("noise-aware tuning").
    * among alphas within 0.1% of the best accuracy the SMALLEST is kept, so the explainable fuzzy part stays active.
    * missing inputs (NaN) are supported: the fuzzy part skips them and the ML part uses a forest trained on
      only the available columns (cached).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import FEATURES, ROOT, perturb_inputs, load_dataset, TARGET
from .scoring import FuzzyCropScorer, suitability_index

MODEL_PATH = ROOT / "models" / "hybrid_model.joblib"


# ----------------------------------------------------------------------------- baseline ML models
def make_rf(seed=42, n_estimators=300):
    return RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1)


def make_lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))


def make_xgb(seed=42):
    from xgboost import XGBClassifier            # imported lazily: the app does not need it
    return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
                         eval_metric="mlogloss", n_jobs=-1, random_state=seed)


# ----------------------------------------------------------------------------- hybrid model
class HybridCropModel:
    def __init__(self, alphas=np.linspace(0, 1, 11), widths=(0.25, 0.5, 0.75, 1.0, 1.5),
                 gammas=(1, 2, 4, 8, 16), inner_splits=3, seed=42, noise_sigma=0.15,
                 fixed_alpha=None, fuzzy_agg="geometric", fuzzy_weight_power=0.5):
        self.alphas, self.widths, self.gammas = alphas, widths, gammas
        self.inner_splits, self.seed, self.noise_sigma = inner_splits, seed, noise_sigma
        self.fixed_alpha, self.fuzzy_agg, self.fuzzy_weight_power = fixed_alpha, fuzzy_agg, fuzzy_weight_power
        self._subset_cache: dict = {}

    # pickling: the cache of subset forests is rebuilt on demand
    def __getstate__(self):
        s = self.__dict__.copy(); s["_subset_cache"] = {}; return s

    @staticmethod
    def _top1(S, classes, y):
        return (classes[S.argmax(1)] == np.asarray(y)).mean()

    # ------------------------------------------------------------------ training
    def fit(self, X: pd.DataFrame, y):
        X = X[FEATURES].reset_index(drop=True); y = pd.Series(np.asarray(y))
        self.Xtr_, self.ytr_ = X.copy(), y.copy()
        self.classes_ = np.array(sorted(y.unique())); self.std_ = X.std()
        rng = np.random.default_rng(self.seed)
        idx = np.searchsorted(self.classes_, y.values)
        folds = list(StratifiedKFold(self.inner_splits, shuffle=True, random_state=self.seed).split(X, y))
        views = {"clean": X.copy()}
        if self.noise_sigma > 0:
            views["noisy"] = perturb_inputs(X, self.std_, self.noise_sigma, 1, rng)

        # inner out-of-fold ML probabilities (clean + noisy validation views)
        oof_ml = {k: np.zeros((len(X), len(self.classes_))) for k in views}
        for tr, va in folds:
            rf = make_rf(self.seed).fit(X.iloc[tr], y.iloc[tr])
            for k, V in views.items():
                oof_ml[k][va] = rf.predict_proba(V.iloc[va])

        def ll(P): return -np.log(P[np.arange(len(y)), idx] + 1e-12).mean()
        def norm(S, g): P = S ** g + 1e-12; return P / P.sum(1, keepdims=True)

        # 1) tune fuzzy width + gamma (mean log-loss over clean + noisy views)
        best = (1e9, None)
        for w in self.widths:
            S = {k: np.zeros_like(oof_ml["clean"]) for k in views}
            for tr, va in folds:
                fz = FuzzyCropScorer(width=w, agg=self.fuzzy_agg, weight_power=self.fuzzy_weight_power
                                     ).fit(X.iloc[tr], y.iloc[tr])
                for k, V in views.items():
                    S[k][va] = fz.score(V.iloc[va])
            for g in self.gammas:
                loss = np.mean([ll(norm(S[k], g)) for k in views])
                if loss < best[0]:
                    best = (loss, (w, g, S))
        _, (self.width_, self.gamma_, S_oof) = best

        # 2) tune alpha (mean top-1 over views; smallest alpha within 0.1% of the best -> keeps rules active)
        scores = []
        for a in self.alphas:
            acc = np.mean([self._top1(suitability_index(oof_ml[k], S_oof[k], a), self.classes_, y) for k in views])
            nll = np.mean([ll(a * oof_ml[k] + (1 - a) * norm(S_oof[k], self.gamma_)) for k in views])
            scores.append((round(acc, 5), -nll, a))
        best_acc = max(sc[0] for sc in scores)
        self.alpha_ = min(sc[2] for sc in scores if sc[0] >= best_acc - 0.001)
        if self.fixed_alpha is not None:                       # used by the ablation study
            self.alpha_ = float(self.fixed_alpha)
        self.alpha_curve_ = [(a, acc, nl) for acc, nl, a in scores]

        # 3) refit the final components on all training data
        self.rf_ = make_rf(self.seed).fit(X, y)
        self.fuzzy_ = FuzzyCropScorer(width=self.width_, gamma=self.gamma_, agg=self.fuzzy_agg,
                                      weight_power=self.fuzzy_weight_power).fit(X, y)
        return self

    # ------------------------------------------------------------------ helpers for missing inputs
    def rf_for(self, cols):
        """Forest trained on only `cols` (cached). Full feature set -> the main forest."""
        cols = tuple(cols)
        if cols == tuple(FEATURES):
            return self.rf_
        if cols not in self._subset_cache:
            self._subset_cache[cols] = make_rf(self.seed, n_estimators=150).fit(self.Xtr_[list(cols)], self.ytr_)
        return self._subset_cache[cols]

    @staticmethod
    def active_cols(row: pd.Series):
        return [f for f in FEATURES if not pd.isna(row[f])]

    # ------------------------------------------------------------------ inference
    def components(self, X: pd.DataFrame):
        """-> (P_ml, fuzzy_score), each (n, n_crops). Rows may contain NaN = 'not provided'."""
        X = X[FEATURES]
        S = self.fuzzy_.score(X)
        if not X.isna().any().any():
            return self.rf_.predict_proba(X), S
        P = np.zeros_like(S)
        masks = X.notna().apply(tuple, axis=1)
        for mask in masks.unique():
            rows = np.where((masks == mask).values)[0]
            cols = [f for f, keep in zip(FEATURES, mask) if keep]
            P[rows] = self.rf_for(cols).predict_proba(X.iloc[rows][cols])
        return P, S

    def suitability_index(self, X):
        P_ml, S = self.components(X)
        return suitability_index(P_ml, S, self.alpha_)

    def predict_proba(self, X):
        P_ml, S = self.components(X)
        Pf = S ** self.gamma_ + 1e-12; Pf /= Pf.sum(1, keepdims=True)
        return self.alpha_ * P_ml + (1 - self.alpha_) * Pf

    def predict(self, X):
        return self.classes_[self.suitability_index(X).argmax(1)]


# ----------------------------------------------------------------------------- save / load
def save_model(model: HybridCropModel, path=MODEL_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)


def load_or_train(path=MODEL_PATH, retrain=False) -> HybridCropModel:
    """Load the shipped model (fast start-up, ~1 s). If it is missing or was saved with an incompatible
    scikit-learn version, train a fresh one (~10 s) and save it."""
    if not retrain and Path(path).exists():
        try:
            return joblib.load(path)
        except Exception:
            pass
    df = load_dataset()
    m = HybridCropModel().fit(df[FEATURES], df[TARGET])
    try:
        save_model(m, path)
    except OSError:
        pass
    return m
