"""STEP 3 - Hybrid scoring engine: compare ML, rules-only and the hybrid (single 5-fold CV).
Run: python steps/step03_hybrid_scoring.py      (~1 min)   For the full evaluation use step08_evaluation.py"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from cropds.data import FEATURES, TARGET, load_dataset
from cropds import evaluate as ev

df = load_dataset(); X, y = df[FEATURES], df[TARGET]
classes = np.array(sorted(y.unique())); yi = np.searchsorted(classes, y.values)
rows = []
for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y), 1):
    fm = ev.FoldModels(X.iloc[tr], y.iloc[tr], classes)
    for n in fm.NAMES:
        S, P = fm.predict(n, X.iloc[te].reset_index(drop=True)); rows.append(dict(model=n, **ev.metrics(S, P, yi[te])))
    print(f"fold {k}/5  tuned alpha={fm.hyb.alpha_:.1f} fuzzy width={fm.hyb.width_} gamma={fm.hyb.gamma_}", flush=True)
print(pd.DataFrame(rows).groupby("model").mean().reindex(fm.NAMES).round(4).to_string())
