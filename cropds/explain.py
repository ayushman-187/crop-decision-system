"""explain.py - SHAP (data-driven) + fuzzy memberships (agronomy-driven) -> plain-language reasons.

SHAP is imported lazily and is OPTIONAL: if it is not installed (low-resource machines) the explanation falls
back to the forest's impurity feature importance for that field's available columns.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .data import FEATURES, LABEL, UNITS


class Explainer:
    def __init__(self, model):
        self.m = model
        self._tree = {}                                   # cols -> shap explainer (cached)

    # ---------------------------------------------------------------- SHAP
    def shap_for(self, row: pd.Series, crop: str) -> pd.Series:
        """Per-feature contribution to the probability of `crop` (positive = pushes towards the crop)."""
        cols = self.m.active_cols(row)
        rf = self.m.rf_for(cols)
        ci = list(self.m.classes_).index(crop)
        x = row.to_frame().T[cols].astype(float)
        try:
            import shap
            if tuple(cols) not in self._tree:
                self._tree[tuple(cols)] = shap.TreeExplainer(rf)
            sv = self._tree[tuple(cols)].shap_values(x)           # (n, features, classes)
            return pd.Series(np.asarray(sv)[0, :, ci], index=cols).reindex(FEATURES)
        except Exception:                                         # shap missing -> importance fallback
            imp = pd.Series(rf.feature_importances_, index=cols)
            return imp.reindex(FEATURES)

    # ---------------------------------------------------------------- reasons
    @staticmethod
    def limiting_factor(table: pd.DataFrame):
        """Liebig: the feature with the lowest membership (ties -> higher weight). None if all are ideal."""
        t = table.dropna(subset=["membership"]).sort_values(["membership", "weight"], ascending=[True, False])
        if t.empty or t.iloc[0].membership >= 0.999:
            return None
        return t.iloc[0]

    def reasons(self, row: pd.Series, crop: str, rank: int | None = None) -> dict:
        tbl = self.m.fuzzy_.explain(row, crop)
        sv = self.shap_for(row, crop)
        ok_rows = tbl.dropna(subset=["membership"])
        ideal = [r for r in ok_rows.itertuples() if r.membership >= 0.999]
        okay = [r for r in ok_rows.itertuples() if 0.5 <= r.membership < 0.999]
        bad = [r for r in ok_rows.itertuples() if r.membership < 0.5]
        missing = [r.feature for r in tbl.itertuples() if np.isnan(r.membership)]

        def fmt(r, strip=""):
            lo, hi = r.ideal
            return f"{LABEL[r.feature]} {r.value:g}{UNITS[r.feature]} (ideal {lo:g}-{hi:g}){strip}"
        parts = []
        if ideal: parts.append("in ideal range: " + "; ".join(fmt(r) for r in ideal[:4]))
        if okay:  parts.append("acceptable: " + "; ".join(fmt(r, r.status.replace("acceptable", "")) for r in okay))
        if bad:   parts.append("PROBLEM: " + "; ".join(fmt(r, r.status.replace("marginal", "").replace("unsuitable", "")) for r in bad))
        if missing: parts.append("not provided (ignored): " + ", ".join(LABEL[f] for f in missing))
        lf = self.limiting_factor(tbl)
        if lf is None:
            lim = "Limiting factor: none - every provided factor is inside its ideal range."
        else:
            direction = lf.status.split("(")[-1].strip(")") if "(" in lf.status else lf.status
            lim = f"Limiting factor: {LABEL[lf.feature]} is {direction}."
        top = sv.dropna().reindex(sv.dropna().abs().sort_values(ascending=False).index).head(3)
        shap_txt = ", ".join(f"{LABEL[f]} ({v:+.2f})" for f, v in top.items())
        head = crop.capitalize() + (f" ranked {rank}" if rank else "")
        text = f"{head}: " + " | ".join(parts) + f". {lim} Strongest model drivers: {shap_txt}."
        return dict(text=text, table=tbl, shap=sv, limiting=None if lf is None else lf.feature)

    # ---------------------------------------------------------------- figure
    def figure(self, row: pd.Series, crop: str):
        """Two-panel matplotlib figure: SHAP bars + fuzzy-fit bars. Returns (fig, reasons_dict)."""
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        r = self.reasons(row, crop)
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        s = r["shap"].dropna().sort_values()
        ax[0].barh([LABEL[f] for f in s.index], s.values, color=["#c62828" if v < 0 else "#2e7d32" for v in s.values])
        ax[0].set_title(f"Model drivers for '{crop}'", fontsize=10); ax[0].axvline(0, color="k", lw=.8)
        t = r["table"].set_index("feature")["membership"].reindex(FEATURES).fillna(0)
        ax[1].barh([LABEL[f] for f in t.index], t.values, color=plt.cm.RdYlGn(t.values))
        ax[1].set_xlim(0, 1.05); ax[1].set_title(f"Agronomic fit for '{crop}' (1 = ideal)", fontsize=10)
        plt.tight_layout()
        return fig, r


def radar_figure(model, row: pd.Series, crops: list):
    """Radar chart: fuzzy fit (0-1) of the field against each crop's ideal ranges, one polygon per crop."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    feats = [f for f in FEATURES if not pd.isna(row[f])]
    ang = np.linspace(0, 2 * np.pi, len(feats), endpoint=False).tolist(); ang += ang[:1]
    fig = plt.figure(figsize=(4.6, 4.6)); ax = plt.subplot(111, polar=True)
    colors = ["#2e7d32", "#ef6c00", "#1565c0", "#6a1b9a", "#c62828"]
    for k, crop in enumerate(crops):
        t = model.fuzzy_.explain(row, crop).set_index("feature")["membership"].reindex(feats).fillna(0)
        vals = t.tolist() + t.tolist()[:1]
        ax.plot(ang, vals, color=colors[k % 5], lw=2, label=crop); ax.fill(ang, vals, color=colors[k % 5], alpha=0.10)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels([LABEL[f] for f in feats], fontsize=8)
    ax.set_ylim(0, 1); ax.set_yticks([0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["", "", "", ""])
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.12), fontsize=8)
    ax.set_title("Fit to ideal ranges (outer edge = ideal)", fontsize=9, pad=14)
    plt.tight_layout()
    return fig
