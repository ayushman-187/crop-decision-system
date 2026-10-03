"""STEP 2 - Exploratory data analysis.   Run: python steps/step02_eda.py   ->  outputs/step2/*.png + console summary"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_selection import f_classif
from cropds.data import FEATURES, TARGET, load_dataset

OUT = Path(__file__).resolve().parent.parent / "outputs" / "step2"; OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")
df = load_dataset()

# 1. audit --------------------------------------------------------------------------------------------
print("Shape:", df.shape, "| missing:", int(df.isna().sum().sum()), "| duplicates:", int(df.duplicated().sum()))
print("Crops:", df[TARGET].nunique(), "| rows per crop:", sorted(df[TARGET].value_counts().unique()))
print(df[FEATURES].describe().T.round(2).to_string())

# 2. distributions ------------------------------------------------------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, f in zip(axes.ravel(), FEATURES):
    sns.histplot(df[f], kde=True, ax=ax, color="#2e7d32"); ax.set_title(f"{f} (skew={df[f].skew():.2f})")
axes.ravel()[-1].axis("off"); plt.suptitle("Feature distributions", y=1.02); plt.tight_layout()
plt.savefig(OUT / "eda_1_distributions.png", dpi=130, bbox_inches="tight"); plt.close()

# 3. correlation --------------------------------------------------------------------------------------
corr = df[FEATURES].corr()
plt.figure(figsize=(7, 5.5)); sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0, square=True)
plt.title("Feature correlation"); plt.tight_layout(); plt.savefig(OUT / "eda_2_correlation.png", dpi=130); plt.close()
off = corr.where(~np.eye(len(corr), dtype=bool)).abs().stack()
print("\nStrongest correlation:", off.idxmax(), round(off.max(), 2))

# 4. per-crop ranges (basis of the fuzzy engine) ------------------------------------------------------
fig, axes = plt.subplots(4, 2, figsize=(15, 18))
for ax, f in zip(axes.ravel(), FEATURES):
    order = df.groupby(TARGET)[f].median().sort_values().index
    sns.boxplot(data=df, y=TARGET, x=f, order=order, ax=ax, palette="viridis", fliersize=2); ax.set_title(f"{f} range per crop"); ax.set_ylabel("")
axes.ravel()[-1].axis("off"); plt.tight_layout(); plt.savefig(OUT / "eda_3_per_crop_ranges.png", dpi=110); plt.close()

# 5. crop fingerprint ---------------------------------------------------------------------------------
med = df.groupby(TARGET)[FEATURES].median(); scaled = (med - med.min()) / (med.max() - med.min())
plt.figure(figsize=(9, 8)); sns.heatmap(scaled, annot=med.round(1), fmt="", cmap="YlGnBu", cbar_kws={"label": "relative level"})
plt.title("Crop fingerprint: median requirement per crop"); plt.tight_layout(); plt.savefig(OUT / "eda_4_crop_fingerprint.png", dpi=130); plt.close()

# 6. which crops are hard to separate -----------------------------------------------------------------
z = (med - df[FEATURES].mean()) / df[FEATURES].std()
d = pd.DataFrame(squareform(pdist(z)), index=z.index, columns=z.index)
print("\nMost similar crop pairs:\n", d.where(np.triu(np.ones(d.shape), 1).astype(bool)).stack().sort_values().head(6).round(2).to_string())
F, _ = f_classif(df[FEATURES], df[TARGET])
print("\nANOVA F per feature:\n", pd.Series(F, index=FEATURES).sort_values(ascending=False).round(0).to_string())
print("\nSaved plots to", OUT)
