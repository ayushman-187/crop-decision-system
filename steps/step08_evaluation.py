"""STEP 8 - Evaluation: accuracy, calibration, ablation, robustness, fairness/limitations.

Run:   python steps/step08_evaluation.py            (about 8-10 min on 1 CPU)
       python steps/step08_evaluation.py --quick    (1 repeat, no ablation; ~3 min)
Output: outputs/step8/*.png, *.csv, results_tables.md, results.pkl
"""
import sys, argparse, pickle, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import confusion_matrix
from cropds import evaluate as ev

ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); ap.add_argument("--no-ablation", action="store_true")
args = ap.parse_args()
OUT = Path(__file__).resolve().parent.parent / "outputs" / "step8"; OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")
pm = lambda m, s, d=4: f"{m:.{d}f} ± {s:.{d}f}"

print("== MAIN experiment (repeated stratified 5-fold CV) ==", flush=True)
R = ev.run_main(repeats=1 if args.quick else 2)
names, agg = R["names"], R["agg"]
md = ["# Step 8 - generated tables\n", f"Repeated stratified 5-fold CV, {'1' if args.quick else '2'} repeat(s). Mean ± std over folds.\n"]

# ---- 1. accuracy / F1 / calibration table --------------------------------------------------------------
t = pd.DataFrame({n: {"Top-1": pm(agg.loc[n, ("top1", "mean")], agg.loc[n, ("top1", "std")]),
                      "Top-3": pm(agg.loc[n, ("top3", "mean")], agg.loc[n, ("top3", "std")]),
                      "Macro-F1": pm(agg.loc[n, ("macro_f1", "mean")], agg.loc[n, ("macro_f1", "std")]),
                      "Log-loss": pm(agg.loc[n, ("logloss", "mean")], agg.loc[n, ("logloss", "std")], 3),
                      "Brier": pm(agg.loc[n, ("brier", "mean")], agg.loc[n, ("brier", "std")], 3),
                      "ECE": pm(agg.loc[n, ("ece", "mean")], agg.loc[n, ("ece", "std")], 3)} for n in names}).T
print("\n", t.to_string()); md += ["\n## Clean performance + calibration\n", t.to_markdown(), "\n"]
s = R["significance"]
print(f"\nHybrid vs RF (pooled out-of-fold): hybrid-only-correct={s['hybrid_only_correct']}, rf-only-correct={s['rf_only_correct']}, "
      f"McNemar p={s['mcnemar_p']:.3f}, mean diff={s['diff_mean']:+.4f}, 95% CI [{s['diff_ci95'][0]:+.4f}, {s['diff_ci95'][1]:+.4f}]")
md.append(f"\nHybrid vs RF: hybrid-only-correct={s['hybrid_only_correct']}, RF-only-correct={s['rf_only_correct']}, McNemar exact p={s['mcnemar_p']:.3f}, "
          f"top-1 difference {s['diff_mean']:+.4f} (95% bootstrap CI {s['diff_ci95'][0]:+.4f} to {s['diff_ci95'][1]:+.4f}).\n")

# ---- 2. noise ------------------------------------------------------------------------------------------
nz = R["noise"].groupby(["model", "sigma"])[["top1", "top3"]].mean().reset_index()
for k in ("top1", "top3"):
    tab = nz.pivot(index="model", columns="sigma", values=k).reindex(names).round(4)
    tab.columns = [f"noise {int(c*100)}%" for c in tab.columns]
    print(f"\n-- {k} under Gaussian input noise --\n", tab.to_string()); md += [f"\n## {k} under Gaussian input noise\n", tab.to_markdown(), "\n"]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for a, k in zip(ax, ("top1", "top3")):
    for n in names:
        d = nz[nz.model == n]; a.plot(d.sigma * 100, d[k], marker="o", lw=2.5 if n == "Hybrid" else 1.4, label=n)
    a.set_xlabel("input noise (% of feature std)"); a.set_ylabel(k); a.set_title(f"{k} vs input noise")
ax[0].legend(); plt.tight_layout(); plt.savefig(OUT / "robustness_noise.png", dpi=130); plt.close()

# ---- 3. systematic bias ---------------------------------------------------------------------------------
b = R["bias"].groupby(["model", "scenario"]).top1.mean().unstack().reindex(names).round(4)
print("\n-- top-1 under systematic bias --\n", b.to_string()); md += ["\n## top-1 under systematic sensor bias\n", b.to_markdown(), "\n"]
b.T.plot.bar(figsize=(11, 5.0), rot=15, width=.85); plt.ylabel("top-1"); plt.title("Systematic input bias")
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8); plt.tight_layout()
plt.savefig(OUT / "robustness_bias.png", dpi=130); plt.close()

# ---- 4. missing inputs ----------------------------------------------------------------------------------
mi = R["missing"].groupby(["model", "scenario"]).top1.mean().unstack().round(4)
order = [m for m in ["Logistic Regression (mean-imputed)", "Random Forest (mean-imputed)", "XGBoost (mean-imputed)",
                     "Random Forest (trained on available cols)", "Rules only (fuzzy)", "Hybrid"] if m in mi.index]
mi = mi.reindex(order)
print("\n-- top-1 when inputs are MISSING --\n", mi.to_string()); md += ["\n## top-1 when inputs are missing (the 'no soil test' case)\n", mi.to_markdown(), "\n"]
mi.T.plot.bar(figsize=(11, 5.2), rot=10, width=.85); plt.ylabel("top-1"); plt.title("Missing inputs: who copes?")
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8); plt.tight_layout()
plt.savefig(OUT / "robustness_missing.png", dpi=130); plt.close()

# ---- 5. calibration plot --------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.6, 5.2)); ax.plot([0, 1], [0, 1], "k--", label="perfect")
for n, c in R["calib"].items():
    ax.plot(c.confidence, c.accuracy, marker="o", lw=2.4 if n == "Hybrid" else 1.3, label=f"{n}")
ax.set_xlabel("predicted confidence (top label)"); ax.set_ylabel("observed accuracy"); ax.set_title("Reliability diagram (out-of-fold)")
ax.legend(fontsize=8); plt.tight_layout(); plt.savefig(OUT / "calibration.png", dpi=130); plt.close()

# ---- 6. fairness / per-crop / per-group ------------------------------------------------------------------
bg, bc = R["by_group"].round(3), R["by_crop"].round(3)
print("\n-- recall by agronomic group --\n", bg.to_string()); md += ["\n## Recall by agronomic group (clean | 20% noise)\n", bg.to_markdown(), "\n"]
worst = bc.sort_values("ok_h_noisy").head(6)
print("\n-- weakest crops at 20% noise (hybrid) --\n", worst.to_string()); md += ["\n## Weakest crops under 20% noise\n", worst.to_markdown(), "\n"]
print("\n-- most frequent hybrid confusions (true, predicted, count) --\n", R["confusions"]); md.append(f"\nMost frequent clean confusions: {R['confusions']}\n")
gap_clean = bg.ok_h.max() - bg.ok_h.min(); gap_noisy = bg.ok_h_noisy.max() - bg.ok_h_noisy.min()
print(f"group recall gap clean={gap_clean:.3f} noisy={gap_noisy:.3f}"); md.append(f"\nGroup recall gap (max-min): clean {gap_clean:.3f}, 20% noise {gap_noisy:.3f}\n")
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
bg[["ok_rf", "ok_h", "ok_rf_noisy", "ok_h_noisy"]].rename(columns={"ok_rf": "RF clean", "ok_h": "Hybrid clean", "ok_rf_noisy": "RF 20% noise", "ok_h_noisy": "Hybrid 20% noise"}).plot.bar(ax=ax[0], rot=0, width=.85)
ax[0].set_ylim(0.5, 1.02); ax[0].set_title("Recall by crop group"); ax[0].set_ylabel("recall")
bc.sort_values("ok_h_noisy")["ok_h_noisy"].plot.barh(ax=ax[1], color="#2e7d32"); ax[1].set_title("Per-crop recall at 20% noise (Hybrid)"); ax[1].set_xlim(0.3, 1.02)
plt.tight_layout(); plt.savefig(OUT / "fairness_by_group_and_crop.png", dpi=130); plt.close()
classes = R["classes"]; cm = confusion_matrix(R["y_idx"], R["oofS"]["Hybrid"].argmax(1))
plt.figure(figsize=(10, 8.5)); sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False, xticklabels=classes, yticklabels=classes, annot_kws={"size": 6})
plt.title("Hybrid - out-of-fold confusion matrix"); plt.xticks(fontsize=7, rotation=90); plt.yticks(fontsize=7); plt.tight_layout()
plt.savefig(OUT / "confusion_matrix_hybrid.png", dpi=120); plt.close()

# ---- 7. headline bar chart -------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 4.2)); x = np.arange(len(names)); w = .27
for i, (k, lab) in enumerate([("top1", "Top-1"), ("top3", "Top-3"), ("macro_f1", "Macro-F1")]):
    ax.bar(x + (i - 1) * w, [agg.loc[n, (k, "mean")] for n in names], w, label=lab,
           yerr=[agg.loc[n, (k, "std")] for n in names], capsize=2)
ax.set_xticks(x); ax.set_xticklabels(names, rotation=15); ax.set_ylim(0.9, 1.005); ax.legend(loc="lower right"); ax.set_title("Clean cross-validated performance")
plt.tight_layout(); plt.savefig(OUT / "model_comparison_clean.png", dpi=130); plt.close()

# ---- 8. ablation ------------------------------------------------------------------------------------------
A = None
if not (args.quick or args.no_ablation):
    print("\n== ABLATION ==", flush=True)
    A, A_raw = ev.run_ablation()
    cols = [("top1", "Top-1"), ("top3", "Top-3"), ("macro_f1", "Macro-F1"), ("logloss", "Log-loss"),
            ("top1_noise20", "Top-1 @20% noise"), ("top1_noise30", "Top-1 @30% noise"), ("top3_noise30", "Top-3 @30% noise")]
    at = pd.DataFrame({lab: [pm(A.loc[v, (c, "mean")], A.loc[v, (c, "std")], 4 if c != "logloss" else 3) for v in A.index] for c, lab in cols}, index=A.index)
    print("\n", at.to_string()); md += ["\n## Ablation (5-fold CV, one design choice changed at a time)\n", at.to_markdown(), "\n"]
    A_raw.to_csv(OUT / "ablation_raw.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6)); x = np.arange(len(A.index)); w = .38
    ax[0].bar(x - w / 2, A[("top1", "mean")], w, label="clean"); ax[0].bar(x + w / 2, A[("top1_noise30", "mean")], w, label="30% noise")
    ax[0].set_xticks(x); ax[0].set_xticklabels([v.replace("Hybrid - ", "H: ") for v in A.index], rotation=28, ha="right", fontsize=8); ax[0].set_ylim(0.4, 1.02)
    ax[0].legend(); ax[0].set_title("Ablation: top-1")
    ax[1].bar(x, A[("logloss", "mean")], color="#8e24aa"); ax[1].set_xticks(x); ax[1].set_xticklabels([v.replace("Hybrid - ", "H: ") for v in A.index], rotation=28, ha="right", fontsize=8)
    ax[1].set_title("Ablation: log-loss (lower is better)")
    plt.tight_layout(); plt.savefig(OUT / "ablation.png", dpi=130); plt.close()

(OUT / "results_tables.md").write_text("\n".join(md), encoding="utf-8")
agg.to_csv(OUT / "clean_metrics.csv"); R["noise"].to_csv(OUT / "noise_raw.csv", index=False)
pickle.dump({k: v for k, v in R.items() if k not in ("oofP", "oofS")} | {"ablation": A}, open(OUT / "results.pkl", "wb"))
print("\nSaved to", OUT)
