"""STEPS 4-7 - validation of the trust signal and the what-if engine + demo on three example fields.
Run: python steps/step04_07_validation_and_demo.py        (~3-4 min on 1 CPU)
Output: outputs/step4_7/*.png and a full console log (also saved as console_log.txt by the README command)."""
import sys, warnings, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from cropds import rank as rk, whatif
from cropds.data import FEATURES, TARGET, EXAMPLE_FIELDS, field_series, load_dataset, load_profit_index, perturb_inputs
from cropds.explain import Explainer
from cropds.model import HybridCropModel, load_or_train
from cropds.uncertainty import BORDERLINE_MARGIN, field_confidence, trust_report

OUT = Path(__file__).resolve().parent.parent / "outputs" / "step4_7"; OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")
df = load_dataset(); X, y = df[FEATURES], df[TARGET]

# =============================================================================================== A. validation
rng = np.random.default_rng(7); rec, wi_rows, t0 = [], [], time.time()
for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y), 1):
    m = HybridCropModel().fit(X.iloc[tr], y.iloc[tr])
    for sigma in (0.0, 0.2):                                             # STEP 5: does confidence predict errors?
        Xt = X.iloc[te].reset_index(drop=True)
        if sigma: Xt = perturb_inputs(Xt, m.std_, sigma, 1, rng)
        P = m.predict_proba(Xt); conf, _, margin = field_confidence(P); order = np.argsort(-P, axis=1); truth = y.iloc[te].values
        for i in range(len(te)):
            rec.append(dict(sigma=sigma, conf=conf[i], margin=margin[i], top1=m.classes_[order[i, 0]] == truth[i],
                            top2=truth[i] in m.classes_[order[i, :2]]))
    sens = ["rice", "cotton", "maize", "coffee", "banana", "muskmelon", "watermelon", "jute"]   # STEP 7: halve N, can we undo it?
    Xte, yte = X.iloc[te], y.iloc[te]
    for i in [i for i in range(len(te)) if yte.iloc[i] in sens][:14]:
        row, crop = Xte.iloc[i].astype(float).copy(), yte.iloc[i]; ci = list(m.classes_).index(crop)
        orig = float(m.suitability_index(row.to_frame().T)[0, ci]); deg = row.copy(); deg["N"] = row["N"] * 0.5
        s_deg = float(m.suitability_index(deg.to_frame().T)[0, ci])
        if orig - s_deg < 10: continue
        base, goal, plans, _ = whatif.search(m, deg, crop, target=0.9 * (orig - s_deg), levers=("N",))
        wi_rows.append(dict(found=bool(plans), ok=bool(plans) and plans[0]["new_SI"] >= goal,
                            rec_dN=plans[0]["changes"]["N"] if plans else np.nan, true_dN=row["N"] * 0.5))
    print(f"validation fold {k}/5 ({time.time()-t0:.0f}s)", flush=True)

R = pd.DataFrame(rec); print("\n=== STEP 5 VALIDATION (out-of-fold) ===")
for sigma, g in R.groupby("sigma"):
    bl = g.margin < BORDERLINE_MARGIN
    print(f"noise {int(sigma*100):>2}%: top1={g.top1.mean():.4f} AUROC(conf vs correct)={roc_auc_score(g.top1, g.conf):.3f} "
          f"acc non-borderline={g[~bl].top1.mean():.3f} acc borderline={g[bl].top1.mean():.3f} (n={int(bl.sum())}) "
          f"top-2 when borderline={g[bl].top2.mean():.3f} errors caught by flag={int(((~g.top1) & bl).sum())}/{int((~g.top1).sum())}")
g = R[R.sigma == 0.2].copy(); g["bin"] = pd.cut(g.margin, [0, .1, .2, .4, .6, .8, 1.0], include_lowest=True)
tab = g.groupby("bin", observed=True).agg(n=("top1", "size"), top1=("top1", "mean"), top2=("top2", "mean")).round(3); print("\nAccuracy by margin bin (20% noise):\n", tab.to_string())
g["band"] = np.where(g.conf >= 70, "HIGH", np.where(g.conf >= 40, "MEDIUM", "LOW")); cb = g.groupby("band").agg(n=("top1", "size"), top1=("top1", "mean")).round(3)
print("\nAccuracy by confidence band (20% noise):\n", cb.to_string())
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
tab["top1"].plot.bar(ax=ax[0], color="#2e7d32", label="top-1"); tab["top2"].plot(ax=ax[0], color="#ef6c00", marker="o", label="top-2")
ax[0].set_ylim(0, 1.05); ax[0].set_title("Accuracy vs margin (20% noise)"); ax[0].legend()
cb.reindex(["LOW", "MEDIUM", "HIGH"])["top1"].plot.bar(ax=ax[1], color=["#c62828", "#f9a825", "#2e7d32"]); ax[1].set_ylim(0, 1.05); ax[1].set_title("Accuracy by confidence band")
plt.tight_layout(); plt.savefig(OUT / "step5_trust_validation.png", dpi=130); plt.close()
W = pd.DataFrame(wi_rows); f = W[W.found]
print("\n=== STEP 7 VALIDATION: halve N on N-sensitive held-out cases; can the engine restore 90% of the lost suitability? ===")
print(dict(cases=len(W), plan_found=round(W.found.mean(), 3), goal_met=round(W.ok.mean(), 3), median_recommended_dN=round(f.rec_dN.median(), 1),
           median_true_lost_N=round(f.true_dN.median(), 1), recommended_leq_lost=round((f.rec_dN <= f.true_dN + 1e-6).mean(), 3)))

# =============================================================================================== B. demo on 3 fields
m = load_or_train(); ex = Explainer(m); profit = load_profit_index()
for name, d in EXAMPLE_FIELDS.items():
    row = field_series(d); x = row.to_frame().T
    print("\n" + "=" * 100 + f"\nFIELD {name}\n" + "=" * 100); print(row.round(1).to_dict())
    tbl, info = trust_report(m, x); print("\n[STEP 5] TRUST\n", tbl.to_string(index=False)); print(" ->", info["message"])
    Dm = rk.build_matrix(m, x, profit_index=profit); print("\n[STEP 4] decision matrix (profit = ILLUSTRATIVE placeholder)\n", Dm.round(3).to_string())
    ranks = {}
    for pname, w in rk.PRESETS.items():
        r = rk.rank(Dm, w); ranks[pname] = r.TOPSIS; print(f"  {pname:<14} -> " + " > ".join(f"{c}({s:.2f})" for c, s in r.TOPSIS.head(4).items()))
    sens = rk.weight_sensitivity(Dm, rk.PRESETS["balanced"]); print("  rank-1 stability under random re-weighting:", {k: f"{v:.0%}" for k, v in sens.head(3).items()})
    top2 = list(tbl.crop[:2]); fig, r1 = ex.figure(row, top2[0]); fig.savefig(OUT / f"step6_{name[0]}_{top2[0]}.png", dpi=130); plt.close(fig)
    print("\n[STEP 6] EXPLANATION\n ", r1["text"])
    print("\n[STEP 7] WHAT-IF (target +15 SI)")
    for c, v in whatif.recommend(m, row, top2, target=15.0).items():
        print(f"  {c}: SI {v['base_SI']:.1f} -> goal {v['goal']:.1f}")
        if v["reachable"]:
            for p in v["plans"]: print(f"     plan (SI {p['new_SI']:.1f}; under 5% input error {p['robust_SI']:.1f}{'' if p['robust'] else ' - FRAGILE'}): " + " ".join(p["advice"]))
        elif v["limiting"] is None: print("     already near the ceiling - gains are marginal")
        else:
            print(f"     goal NOT reachable. Limiting factor: {v['limiting']}" + ("" if v["limiting_controllable"] else " (climate - cannot be changed)"))
            if v["best_effort"]: print(f"     best effort (SI {v['best_effort']['new_SI']:.1f}): " + " ".join(v["best_effort"]["advice"]))
    print("  NOTE:", whatif.DISCLAIMER)
    if name.startswith("B"):
        fig, ax = plt.subplots(1, 2, figsize=(14, 4.5)); pd.DataFrame(ranks).plot.bar(ax=ax[0], rot=25, width=.8)
        ax[0].set_title("Field B: priority score by preset"); ax[0].set_ylabel("closeness to ideal")
        yerr = np.array([tbl.SI - tbl.SI_p10, tbl.SI_p90 - tbl.SI]).clip(0)
        ax[1].bar(tbl.crop, tbl.SI, yerr=yerr, capsize=5, color="#2e7d32"); ax[1].set_title("Field B: SI with P10-P90 range (10% input error)")
        plt.tight_layout(); plt.savefig(OUT / "step4_fieldB_topsis_and_trust.png", dpi=130); plt.close()
print("\nSaved to", OUT)
