# STEP 2 + STEP 3 - Results
Reproduce: `python steps/step02_eda.py`, `python steps/step03_hybrid_scoring.py`. The definitive comparison (repeated CV, calibration, ablation) is `docs/STEP8_evaluation.md`.

## Step 2 - Data (verified by re-running `step02_eda.py`)
* 2200 rows, 7 features, 22 crops x exactly 100 rows, **0 missing, 0 duplicates**.
* Strongest feature correlation: **P-K = 0.74**; the others are weak, so the features are largely independent.
* Best separators (ANOVA F): **K 27238** >> humidity 3104 > P 1886 > N 898 > rainfall 606 > temperature 102 > pH 60.
* Most similar crop pairs (standardised median distance): muskmelon-watermelon 0.87, apple-grapes 0.95, orange-pomegranate 0.99, blackgram-mothbeans 1.00, lentil-mothbeans 1.08, jute-rice 1.17.
* Honest limits: balanced, clean, synthetic-like, India-centric (see `STEP1_problem_framing.md`).
* Plots: `outputs/step2/` (distributions, correlation heatmap, per-crop ranges, crop fingerprint).

## Step 3 - Hybrid scoring engine
**a) Rules (`cropds/scoring.py`)** - for each (crop, feature) a trapezoidal membership: core = p10-p90 of the crop's values (membership 1.0), support widened by `width` x core span, linear decay to 0. Crop score = 100 x weighted **geometric** mean of memberships (Liebig's law of the minimum). Missing inputs are skipped and the weights re-normalised. `EXPERT_OVERRIDES` is empty - fill it from a cited source (e.g. FAO EcoCrop) before claiming expert knowledge.

**b) ML (`cropds/model.py`)** - Logistic Regression, Random Forest (300 trees), XGBoost.

**c) Hybrid** - `SI = alpha x 100 x P_ml + (1 - alpha) x fuzzy_score`.
* alpha, fuzzy width and fuzzy sharpness (gamma) are tuned on **inner** CV folds of the training data only, on clean **and noise-corrupted** validation copies.
* Among alphas within 0.1% of the best accuracy the **smallest** is kept, so the rules stay active (otherwise tuning picks alpha = 1, i.e. pure ML).
* Tuned values: alpha was 0.8-1.0 across the 10 outer folds (0.9 on the full data), fuzzy width 1.5, gamma 4.

| Model (10-fold CV) | Top-1 | Top-3 | Macro-F1 |
|---|---|---|---|
| Logistic Regression | 0.9711 | 0.9993 | 0.9710 |
| Random Forest | 0.9945 | 1.0000 | 0.9945 |
| XGBoost | 0.9941 | 0.9998 | 0.9941 |
| Rules only | 0.9907 | 1.0000 | 0.9906 |
| **Hybrid** | **0.9948** | 1.0000 | 0.9948 |

**Honest reading:** the Hybrid *matches* Random Forest / XGBoost (McNemar p = 1.0); it does not beat them. A rules-only system reaches 99.1%, so the crop ranges themselves carry almost all of the signal. The Hybrid's value is explainability, graded 0-100 scores, trust signals and missing-input handling - see Step 8.
