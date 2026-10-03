# STEP 8 - Evaluation

All numbers come from `python steps/step08_evaluation.py` (stratified 5-fold CV, repeated twice for the main table; ablation = one 5-fold run).
Everything tuned (alpha, fuzzy width, sharpness) is tuned *inside* the training fold. Plots are in `outputs/step8/`; the raw generated tables are appended at the bottom.

## 8.1 Headline results (clean data, mean ± std over 10 folds)
| Model | Top-1 | Top-3 | Macro-F1 | Log-loss | Brier | ECE |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9711 ± 0.0049 | 0.9993 | 0.9710 | 0.210 | 0.080 | 0.129 |
| Random Forest | 0.9945 ± 0.0027 | 1.0000 | 0.9945 | 0.052 | 0.015 | 0.041 |
| XGBoost | 0.9941 ± 0.0032 | 0.9998 | 0.9941 | 0.028 | 0.010 | 0.014 |
| Rules only (fuzzy) | 0.9907 ± 0.0041 | 1.0000 | 0.9906 | 0.444 | 0.218 | 0.300 |
| **Hybrid** | **0.9948 ± 0.0026** | 1.0000 | 0.9948 | 0.096 | 0.024 | 0.081 |

**Reading.** The Hybrid, Random Forest and XGBoost are statistically indistinguishable on clean data (Hybrid vs RF: one sample differs out of 2200, McNemar exact p = 1.0, top-1 difference +0.0005, 95% bootstrap CI +0.0000 to +0.0014). The dataset is nearly saturated, so accuracy cannot show a winner. A rules-only system with no learning already reaches 99.1%, which tells us the per-crop agronomic ranges carry almost all the signal.

## 8.2 Calibration
Reliability diagram: `outputs/step8/calibration.png`. Expected calibration error (lower is better): XGBoost 0.014 < Random Forest 0.041 < **Hybrid 0.081** < Logistic Regression 0.129 < Rules only 0.300.
**The Hybrid is NOT better calibrated than the tree models**: blending in the fuzzy part (whose scores are not probabilities) makes its probabilities less trustworthy than Random Forest's. This is a real weakness. Consequences and mitigation: (1) the app labels the number "model confidence", not "probability"; (2) the trust signal is validated as a *ranking of reliability* (Step 5: AUROC 0.99 clean, 0.84 at 20% noise), which does not need calibrated probabilities; (3) future work: temperature scaling or isotonic calibration of the blended probability on the inner folds.

## 8.3 Ablation (what does each design choice contribute?)
| Variant | Top-1 clean | Log-loss | Top-1 @20% noise | Top-1 @30% noise |
|---|---|---|---|---|
| ML only (Random Forest) | 0.9950 | 0.052 | 0.8809 | 0.7550 |
| Rules only (fuzzy) | 0.9905 | 0.446 | 0.6932 | 0.4964 |
| **Hybrid - FULL** | 0.9955 | 0.093 | 0.8818 | 0.7464 |
| Hybrid - without noise-aware tuning | 0.9959 | 0.029 | **0.7386 (±0.09)** | **0.5886 (±0.11)** |
| Hybrid - alpha fixed at 0.5 | 0.9955 | 0.212 | 0.8450 | 0.6991 |
| Hybrid - arithmetic mean (no Liebig minimum) | 0.9950 | 0.068 | 0.8786 | 0.7455 |
| Hybrid - equal feature weights | 0.9968 | 0.052 | 0.8941 | 0.7695 |

**Findings (including the unflattering ones):**
1. **Noise-aware tuning is the component that matters.** Without it the hybrid looks best on clean data but collapses under noise (0.739 at 20% noise vs 0.882) and becomes unstable across folds (std 0.09). Tuning alpha/width/sharpness on clean *and* noisy validation copies is what lets the rules and the forest combine safely. This is the main design contribution.
2. **Tuning alpha matters:** a fixed 50/50 blend is worse under noise (0.845) and has worse log-loss (0.212).
3. **The Liebig "minimum" (geometric mean) gave no measurable benefit** over a plain weighted average (0.8818 vs 0.8786 at 20% noise, well inside one std). It is kept because it is agronomically interpretable (one bad factor lowers the whole score), not because it improves accuracy.
4. **ANOVA-F feature weighting did not help; equal weights were as good or slightly better** (0.9968 clean, 0.8941 at 20% noise, log-loss 0.052). Differences are within about 1-1.5 std, so not conclusive, but it means the weighting is not justified by the data. Changing it is a one-line edit (`fuzzy_weight_power=0.0` in `HybridCropModel`); we kept the default used in all earlier experiments and flag it as an open design choice.
5. Hybrid FULL ≈ ML only on every accuracy metric. **The hybrid does not beat pure ML; it matches it while adding explainability, graded 0-100 scores and the ability to handle missing inputs.**

## 8.4 Robustness
**Gaussian input noise** (noise = % of each feature's std, added to test inputs only; `outputs/step8/robustness_noise.png`), top-1:
| Model | 0% | 10% | 20% | 30% | 40% |
|---|---|---|---|---|---|
| Logistic Regression | 0.971 | 0.959 | 0.920 | 0.847 | 0.764 |
| Random Forest | 0.995 | 0.963 | 0.872 | 0.754 | 0.643 |
| XGBoost | 0.994 | 0.895 | 0.775 | 0.657 | 0.567 |
| Rules only | 0.991 | 0.903 | 0.687 | 0.484 | 0.376 |
| Hybrid | 0.995 | 0.966 | 0.865 | 0.747 | 0.640 |

Hybrid ≈ Random Forest (slightly ahead at 10%, slightly behind from 20%), and clearly more robust than XGBoost and rules-only. **Logistic Regression is the most noise-robust at high noise** because its smooth decision boundaries do not over-fit exact ranges. Top-3 at 20% noise: Hybrid 0.980, RF 0.980, LR 0.997 - the "show top 3" design is much safer than "show top 1".

**Systematic sensor bias** (`robustness_bias.png`), top-1: a +0.5 pH error or +3 °C temperature error barely matters (Hybrid 0.994 and 0.993). Rainfall ±20% costs about 5 points (0.946 / 0.943). **A 10-point humidity under-reading is the most damaging error for every model** (Hybrid 0.831, RF 0.856, rules-only 0.598): humidity is a strong separator in this dataset, so the app should warn users to check it.

**Missing inputs** (`robustness_missing.png`) - the case of a farmer with no soil-test report, top-1:
| Model | no soil test (N,P,K) | no humidity | no pH |
|---|---|---|---|
| Logistic Regression (mean-imputed) | 0.463 | 0.730 | 0.971 |
| Random Forest (mean-imputed) | **0.351** | 0.806 | 0.992 |
| XGBoost (mean-imputed) | 0.489 | 0.750 | 0.986 |
| Random Forest trained on available columns | 0.962 | 0.983 | 0.992 |
| Rules only | 0.903 | 0.935 | 0.989 |
| **Hybrid** | **0.963** | 0.983 | 0.992 |

Standard models with mean imputation collapse without soil data (RF 0.35). The Hybrid keeps 0.963. **Honest attribution:** the gain comes from *training the ML part on the columns that are available* (RF on available columns: 0.962) and from the rules engine simply skipping missing factors (rules-only: 0.903). The hybrid adds about 0.001 over the subset-trained forest. The contribution is the missing-input-aware design, not the blend itself.

## 8.5 Fairness and limitations
There are no people-level or regional attributes in the data, so "fairness" is analysed as **parity across crop groups** and as who is served well by the system.
* **Group recall** (clean | 20% noise, Hybrid): cereals 0.965 | 0.765, fibre 0.990 | 0.770, pulses 0.999 | 0.837, fruits 1.000 | 0.911, plantation 1.000 | 0.910. Gap between best and worst group: 0.035 clean, **0.146 under noise**. Farmers whose best crop is rice or jute are served worst, because those two overlap in climate (humid, high rainfall).
* **Confusions:** the only systematic clean error is rice -> jute (7 cases) and jute -> rice (2). Under 20% noise the weakest single crop is **blackgram (recall 0.48)**, then jute and rice (0.75). Where blackgram errors go was not analysed - a follow-up item.
* **Smallholders without soil tests** are handled (8.4), but accuracy is lower than with a full report and the app says so.
* **Dataset limits:** balanced (100 rows/crop) so rare-crop performance is untested; clean and synthetic-like; India-centric; 22 crops only (no wheat, sugarcane, ...); one value per variable (no seasonality); NPK are relative levels, so fertiliser quantities are indicative.
* **Not modelled:** prices, labour, pests, rotation, irrigation cost. The profit criterion is user-supplied; the shipped demo values are placeholders.
* **Decision support, not a decision-maker:** low-confidence or borderline fields should be referred to a local agronomist.

---
## Appendix - raw generated tables (`outputs/step8/results_tables.md`)

# Step 8 - generated tables

Repeated stratified 5-fold CV, 2 repeat(s). Mean ± std over folds.


## Clean performance + calibration

|                     | Top-1           | Top-3           | Macro-F1        | Log-loss      | Brier         | ECE           |
|:--------------------|:----------------|:----------------|:----------------|:--------------|:--------------|:--------------|
| Logistic Regression | 0.9711 ± 0.0049 | 0.9993 ± 0.0015 | 0.9710 ± 0.0050 | 0.210 ± 0.012 | 0.080 ± 0.006 | 0.129 ± 0.005 |
| Random Forest       | 0.9945 ± 0.0027 | 1.0000 ± 0.0000 | 0.9945 ± 0.0027 | 0.052 ± 0.003 | 0.015 ± 0.002 | 0.041 ± 0.003 |
| XGBoost             | 0.9941 ± 0.0032 | 0.9998 ± 0.0007 | 0.9941 ± 0.0033 | 0.028 ± 0.007 | 0.010 ± 0.004 | 0.014 ± 0.003 |
| Rules only (fuzzy)  | 0.9907 ± 0.0041 | 1.0000 ± 0.0000 | 0.9906 ± 0.0042 | 0.444 ± 0.006 | 0.218 ± 0.004 | 0.300 ± 0.005 |
| Hybrid              | 0.9948 ± 0.0026 | 1.0000 ± 0.0000 | 0.9948 ± 0.0026 | 0.096 ± 0.023 | 0.024 ± 0.005 | 0.081 ± 0.021 |



Hybrid vs RF: hybrid-only-correct=1, RF-only-correct=0, McNemar exact p=1.000, top-1 difference +0.0005 (95% bootstrap CI +0.0000 to +0.0014).


## top1 under Gaussian input noise

| model               |   noise 0% |   noise 10% |   noise 20% |   noise 30% |   noise 40% |
|:--------------------|-----------:|------------:|------------:|------------:|------------:|
| Logistic Regression |     0.9711 |      0.9591 |      0.92   |      0.8473 |      0.7636 |
| Random Forest       |     0.9945 |      0.9634 |      0.872  |      0.7543 |      0.6434 |
| XGBoost             |     0.9941 |      0.895  |      0.7745 |      0.6566 |      0.5673 |
| Rules only (fuzzy)  |     0.9907 |      0.9027 |      0.6866 |      0.4841 |      0.3764 |
| Hybrid              |     0.9948 |      0.9664 |      0.865  |      0.7468 |      0.64   |



## top3 under Gaussian input noise

| model               |   noise 0% |   noise 10% |   noise 20% |   noise 30% |   noise 40% |
|:--------------------|-----------:|------------:|------------:|------------:|------------:|
| Logistic Regression |     0.9993 |      0.9989 |      0.9968 |      0.985  |      0.9632 |
| Random Forest       |     1      |      0.9989 |      0.9802 |      0.9373 |      0.8852 |
| XGBoost             |     0.9998 |      0.9755 |      0.933  |      0.8789 |      0.8218 |
| Rules only (fuzzy)  |     1      |      0.9857 |      0.8705 |      0.7207 |      0.6325 |
| Hybrid              |     1      |      0.9998 |      0.9802 |      0.9289 |      0.8795 |



## top-1 under systematic sensor bias

| model               |   humidity -10 pts |   pH +0.5 |   rainfall +20% |   rainfall -20% |   temperature +3C |
|:--------------------|-------------------:|----------:|----------------:|----------------:|------------------:|
| Logistic Regression |             0.9295 |    0.9657 |          0.9423 |          0.9268 |            0.9295 |
| Random Forest       |             0.8561 |    0.9939 |          0.9448 |          0.9425 |            0.9925 |
| XGBoost             |             0.8689 |    0.9882 |          0.905  |          0.935  |            0.9907 |
| Rules only (fuzzy)  |             0.5984 |    0.9875 |          0.8911 |          0.893  |            0.9841 |
| Hybrid              |             0.8311 |    0.9939 |          0.9457 |          0.9434 |            0.9927 |



## top-1 when inputs are missing (the 'no soil test' case)

| model                                     |   no humidity |   no pH |   no soil test (N,P,K unknown) |
|:------------------------------------------|--------------:|--------:|-------------------------------:|
| Logistic Regression (mean-imputed)        |        0.7298 |  0.9707 |                         0.4625 |
| Random Forest (mean-imputed)              |        0.8055 |  0.9923 |                         0.3514 |
| XGBoost (mean-imputed)                    |        0.7495 |  0.9855 |                         0.4891 |
| Random Forest (trained on available cols) |        0.9832 |  0.9923 |                         0.9618 |
| Rules only (fuzzy)                        |        0.935  |  0.9886 |                         0.9025 |
| Hybrid                                    |        0.9834 |  0.9923 |                         0.963  |



## Recall by agronomic group (clean | 20% noise)

| group      |   ok_h |   ok_rf |   ok_h_noisy |   ok_rf_noisy |   n_crops |
|:-----------|-------:|--------:|-------------:|--------------:|----------:|
| cereal     |  0.965 |   0.965 |        0.765 |         0.795 |         2 |
| fibre      |  0.99  |   0.99  |        0.77  |         0.755 |         2 |
| fruit      |  1     |   1     |        0.911 |         0.926 |         9 |
| plantation |  1     |   1     |        0.91  |         0.925 |         2 |
| pulse      |  0.999 |   0.997 |        0.837 |         0.831 |         7 |



## Weakest crops under 20% noise

| crop      |   ok_h |   ok_rf |   ok_h_noisy |   ok_rf_noisy |
|:----------|-------:|--------:|-------------:|--------------:|
| blackgram |   0.99 |    0.99 |         0.48 |          0.47 |
| jute      |   0.98 |    0.98 |         0.75 |          0.71 |
| rice      |   0.93 |    0.93 |         0.75 |          0.8  |
| lentil    |   1    |    1    |         0.78 |          0.78 |
| maize     |   1    |    1    |         0.78 |          0.79 |
| cotton    |   1    |    1    |         0.79 |          0.8  |



Most frequent clean confusions: [('rice', 'jute', 7), ('jute', 'rice', 2), ('blackgram', 'maize', 1)]


Group recall gap (max-min): clean 0.035, 20% noise 0.146


## Ablation (5-fold CV, one design choice changed at a time)

| variant                                  | Top-1           | Top-3           | Macro-F1        | Log-loss      | Top-1 @20% noise   | Top-1 @30% noise   | Top-3 @30% noise   |
|:-----------------------------------------|:----------------|:----------------|:----------------|:--------------|:-------------------|:-------------------|:-------------------|
| ML only (Random Forest)                  | 0.9950 ± 0.0030 | 1.0000 ± 0.0000 | 0.9950 ± 0.0030 | 0.052 ± 0.002 | 0.8809 ± 0.0083    | 0.7550 ± 0.0190    | 0.9395 ± 0.0114    |
| Rules only (fuzzy)                       | 0.9905 ± 0.0030 | 1.0000 ± 0.0000 | 0.9904 ± 0.0030 | 0.446 ± 0.005 | 0.6932 ± 0.0196    | 0.4964 ± 0.0228    | 0.7486 ± 0.0144    |
| Hybrid - FULL                            | 0.9955 ± 0.0028 | 1.0000 ± 0.0000 | 0.9954 ± 0.0028 | 0.093 ± 0.025 | 0.8818 ± 0.0107    | 0.7464 ± 0.0186    | 0.9405 ± 0.0114    |
| Hybrid - w/o noise-aware tuning          | 0.9959 ± 0.0034 | 1.0000 ± 0.0000 | 0.9959 ± 0.0034 | 0.029 ± 0.008 | 0.7386 ± 0.0914    | 0.5886 ± 0.1077    | 0.8255 ± 0.0732    |
| Hybrid - fixed alpha = 0.5               | 0.9955 ± 0.0028 | 1.0000 ± 0.0000 | 0.9954 ± 0.0028 | 0.212 ± 0.002 | 0.8450 ± 0.0134    | 0.6991 ± 0.0164    | 0.8945 ± 0.0130    |
| Hybrid - arithmetic mean (no Liebig min) | 0.9950 ± 0.0030 | 1.0000 ± 0.0000 | 0.9950 ± 0.0030 | 0.068 ± 0.018 | 0.8786 ± 0.0102    | 0.7455 ± 0.0267    | 0.9382 ± 0.0106    |
| Hybrid - equal feature weights           | 0.9968 ± 0.0026 | 1.0000 ± 0.0000 | 0.9968 ± 0.0026 | 0.052 ± 0.003 | 0.8941 ± 0.0089    | 0.7695 ± 0.0112    | 0.9486 ± 0.0086    |

