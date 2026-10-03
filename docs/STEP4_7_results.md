# STEPS 4-7 - Results (reproduce: `python steps/step04_07_validation_and_demo.py`; full console output in `outputs/step4_7_console_log.txt`)

## Step 4 - Multi-criteria ranking (`cropds/rank.py`)
Two-stage design:
1. **Hard gates** - a crop is only eligible if SI >= 30% of the best crop's SI (min 10) AND rainfall(+irrigation) fit >= 0.2. Without this, extreme weights (e.g. profit 50%) recommended a crop with SI 6.6 (coffee). Gates make that impossible.
2. **TOPSIS** (min-max normalised) over: Suitability Index (benefit), water fit (benefit), risk (cost), profit index (benefit, optional).
- **Risk** = 1 - pessimistic SI/100, where pessimistic SI = 10th percentile of a lower-confidence-bound SI (forest mean - tree std) under 10% input error. It folds model uncertainty and input/season error into one number.
- **Weights are user-editable**: presets `balanced`, `risk_averse`, `profit_driven`, `water_scarce`, or any custom dict. A criterion whose spread across candidates is < 5% of its scale is treated as a tie so it cannot decide the ranking by noise.
- **Profit**: the dataset has no prices. `data/profit_index_TEMPLATE.csv` is blank for real mandi prices; `data/profit_index_ILLUSTRATIVE_DEMO.csv` holds PLACEHOLDER tiers (not real data) used only to demo the weights.
- **Weight-sensitivity analysis**: 2000 random weight vectors around the user's weights -> how often each crop is rank 1.

| Field | balanced | risk_averse | profit_driven (placeholder profits) | water_scarce | rank-1 stability |
|---|---|---|---|---|---|
| A (jute vs rice) | jute | jute | **rice** | jute | jute 93% |
| B (borderline) | mungbean | mungbean | mungbean | mungbean | mungbean 100% |
| C (low-N acidic rice) | rice | rice | **papaya** | rice | rice 98% |

Reading: weights change the winner only where a real trade-off exists (A, C). In B, mungbean wins under every preset, so the recommendation is robust to weights even though the *confidence* is low.

## Step 5 - Uncertainty and trust (`cropds/uncertainty.py`)
Field confidence (0-100) = 50% (1 - normalised entropy) + 50% margin(rank 1 - rank 2). Borderline flag when margin < 0.20. Per-crop: tree disagreement + SI P10-P90 under 10% input error.

Validation, 5-fold out-of-fold (does the trust signal predict errors?):
| | clean | 20% input noise |
|---|---|---|
| Top-1 accuracy | 0.9955 | 0.8786 |
| AUROC (confidence vs correctness) | 0.992 | 0.841 |
| Accuracy when NOT borderline | 0.998 | 0.942 |
| Accuracy when borderline | 0.615 (13 rows) | 0.602 (412 rows) |
| True crop in top-2 when borderline | 1.00 | 0.867 |
| Errors flagged as borderline | 5 of 10 | 164 of 267 |

Accuracy by confidence band (20% noise): HIGH 0.993 (n=860), MEDIUM 0.901 (n=816), LOW 0.656 (n=524). Accuracy rises monotonically with margin (0.55 -> 0.995).
Honest limits: the flag catches ~60% of errors, not all; "consider both" contains the truth ~87% of the time under noise.

## Step 6 - Explainability (`cropds/explain.py`)
SHAP (TreeExplainer on the forest, per prediction and per crop) + fuzzy memberships -> sentence with ideal / acceptable / PROBLEM features, the **limiting factor** (lowest membership, ties by weight) and top-3 SHAP drivers. Plots: `outputs/step4_7/step6_*.png`.
Example output (Field C): "Rice: in ideal range: phosphorus 47; potassium 40; temperature 23.7C; humidity 82.2% | acceptable: nitrogen 45 (ideal 62.9-95, too low); soil pH 5 (ideal 5.38-7.48, too low). Limiting factor: nitrogen is too low. SHAP drivers: nitrogen (-0.13), rainfall (+0.13), humidity (+0.10)."
SHAP agrees with the fuzzy diagnosis here (N pushes AWAY from rice). Note SHAP explains the forest's probability, the fuzzy part explains agronomic fit; they can disagree, which is itself informative.

## Step 7 - What-if engine (`cropds/whatif.py`)
Searches N, P, K (add-only), pH (lime up / sulphur down), optional user-supplied season climates; single levers and pairs; cheapest plan reaching base SI + target; a pair is kept only if no single lever suffices. Each plan is re-tested under 5% input error and flagged FRAGILE if it only works at one exact value (the forest is piecewise-constant).
Conversions (approximate, printed with a disclaimer): 1 dataset unit ~ 1 kg/ha; urea 46% N, DAP 46% P2O5, MOP 60% K2O; lime ~2 t/ha per pH unit. If the goal is unreachable the engine says which factor limits and whether it is controllable.

Demo (target +15 SI): Field C rice: "Add ~15 kg/ha N (~33 kg/ha urea)" -> SI 40.8 to 99.3 (98.1 under 5% error). Field C papaya: add ~8 kg/ha K + raise pH ~1.0 (~2 t/ha lime) -> 29.5 to 45.0. Field B: goal NOT reachable for mungbean or lentil because the limiting factor is humidity (climate), and the engine says so instead of inventing advice.

Validation (halve N on 70 N-sensitive held-out cases, ask the engine to restore 90% of the lost SI): plan found 100%, goal met 100%; recommended dN median 21 vs 41.5 lost, always <= the lost amount. It is a MINIMUM-change engine, so it under-recommends relative to full restoration by design.

## Known limitations to state in your report
- Profit values are placeholders; real use needs market data. Sowing-month what-if needs user-supplied climate for the alternative window (the dataset has no season/month).
- NPK in the dataset are relative; kg/ha figures are indicative only.
- The forest's response is step-like, so single plans can show big SI jumps for small changes (Field A rice via pH 0.3). The robustness flag helps but does not remove this.
- Risk values for weak candidates all sit near 0.9-1.0; the tie tolerance prevents tiny differences from deciding rankings.
