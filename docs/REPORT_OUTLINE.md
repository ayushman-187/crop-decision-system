# Project Report Outline  (suggested length: 12-18 pages)

Every number below is verified from the code in this repo (see `docs/STEP8_evaluation.md`, `docs/STEP4_7_results.md`).
Do not invent citations: the reference list at the end names only classic, well-known sources - verify each one before you cite it.

---
## Title page
Crop Suitability Ranking and Agronomic Recommendation Decision System - AI and Foundation project. Name, roll no., instructor, date.

## Abstract (150-200 words) - draft you can adapt
Most crop-recommendation systems return a single label with no indication of reliability, reasoning or actionability. We present a hybrid decision-support system that (i) ranks 22 crops with a Suitability Index blending a Random Forest with a fuzzy agronomic rule base, (ii) reports confidence and flags borderline cases, (iii) re-ranks by user priorities with TOPSIS under hard agronomic gates, (iv) explains each ranking with SHAP and plain-language reasons including the limiting factor, and (v) proposes the minimum fertiliser/pH change that raises suitability. Evaluated with repeated stratified 5-fold cross-validation on the Kaggle Crop Recommendation dataset, the hybrid reaches 99.5% top-1 accuracy, statistically indistinguishable from Random Forest and XGBoost, so its contribution lies elsewhere: noise-aware tuning keeps it as robust as Random Forest under input noise (86.5% vs 87.2% at 20% noise, versus 68.7% for rules alone), and when no soil test is available it retains 96.3% accuracy while mean-imputed baselines fall to 35-49%. We also report limitations honestly, including weaker probability calibration than tree ensembles and a dataset that is balanced, clean and synthetic-like.

## 1. Introduction  (1-1.5 pages)
* Problem: farmers/extension officers must choose crops from soil and climate data; a single-label answer is not enough.
* Gap: no reliability signal, no reasons, no "what can I change?", fails when soil tests are missing.
* Objectives: rank, trust, explain, what-if, low-resource app.
* Contributions (state them as in the ablation - be precise): (1) hybrid rules + ML Suitability Index with **noise-aware tuning**; (2) trust layer validated against errors; (3) two-stage multi-criteria ranking (gates + TOPSIS + weight-stability); (4) minimum-change what-if engine; (5) missing-input-aware inference; (6) honest evaluation incl. ablation, calibration, fairness.
* Report structure.

## 2. Literature review  (1.5-2 pages) - organise by theme, not by paper
1. Crop recommendation with ML (Random Forest, SVM, XGBoost on soil/climate data) - search: "crop recommendation machine learning NPK rainfall".
2. Knowledge-based land/crop suitability: FAO land-evaluation framework, FAO EcoCrop database, Liebig's law of the minimum, fuzzy suitability evaluation (Zadeh 1965 fuzzy sets).
3. Hybrid / neuro-fuzzy and knowledge-guided ML.
4. Multi-criteria decision making: TOPSIS (Hwang & Yoon 1981), weight sensitivity analysis.
5. Explainable AI: SHAP (Lundberg & Lee 2017), tree models (Breiman 2001; Chen & Guestrin 2016).
6. Uncertainty and calibration (Guo et al. 2017), selective prediction / abstention.
7. Decision support for smallholders, low-resource deployment.
Close the section with a **gap table**: (rows = prior approach, columns = ranking / confidence / explanation / what-if / missing inputs).

## 3. Dataset and problem formulation  (1 page)
Kaggle "Crop Recommendation Dataset" (verify the author on the dataset page): 2200 rows, 7 features, 22 crops x 100 rows, no missing values, no duplicates. Strongest correlation P-K = 0.74; best separator K (ANOVA F 27238), weakest pH (60). Hardest pairs: muskmelon-watermelon, apple-grapes, jute-rice. Include EDA figures (`outputs/step2/`). State assumptions/limits from `STEP1_problem_framing.md`.

## 4. Methodology  (4-5 pages) - one subsection per module, each with equation + figure
4.1 Fuzzy rules engine - trapezoid (core p10-p90, widened by `width`), weighted geometric mean, missing-input handling. Equation: score = 100 · exp(Σ wᵢ log μᵢ).
4.2 ML component - Random Forest 300 trees; baselines LR, XGBoost.
4.3 Hybrid Suitability Index - SI = α·100·P_ml + (1−α)·fuzzy; inner-CV tuning of α, width, sharpness on clean + noisy views; parsimony rule (smallest α within 0.1% of best).
4.4 Trust layer - normalised entropy, margin, confidence = 50% (1−H) + 50% margin; borderline if margin < 0.20; per-crop tree disagreement and SI P10-P90 under 10% input error.
4.5 Multi-criteria ranking - criteria (SI, water fit, risk, profit), hard gates, TOPSIS with min-max normalisation, risk = 1 − pessimistic SI/100, Dirichlet weight-sensitivity.
4.6 Explainability - SHAP per prediction + fuzzy memberships → sentence; limiting factor (lowest membership, Liebig).
4.7 What-if engine - levers N, P, K (add-only), pH (±1), optional season scenario; cheapest plan reaching +X; fragility check; fertiliser conversions (urea 46% N, DAP 46% P2O5, MOP 60% K2O, lime ~2 t/ha per pH unit) as indicative assumptions.
4.8 Application - Streamlit, 2 MB model, cached analysis, optional SHAP, PDF/CSV report. Include `docs/architecture_slide.png`.
4.9 Evaluation protocol - stratified 5-fold (×2 repeats), everything tuned inside folds, noise/bias/missing-input scenarios, ablation, McNemar + bootstrap CI.

## 5. Results  (3-4 pages) - tables and figures already generated
5.1 Accuracy / F1 / calibration table (8.1) + significance statement.
5.2 Ablation table (8.3) - discuss all findings, including that Liebig geometric mean and F-weights did not help.
5.3 Robustness: noise (`robustness_noise.png`), bias, **missing inputs** (the headline practical result: 0.963 vs 0.351-0.489).
5.4 Trust-layer validation (Step 5): AUROC 0.992 clean / 0.841 at 20% noise; accuracy HIGH 0.993, MEDIUM 0.901, LOW 0.656; borderline → true crop in top-2 86.7% of the time.
5.5 Multi-criteria and what-if demonstrations on fields A, B, C (`docs/STEP4_7_results.md`); what-if validation (70 cases, goal met 100%, recommended N always ≤ the amount removed).
5.6 App screenshots (take them yourself from `streamlit run app.py`).

## 6. Discussion  (1 page)
What worked (noise-aware tuning, missing-input handling, trust signals); what did not (no accuracy gain over RF, calibration worse than trees, geometric mean/F-weights not justified); why 99% is not the headline (saturated dataset).

## 7. Limitations and ethical considerations  (1 page)
Dataset (balanced, synthetic-like, India-centric, 22 crops, no season), relative NPK → indicative kg/ha, profit values are placeholders, group parity gap under noise (cereal/fibre vs fruit: 0.146), blackgram weak under noise, decision support not decision maker, risk of over-trust → confidence + borderline + disclaimer.

## 8. Future work  (0.5 page)
Calibrate the blended probability (temperature scaling / isotonic); real field data and yield labels; seasonal/monthly climate and crop calendars; fill `EXPERT_OVERRIDES` from FAO EcoCrop; live weather and market-price feeds; local-language/voice interface; test equal fuzzy weights as default; field trials with extension officers.

## 9. Conclusion  (0.3 page)
Restate: a trustworthy, explainable, actionable decision-support layer on top of a strong but saturated classifier.

## References (verify each before citing)
* Breiman, L. (2001). Random Forests. *Machine Learning* 45.
* Chen, T., Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD*.
* Guo, C., Pleiss, G., Sun, Y., Weinberger, K. (2017). On calibration of modern neural networks. *ICML*.
* Hwang, C.-L., Yoon, K. (1981). *Multiple Attribute Decision Making: Methods and Applications.* Springer.
* Lundberg, S., Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
* Zadeh, L. (1965). Fuzzy sets. *Information and Control* 8.
* FAO. EcoCrop database; FAO land evaluation framework.
* Kaggle - Crop Recommendation Dataset (add author/URL from the dataset page).

## Appendix
A. Repository map (README). B. Full result tables (`outputs/step8/results_tables.md`). C. Console logs. D. Example PDF report from the app.
