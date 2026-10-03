# 10 Viva Questions with Answers (+ 5 bonus tough ones)
All numbers are from this repo's own runs (`docs/STEP8_evaluation.md`, `docs/STEP4_7_results.md`). Answer honestly - the honest answers are also the strongest ones.

**Q1. What problem does your project solve, and why is it a "decision system" and not just a classifier?**
A farmer needs more than a label. A classifier says "rice". My system ranks crops, says how much to trust the ranking (confidence and a "borderline - consider both" flag), re-ranks by the user's priorities (risk, water, profit), explains why (SHAP + the limiting factor) and says what to change (add ~15 kg/ha N). Each of those four is a separate decision-support function.

**Q2. Why a hybrid of rules and machine learning? Does the hybrid beat Random Forest?**
No - and I tested it. On clean data Hybrid 99.48% vs Random Forest 99.45% vs XGBoost 99.41% (McNemar p = 1.0, not significant), because the dataset is nearly saturated. The hybrid's value is: graded 0-100 scores for every crop (needed by the what-if engine), plain-language reasons from the rules, and robustness when inputs are missing. Rules alone already reach 99.1%, which shows the crop ranges carry most of the signal.

**Q3. How does the fuzzy rule engine work? Why trapezoids and a geometric mean?**
For each crop and each feature I learn a trapezoid: the core is the 10th-90th percentile of that crop's values (membership 1), widening linearly to 0. The crop score is the weighted geometric mean of the seven memberships. A geometric mean means one bad factor (say pH far from the range) pulls the whole score down - Liebig's law of the minimum. Honestly, my ablation shows the geometric mean did not improve accuracy over an arithmetic mean (0.8818 vs 0.8786 at 20% noise); I keep it because it is interpretable.

**Q4. How are alpha and the other hyper-parameters chosen? Is there data leakage?**
SI = α·100·P_ml + (1−α)·fuzzy. α, the fuzzy width and sharpness are tuned on inner cross-validation folds inside each training fold, so test folds are never touched. Among alphas within 0.1% of the best accuracy I pick the smallest, otherwise tuning drifts to α = 1 (pure ML) and the rules become decoration. Tuned α was 0.8-1.0 across folds.

**Q5. What is "noise-aware tuning" and does it matter?**
I tune on clean and noise-corrupted validation copies (15% of feature std), imitating cheap sensors and seasonal error. Ablation: without it the hybrid scores 99.59% clean but only 73.9% at 20% noise (std 0.09, unstable); with it, 88.2% - equal to Random Forest (88.1%). This is the single most important design choice of the hybrid.

**Q6. How do you compute confidence, and how do you know it is meaningful?**
Confidence = 50% (1 − normalised entropy) + 50% margin between rank 1 and rank 2. "Borderline" means the margin is below 0.20. I validated it out-of-fold under 20% input noise: AUROC of confidence vs correctness 0.84 (0.99 clean); accuracy is 99.3% in the HIGH band, 90.1% MEDIUM and 65.6% LOW; when flagged borderline, the true crop is in the top-2 86.7% of the time. Limitation: the flag catches about 60% of errors (164 of 267), not all.

**Q7. Why TOPSIS, why "hard gates", and how is risk defined?**
TOPSIS trades off several criteria (suitability, water fit, risk, profit) by closeness to an ideal point, and the user sets the weights. Without gates, extreme weights (e.g. 50% on profit) recommended a crop with suitability 6.6/100 - so crops must first pass agronomic gates (SI ≥ 30% of the best, workable water). Risk = 1 − pessimistic SI/100, where pessimistic SI is the 10th percentile of a lower-confidence-bound SI under 10% input error - so it includes both model uncertainty and measurement/season error. I also test stability: random re-weighting keeps jute #1 in 93% of draws for field A.

**Q8. How do you explain predictions? Can SHAP and your rules disagree?**
Two views: SHAP says what pushed the model towards a crop; fuzzy memberships say which factors are inside the crop's ideal range. They are combined into a sentence with the limiting factor (lowest membership). They can disagree, because SHAP explains the forest and the rules explain agronomy; a disagreement is useful - it signals the model leans on a pattern the agronomy does not support.

**Q9. How does the what-if engine find advice, and how reliable is it?**
It searches single and paired levers (N, P, K add-only; pH ±1) on a grid for the cheapest plan that raises SI by a target, checks the plan under 5% input error, and flags it "fragile" if it only works at one exact value (the forest is piecewise-constant). Validation: I removed half the nitrogen from 70 held-out fields and asked it to restore 90% of the loss - it found a plan in 100% of cases, always recommending no more than the amount removed (median 21 vs 41.5). Caveat: dataset NPK are relative, so kg/ha and fertiliser quantities are indicative; the app shows a disclaimer. If the goal is unreachable (limiting factor is climate, e.g. humidity in field B) it says so instead of inventing advice.

**Q10. 99% accuracy looks too good. What are the limitations and the weaknesses?**
The dataset is balanced (100 per crop), clean, synthetic-like and India-centric with only 22 crops, so real-world accuracy will be lower - accuracy is therefore not my headline; trust, explanation and robustness are. Weaknesses I measured: the hybrid is worse calibrated than XGBoost/RF (ECE 0.081 vs 0.014/0.041), so the app calls it "model confidence" not probability; under noise rice and jute are confused (cereal/fibre recall 0.77 vs fruit 0.91 at 20% noise); blackgram recall drops to 0.48 under noise; a 10-point humidity error hurts every model. Profit data are placeholders.

---
## Bonus tough questions
**B1. What happens when a farmer has no soil test?**
The fuzzy part skips N, P, K and re-normalises weights; the ML part switches to a forest trained on the available columns. Top-1 is 0.963, versus 0.351 for Random Forest with mean imputation. Honest attribution: the forest trained on available columns alone gets 0.962, so the gain comes from the missing-aware design, not the blend.

**B2. Which model is the most robust to noise?**
Logistic Regression at high noise (0.847 at 30%, vs Hybrid 0.747 and RF 0.754), because smooth boundaries do not over-fit exact ranges. It is less accurate on clean data (0.971). Top-3 accuracy at 20% noise is 0.980 for Hybrid, which is why the app shows five crops, not one.

**B3. Did any of your design choices turn out not to help?**
Yes: the geometric mean gave no gain over the arithmetic mean, and ANOVA-F feature weighting was no better than equal weights (equal weights: 0.9968 clean, 0.894 at 20% noise - not conclusive, but not worse). I report this rather than hide it.

**B4. Why not deep learning?**
2200 rows × 7 features is tiny and tabular; trees already saturate it, and a neural network would reduce explainability and add cost for low-resource deployment. The whole model is 2 MB and loads in about a second.

**B5. Could this advice harm a farmer?**
It is decision support, not a decision-maker: quantities are indicative, a disclaimer is always shown, low-confidence/borderline fields are flagged for an agronomist, and the engine refuses to give advice when the limiting factor is something a farmer cannot change.
