# STEP 1 - Problem Framing
**Project:** Crop Suitability Ranking and Agronomic Recommendation Decision System (AI and Foundation)

## Problem statement
A farmer or agri-officer knows the field's soil test (N, P, K, pH) and the local climate (temperature, humidity, rainfall) but must still decide **what to grow**. Most crop recommenders answer with one label ("rice") and give no sense of how sure the system is, why it chose that crop, or what could be changed to improve the options. This project builds a **decision-support system**, not only a classifier: it *ranks* crops, attaches a *trust level* to each rank, *explains* the ranking in agronomic terms and *simulates what-if changes* ("fix pH, add nitrogen").

## Users
| User | What they need |
|---|---|
| Small / marginal farmer | Simple top-3 list, plain-language reason, one clear action |
| Agri-extension officer | Ranked table, confidence, limiting factor per crop, comparison between 2 close crops |
| Student / reviewer | Transparent method, metrics, ablation, honest limitations |

## Inputs and outputs
| | Variable | Type | Range in dataset |
|---|---|---|---|
| Input (core) | N, P, K | soil nutrient level | N 0-140, P 5-145, K 5-205 |
| | temperature (C) | climate | 8.8-43.7 |
| | humidity (%) | climate | 14.3-100 |
| | pH | soil | 3.5-9.9 |
| | rainfall (mm) | climate | 20-299 |
| Input (optional, future) | soil type, season, region | categorical | not in dataset |
| Output 1 | Ranked top-N crops with **Suitability Index 0-100** | | |
| Output 2 | Confidence / uncertainty per rank (Step 5) | | |
| Output 3 | Per-feature explanation + limiting factor | | |
| Output 4 | What-if advice with new predicted suitability (Step 7) | | |

## Assumptions and limitations (10)
1. Each of the 7 input variables is a single representative value per field, with no time series within a season.
2. The dataset label means "a crop that suits these conditions", not "the highest-yielding or most profitable crop".
3. The dataset is balanced (exactly 100 rows for each of 22 crops), so real-world crop frequency is not represented.
4. The data looks synthetic or augmented (clean, no missing values, no duplicates), so high accuracy will not carry over fully to messy field data.
5. Only 22 crops are covered. Staples such as wheat and sugarcane are absent, so the system cannot recommend them.
6. Data is India-centric, so ranges may not transfer to other climates.
7. Rainfall is treated as one number, and its seasonal distribution is ignored.
8. N, P, K are relative levels, not calibrated kg/ha, so fertiliser advice in Step 7 is only indicative.
9. Market price, irrigation access, labour, pests and crop rotation are outside the model.
10. The system supports a decision and does not replace local agronomic advice.
