# 5-Minute Demo Script
Setup before you start: `streamlit run app.py`, browser at 100% zoom, sidebar visible. Open the app once beforehand so the model is warm. Keep `docs/architecture_slide.png` and the step-8 plots (`outputs/step8/`) open in another tab as a fallback.

| Time | Say | Do (in the app) | What the audience sees |
|---|---|---|---|
| 0:00-0:30 | "A farmer doesn't need only a crop name. They need: which crops, how sure are we, why, and what can I change? This system answers all four." | Show the architecture slide (10 s), then the app. | Slide, then the app's first screen (Field A is pre-loaded). |
| 0:30-1:15 | "Field A: warm and humid. Jute is top with 69/100 confidence. The bars are the Suitability Index and model confidence; the range column shows what happens if the sensor is off by 10%." | Point to the green banner, the Top-5 table, the radar chart. | **Jute** top (SI 83.5), rice second (26.5), banner "Confidence 69/100 (MEDIUM)". |
| 1:15-2:00 | "Priorities change the answer only where there is a real trade-off. Tick demo profit values - these are placeholders - and choose Profit driven." | Sidebar: tick *illustrative demo profit*, select **Profit driven**; then back to *Balanced*. | Priority table flips jute ↔ rice; caption "jute stays #1 in about 93% of random re-weightings" for Balanced. |
| 2:00-2:30 | "Now a hard case." | Sidebar → example **B - borderline field**. | Orange banner: "BORDERLINE: mungbean and lentil are nearly equal (gap 7 pts) - consider both", confidence 29/100. Say: "the system tells you when not to trust it." |
| 2:30-3:00 | "Why this crop?" | Open **Why?** tab. | Sentence with ideal/acceptable/problem factors and the limiting factor (humidity); SHAP bars + fit bars. |
| 3:00-3:50 | "Field C: rice in acidic, low-nitrogen soil. The system says nitrogen is limiting. Move the nitrogen slider and watch the ranking update." | Example **C**; **What-if** tab; drag *Add nitrogen* from 0 to 15. | Rice SI 40.8 → 99.3, papaya drops from rank 2 to 8 (▼ arrows). Then click **Find changes for my top 2 crops** → "Add ~15 kg/ha N (~33 kg/ha urea)". Mention the disclaimer. |
| 3:50-4:20 | "Many small farmers have no soil report." | Untick **I have a soil test**. | Info box appears; rice is still top; confidence recomputed. Say: "0.963 accuracy without soil data vs 0.35 for a standard model with mean imputation." |
| 4:20-4:40 | "Takeaway for the farmer or officer." | **Report** tab → *Prepare report* → download PDF (open it). | A 2-page PDF with inputs, ranking, explanation, advice, disclaimer. |
| 4:40-5:00 | "Honest results: the hybrid matches Random Forest at 99.5% - it doesn't beat it. Its value is trust, explanation, robustness to missing inputs. Limits: synthetic-like balanced data, weaker calibration, indicative kg/ha." | Show `outputs/step8/robustness_missing.png` or the ablation plot. | One chart, then "Questions?" |

## If something goes wrong
* App slow on first load → it is loading the model (~1-2 s) and SHAP (first click on *Why?*, a few seconds). Mention "SHAP loads once".
* Streamlit won't start → run `pip install -r requirements.txt`, or present from `outputs/` plots + `docs/STEP4_7_results.md`.
* Projector shows sidebar cut off → collapse it after setting inputs; everything else is in the main panel.

## Likely questions during the demo
"Is it better than Random Forest?" → no, matches (p = 1.0); see Q2 in `VIVA_QA.md`. "Where do profit values come from?" → placeholders; the CSV template is for real prices. "Can I trust the kg/ha?" → indicative only, NPK are relative levels.
