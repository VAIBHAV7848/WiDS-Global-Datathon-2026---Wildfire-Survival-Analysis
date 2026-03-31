# 🔥 WiDS Global Datathon 2026 — Wildfire Survival Analysis

Predicting the probability of a wildfire hitting an evacuation zone within 12h, 24h, 48h, and 72h using right-censored survival analysis.

## Competition
- **Kaggle**: [WiDS Global Datathon 2026](https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26)
- **Metric**: `Hybrid Score = 0.3 × C-index + 0.7 × (1 − Weighted Brier Score)`
- **Data**: 221 train / 95 test rows — wildfire physics features from [WatchDuty](https://www.watchduty.org/)

## Approach
1. **18 physics-based features** (distance, closing speed, fire growth, alignment)
2. **3-model ensemble**: LightGBM + Logistic Regression + Random Forest
3. **Platt calibration** on OOF predictions
4. **Isotonic monotonicity enforcement**: `prob_12h ≤ prob_24h ≤ prob_48h ≤ prob_72h`

## Results
| Horizon | Ensemble AUC | Brier Score |
|---------|-------------|-------------|
| 12h     | 0.968       | 0.059       |
| 24h     | 0.993       | 0.028       |
| 48h     | 0.999       | 0.015       |
| 72h     | 1.000       | 0.005       |

## Files
- `pipeline.py` — Complete ML pipeline (all 10 phases)
- `submission.csv` — Final predictions
- `train.csv` / `test.csv` — Competition data
- `metaData.csv` — Feature descriptions

## Run
```bash
pip install lightgbm scikit-learn pandas numpy
python pipeline.py
```
