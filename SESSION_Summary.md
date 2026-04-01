# WiDS Global Datathon 2026 — Session Summary
## Date: April 1, 2026

---

## 🏆 CURRENT POSITION
- **Team**: The Neurons
- **Rank**: #898
- **Best Score**: **0.95760** (submission_06.csv, v7.2 - Pending v8.1 LB Check)
- **Target Score**: 0.97566+ (top tier)
- **Gap to close**: 0.01806
- **Competition ends**: ~5h remaining submissions daily

---

## 📊 SUBMISSION HISTORY

| Submission | Version | Score | Change | Notes |
|------------|---------|-------|--------|-------|
| submission_04 | v7.0 | 0.95669 | baseline | Ridge stacking, 5 models, IPCW |
| submission_05 | v7.1 | 0.95663 | -0.00006 | Removed XGB, base-rate recalib (HURT) |
| submission_06 | v7.2 | 0.95760 | +0.00091 | Geo mean blend, 700 models, extreme reg |
| **submission_07** | **v8.1** | **TBD** | **TBD** | **Added RSF, Softmax weights, Dropped 12h XGB, Cummax** |

---

## 🔬 KEY LEARNINGS (CRITICAL FOR TOMORROW)

### What WORKED ✅
1. **Replacing Ridge stacking with Geometric Mean blend** — eliminated overfitting gap
2. **7 random seeds** (vs 5) — more stable predictions
3. **Extreme regularization** — forced models to generalize on 221 samples
4. **60 Optuna trials** optimizing Brier score directly
5. **Bringing XGBoost back** with heavy reg — diversity helps when regularized
6. **Isotonic calibration only** — consistently best calibrator
7. **39 features** (vs 22 in v7.1) — more engineered features helped

### What HURT ❌
1. **Base-rate recalibration** — v7.1 tried to force test mean → train mean, DECREASED score
2. **Ridge stacking** — overfits on 221 samples (OOF=0.987 vs LB=0.957 = 0.03 gap)
3. **Removing XGBoost entirely** — lost diversity, didn't help
4. **Pseudo-labeling** — rejected every time by quality checks

### Key Numbers
- OOF-LB gap reduced: 0.030 (v7.0) → 0.027 (v7.2) — IMPROVING
- OOF estimate v7.2: 0.98579
- Weighted Brier (OOF): 0.01569
- Avg AUC (OOF): 0.98927
- Total models: 700 (5 types × 7 seeds × 4 horizons × 5 folds)

---

## 🔧 CURRENT PIPELINE (v7.2) ARCHITECTURE

```
pipeline.py — v7.2 "Maximum Generalization"

PHASE 1: Data Loading (221 train, 95 test)
PHASE 2: IPCW Censoring Weights (capped at 3x)
PHASE 3: Feature Engineering (39 features)
  - Raw features + log/inverse transforms
  - risk_score composite, near_miss_margin
  - Binary thresholds (is_close, is_very_close, is_approaching)
  - Interaction features (dist×alignment, speed×close)
  - Temporal (hour_sin, hour_cos)
  - Growth threat features
PHASE 4: Optuna Tuning (60 trials, Brier-optimized)
  - LGBM, XGBoost, CatBoost each tuned per horizon
  - Regularization-biased search space
PHASE 5: 5-Model Base Layer × 7 seeds × 5 folds
  - LightGBM, XGBoost, CatBoost, LogisticRegression, RandomForest
PHASE 6: Seed Averaging (7 seeds)
PHASE 7: Isotonic Calibration (CV-applied)
PHASE 8: Weighted Geometric Mean Blend (NO stacking!)
  - Weights from softmax of OOF hybrid scores
  - Falls back to simple average if better
PHASE 9: Gentle Rank Blend (92-95% calibration, 5-8% rank)
PHASE 10-13: Validation, Monotonicity, Save
```

---

## 🎯 STRATEGY FOR TOMORROW (April 2, 2026)

### Priority Changes for v7.3 (highest impact first):

1. **Survival-Specific Models**
   - Add `sksurv.ensemble.RandomSurvivalForest`
   - Add `sksurv.linear_model.CoxPHSurvivalAnalysis`
   - These NATIVELY handle censoring — no IPCW needed
   - Could unlock another 0.01+ improvement

2. **Target Encoding with Leave-One-Out**
   - Encode categorical-like features (month, day_of_week, hour bins)
   - Must use LOO to prevent leakage on 221 samples

3. **Adversarial Validation**
   - Check if train/test distributions differ
   - If they do, reweight training samples to match test
   - This could explain the remaining OOF-LB gap

4. **Optimize Clip Range**
   - Current: [0.005, 0.995]
   - Try: [0.01, 0.99] — Brier heavily penalizes extreme errors
   - A single wrong extreme prediction tanks the score

5. **Horizon-Specific Model Selection**
   - 12h: Distance features dominate
   - 72h: Needs more temporal/growth features
   - Train separate feature subsets per horizon

6. **Ensemble Weight Optimization**
   - Use scipy.optimize to find exact optimal blend weights
   - Current softmax-based weights may not be optimal

### DO NOT TRY (proven failures):
- ❌ Base-rate recalibration (decreased score)
- ❌ Ridge/Linear stacking (overfits on 221 samples)
- ❌ Pseudo-labeling (always rejected by quality checks)
- ❌ Removing entire models (diversity > individual quality)

---

## 📁 FILE INVENTORY

| File | Purpose | Status |
|------|---------|--------|
| `pipeline.py` | Main pipeline v7.2 | ✅ Current |
| `submission_06.csv` | Best submission (0.95760) | ✅ Submitted |
| `submission_05.csv` | v7.1 submission (0.95663) | Archived |
| `submission_04.csv` | v7.0 submission (0.95669) | Archived |
| `train.csv` | Training data (221 rows) | Static |
| `test.csv` | Test data (95 rows) | Static |
| `metaData.csv` | Feature metadata | Static |
| `sample_submission.csv` | Submission format template | Static |
| `run_log.txt` | Latest pipeline output | Auto-generated |
| `README.md` | Project documentation | ✅ Updated |

---

## 🖥️ ENVIRONMENT

- **OS**: Windows
- **Python**: d:\WiDS\.venv\Scripts\python.exe
- **Key packages**: xgboost, catboost, lightgbm, optuna, scikit-learn, lifelines, scipy, pandas, numpy
- **Encoding**: Must run `chcp 65001` and `$env:PYTHONIOENCODING="utf-8"` before pipeline
- **GitHub**: Pushed to main, commit `8aa45cb`

---

## 💡 QUICK START TOMORROW

```powershell
cd d:\WiDS
chcp 65001
$env:PYTHONIOENCODING="utf-8"
d:\WiDS\.venv\Scripts\python.exe d:\WiDS\pipeline.py
# Then submit submission_A.csv to Kaggle
```

---

*Session saved: April 1, 2026, 12:05 PM IST*
