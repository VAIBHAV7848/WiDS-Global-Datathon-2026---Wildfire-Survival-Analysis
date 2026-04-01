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

## 🔬 KEY LEARNINGS (v8.1 COMPLETION)

### What WORKED in v8.1 ✅
1. **RandomSurvivalForest integration** — successfully layered into the ensemble, providing structural diversity natively built for right-censoring.
2. **Softmax Weighting with a 5% Floor** — guarantees ensemble spread. We learned that scipy optimization completely collapses weights (98% CatBoost), destroying diversity. The 5% floor fixed this perfectly.
3. **Horizon-Specific Pruning** — cutting XGBoost purely out of the 12h horizon fixed a major source of noise (it had random-chance 0.53 AUC in v7.2).
4. **Strict `[0.01, 0.99]` Clipping** — mathematics shows this perfectly shields the quadratic Brier score from catastrophe without altering the rank-based C-index at all.
5. **Cummax Monotonicity** — strictly forcing `P(12h) <= P(24h)` using cummax is more accurate to survival CDFs than cross-horizon averaging.

### What HURT ❌
1. **Scipy Optimizer for Blending** — tested in v8.0 and immediately discarded in v8.1. It collapsed to single-model predictions, totally destroying the ensemble.
2. **Post-Blend Isotonic Calibration** — also discarded from v8.0. It universally fell back to raw blending, meaning it added zero value but carried heavy overfitting risk on our 221 samples.

### Key Numbers (v8.1)
- OOF estimate v8.1: **0.98846** (up from 0.98579)
- Weighted Brier (OOF): **0.01283** (down from 0.01569)
- Avg AUC (OOF): **0.99145** (up from 0.98927)
- Total models: 950 (5 types × 10 seeds × 4 horizons × 5 folds)

---

## 🔧 CURRENT PIPELINE (v8.1) ARCHITECTURE

```
pipeline.py — v8.1 "Diversity & Extreme Regularization"

PHASE 1: Data Loading (221 train, 95 test)
PHASE 2: IPCW Censoring Weights
PHASE 3: Survival Feature Engineering (47 features)
PHASE 4: Optuna Tuning (100 trials, heavily regularized)
PHASE 5: 5-Model Base Layer (LGBM, XGB, CatBoost, ExtraTrees, RSF)
PHASE 6: Seed Averaging (10 Seeds)
PHASE 7: Isotonic Calibration (Best per model)
PHASE 8: Diversity-Preserving Blend (Softmax + 5% floor)
PHASE 9: Direct Pass (No stacking)
PHASE 10: Gentle Rank Blend (92-95% calibration-dominant)
PHASE 11: Validation
PHASE 12: Cummax Monotonicity
PHASE 13: Submission Generation (Strict [0.01, 0.99] clipping)
```

---

## 🎯 STRATEGY FOR TOMORROW (April 2, 2026)

When you wake up, check your Kaggle LB score for `submission_07.csv`. Based on the result, here is your path forward:

### If Score > 0.96500 (Success, Gap Shrinking) 📈
Your diversity and extreme regularization strategy is working. The gap is shrinking.
1. **Feature Engineering Focus**: The 47 features are good, but you can try adding non-linear interaction terms or polynomial features inside `pipeline.py`.
2. **Increase Seeds**: Bump seeds from 10 to 15 to squeeze out micro-fractions of score stability.
3. **Try CoxPH**: We added RSF, but you could try adding `sksurv.linear_model.CoxPHSurvivalAnalysis` alongside it for another unique model flavor.

### If Score < 0.95760 (Regression, Gap Growing) 📉
v8.1 overfit despite our best efforts. If `[0.01, 0.99]` clipping caused this, try wider clips.
1. **Fallback**: Submit `submission_06.csv` (v7.2) to secure your ranking.
2. **Tweak the Blend Floor**: Try raising the Softmax minimum floor from 5% to 15% to force even *more* diversity.
3. **Re-evaluate Base Rates**: We saw in v8.1 that the 72h test prediction average (0.31) drifts heavily from the train base rate (0.42). If Kaggle penalizes this drift heavily, we may need to reconsider Base-Rate scaling (even though it failed previously in v7.1).

---

## 📁 FILE INVENTORY

| File | Purpose | Status |
|------|---------|--------|
| `pipeline.py` | Main pipeline v8.1 | ✅ Current |
| `submission_07.csv` | **Best v8.1 submission** | 🚀 Ready to Submit |
| `submission_06.csv` | Fallback v7.2 submission (0.95760) | ✅ Fallback |
| `train.csv` | Training data (221 rows) | Static |
| `test.csv` | Test data (95 rows) | Static |
| `metaData.csv` | Feature metadata | Static |
| `sample_submission.csv` | Submission format template | Static |
| `run_log_v8_1.txt` | Latest pipeline execution log | Saved |
| `README.md` | Project documentation | ✅ Updated |

---

## 💡 QUICK START TOMORROW

```powershell
# 1. Check your Kaggle LB score for submission_07.csv!
# 2. Then, run the environment:
cd d:\WiDS
chcp 65001
$env:PYTHONIOENCODING="utf-8"
d:\WiDS\.venv\Scripts\python.exe d:\WiDS\pipeline.py
```
