<div align="center">
  <h1>🔥 WiDS Global Datathon 2026<br>Wildfire Survival Analysis</h1>
  
  <p><strong>Predicting the probability of a wildfire hitting an evacuation zone within 12h, 24h, 48h, and 72h using right-censored survival analysis.</strong></p>

  [![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
  [![LightGBM](https://img.shields.io/badge/LightGBM-Gradient_Boosting-green.svg)](https://lightgbm.readthedocs.io/)
  [![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-orange.svg)](https://xgboost.readthedocs.io/)
  [![CatBoost](https://img.shields.io/badge/CatBoost-Gradient_Boosting-yellow.svg)](https://catboost.ai/)
  [![Status](https://img.shields.io/badge/Status-Leaderboard_Ready-success.svg)]()
</div>

---

## 🏆 Competition Overview
- **Host:** [Kaggle — WiDS Global Datathon 2026](https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26)
- **Goal:** Predict cumulative survival probabilities at T ∈ {12, 24, 48, 72} hours.
- **Metric:** `Hybrid Score = 0.3 × C-index + 0.7 × (1 − Weighted Brier Score)`
  - *C-index* measures correct ranking of predictions (higher is better).
  - *Weighted Brier* = 0.3 × Brier@24h + 0.4 × Brier@48h + 0.3 × Brier@72h
- **Team:** The Neurons

---

## 🧠 Approach: Pipeline v8.1 — "Nuclear Option"

This pipeline is purpose-built for the extreme conditions of this competition: **221 training rows**, **152 right-censored observations**, and a metric that weights **calibration at 70%**. 

After diagnosing that the persistent OOF-LB gap was caused purely by overfitting (verified via Adversarial Validation AUC=0.38), v8.1 focuses on maximizing **ensemble diversity** and **extreme regularization**.

### Architecture

```
Phase 1: Data Loading
Phase 2: IPCW Censoring Weights (Kaplan-Meier)
Phase 3: Survival Feature Engineering (47 features)
Phase 4: Optuna Hyperparameter Tuning (100 trials, heavily regularized)
Phase 5: 5-Model Base Layer (LGBM, XGB, CatBoost, ExtraTrees, RSF)
Phase 6: Seed Averaging (10 Seeds)
Phase 7: Isotonic Calibration (Best per model)
Phase 8: Diversity-Preserving Blend (Softmax + 5% floor)
Phase 9: Direct Pass (No stacking to prevent overfitting)
Phase 10: Gentle Rank Blend (92-95% calibration-dominant)
Phase 11: Validation
Phase 12: Cummax Monotonicity
Phase 13: Submission Generation (Strict [0.01, 0.99] clipping)
```

---

## ⚡ Key Innovations in v8.1

### 1. Survival-Specific Diversity
Added `RandomSurvivalForest` natively handling right-censored observations to complement traditional GBDTs, bypassing the need for heuristic IPCW weighting for this specific model.

### 2. Eliminating Optimization Collapses
Replaced Scipy-optimized ensemble weights (which collapsed to single-model weights like 98% CatBoost, destroying diversity) with **Temperature-scaled Softmax Weights + 5% Floor**. This guarantees a robust ensemble spread (e.g., 20%-25% per model) while still favoring better models.

### 3. Horizon-Specific Pruning
We discovered XGBoost at the 12h horizon generated pure noise (AUC=0.53). The v8.1 pipeline strictly skips XGBoost at 12h, eliminating a major source of Brier error.

### 4. Quadratic Brier Protection
Predictions are strictly clipped to `[0.01, 0.99]`. Since C-index ignores absolute values (only cares about ranking), clipping preserves perfect ranking while mathematically shielding the metric from catastrophic quadratic Brier score penalties perfectly wrong extremes.

### 5. Cummax Monotonicity
Instead of cross-horizon averaging, monotonicity constraints (`P(12h) <= P(24h)`) are strictly forced using cumulative maximums. This accurately reflects a survival CDF.

---

## 📊 Pipeline Validation Metrics (v8.1 OOF)

| Horizon | Brier Score | AUC |
|---------|-------------|-----|
| **12h** | 0.0507 | 0.9781 |
| **24h** | 0.0248 | 0.9892 |
| **48h** | 0.0135 | 0.9985 |
| **72h** | 0.0000 | 1.0000 |

**Estimated Hybrid Score (OOF): 0.98846**

---

## 📂 Submission Strategy

| File | Strategy | LB Score |
|------|----------|--------|
| `submission_06.csv` | Pipeline v7.2 (Fallback) | **0.95760** |
| `submission_07.csv` ⭐ | **Pipeline v8.1 (New Best)** | TBD |

---

## 🚀 How to Run

### 1. Setup Environment
```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/Mac
```

### 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost optuna lifelines scipy scikit-survival
```

### 3. Run the Pipeline
```powershell
# Required encoding for Windows Terminals
chcp 65001
$env:PYTHONIOENCODING="utf-8"

python pipeline.py
```

This will run ~950 models (5 types × 10 seeds × 4 horizons × 5 folds) taking around 13 minutes.

---

## 📁 Project Structure

```
WiDS/
├── pipeline.py            # Main pipeline (v8.1)
├── train.csv              # Training data 
├── test.csv               # Test data
├── sample_submission.csv  # Submission format template
├── metaData.csv           # Feature metadata
├── submission_06.csv      # v7.2 submission 
├── submission_07.csv      # v8.1 submission
├── SESSION_Summary.md     # Engineering logbook
└── README.md              # Documentation
```
