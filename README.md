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

## 🧠 Approach: Pipeline v7.0 — "Nuclear Option"

This pipeline is purpose-built for the extreme conditions of this competition: **221 training rows**, **152 right-censored observations**, and a metric that weights **calibration at 70%**.

### Architecture

```
Phase 1: Data Loading
Phase 2: IPCW Censoring Weights (Kaplan-Meier)
Phase 3: Lean Feature Engineering (~21 features)
Phase 4: Optuna Hyperparameter Tuning (40 trials/model/horizon)
Phase 5: 6-Model Base Layer (IPCW-weighted, 5-seed, 5-fold CV)
Phase 6: Seed Averaging
Phase 7: Dual Calibration (Platt + Isotonic, best per model)
Phase 8: Ridge Stacking Meta-Learner
Phase 9: Smart Pseudo-Labeling (high-confidence consensus)
Phase 10: Gentle Rank Blend (90-95% calibration-dominant)
Phase 11: Comprehensive Validation
Phase 12: Monotonicity Enforcement
Phase 13-14: Submission Generation & Verification
```

---

## ⚡ Key Innovations

### 1. IPCW Censoring Weights (Inverse Probability of Censoring Weighting)
The dataset has 152 censored observations — fires where we don't know if they hit the evacuation zone. Previous pipelines wrongly labeled ALL of these as negatives at 72h (max censored observation = 66.99h). IPCW uses a Kaplan-Meier censoring model to properly weight samples and exclude truly unknown outcomes.

### 2. Ridge Stacking Meta-Learner
Instead of heuristic weighted averaging, a Ridge Regression meta-learner is trained on OOF predictions from all 6 base models. It *learns* the optimal blending from data. Improved Brier score on **all 4 horizons** vs simple averaging.

### 3. Dual Calibration (Platt + Isotonic)
Both calibration methods are applied per model per horizon, and the one with lower Brier score is kept. Isotonic calibration won for most models — a significant improvement over Platt-only.

### 4. 6-Model Ensemble with IPCW Sample Weights
All models receive proper IPCW sample weights during training:
| Model | Role |
|-------|------|
| LightGBM | Optuna-tuned, high-performance GBDT |
| XGBoost | Optuna-tuned, complementary GBDT |
| CatBoost | Optuna-tuned, best single model (Hybrid=1.0 at 72h) |
| RandomForest | Bagged ensemble for stability |
| LogisticRegression | Linear baseline for calibration |
| ExtraTreesClassifier | Randomized splits for diversity |

### 5. Cascaded Horizon Modeling
Lower-horizon predictions feed into higher-horizon stacking layers (12h → 24h → 48h → 72h), naturally enforcing monotonicity and sharing information across time horizons.

---

## 📊 Pipeline Validation Metrics (OOF)

| Horizon | Ensemble AUC | Stacked Brier | Hybrid Score |
|---------|-------------|---------------|-------------|
| **12h** | 0.9722 | 0.0539 | 0.9539 |
| **24h** | 0.9884 | 0.0265 | 0.9780 |
| **48h** | 0.9930 | 0.0178 | 0.9855 |
| **72h** | 1.0000 | 0.0013 | 0.9991 |

**Estimated Hybrid Score: 0.98572**

---

## 📂 Submission Strategy

| File | Strategy | Submit |
|------|----------|--------|
| `submission_A.csv` ⭐ | **FULL** — IPCW + Stacking + wide clips [0.005, 0.995] | **FIRST** |
| `submission_B.csv` | **SAFE** — Same pipeline, tighter clips [0.015, 0.985] | SECOND |

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
pip install pandas numpy scikit-learn lightgbm xgboost catboost optuna lifelines scipy
```

### 3. Run the Pipeline
```bash
python pipeline.py
```

This will:
- Compute IPCW censoring weights via Kaplan-Meier
- Tune 3 GBDT models with Optuna (40 trials each, 4 horizons)
- Train 600 base models (6 models × 5 seeds × 4 horizons × 5 folds)
- Stack with Ridge meta-learner
- Calibrate, enforce monotonicity, and save submissions

**Runtime:** ~7 minutes on a modern CPU.

### 4. Submit to Kaggle
```bash
kaggle competitions submit -c WiDSWorldWide_GlobalDathon26 -f submission_A.csv -m "v7.0 IPCW+Stacking"
```

---

## 📁 Project Structure

```
WiDS/
├── pipeline.py            # Main pipeline (v7.0)
├── train.csv              # Training data (221 rows)
├── test.csv               # Test data (95 rows)
├── sample_submission.csv  # Submission format template
├── metaData.csv           # Feature metadata
├── submission.csv         # Main submission (= Variant A)
├── submission_A.csv       # Variant A: Full pipeline
├── submission_B.csv       # Variant B: Safer clips
├── SESSION_Summary.md     # Session notes
└── README.md              # This file
```

---

## 📈 Score Progression

| Version | Key Change | Score |
|---------|-----------|-------|
| v1-v4 | LightGBM + aggressive rank sharpening | 0.94691 |
| v5 | Pseudo-labeling + stretch (overfit) | 0.94691 |
| v6 | 5-model ensemble + calibration-first | *not submitted* |
| **v7** | **IPCW + Ridge stacking + 6 models** | **TBD** |

---

<div align="center">
  <strong>Built with 🔥 for WiDS Global Datathon 2026</strong>
</div>
