<div align="center">
  <h1>🔥 WiDS Global Datathon 2026<br>Wildfire Survival Analysis</h1>
  
  <p><strong>Predicting the probability of a wildfire hitting an evacuation zone within 12h, 24h, 48h, and 72h using right-censored survival analysis.</strong></p>

  [![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
  [![LightGBM](https://img.shields.io/badge/LightGBM-Gradient_Boosting-green.svg)](https://lightgbm.readthedocs.io/)
  [![Status](https://img.shields.io/badge/Status-Leaderboard_Ready-success.svg)]()
</div>

---

## 🏆 Competition Overview
- **Host:** [Kaggle — WiDS Global Datathon 2026](https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26)
- **Goal:** Predict cumulative survival probabilities at $T \in \{12, 24, 48, 72\}$ hours.
- **Metric:** `Hybrid Score = 0.3 × C-index + 0.7 × (1 − Weighted Brier Score)`
  - *C-index* measures correct ranking of predictions (higher is better).
  - *Brier Score* measures probability calibration accuracy (lower is better).

---

## 🧠 Approach: The Multi-Agent Pipeline (v5.0)

This repository contains a state-of-the-art predictive pipeline designed specifically to optimize the harsh conditions of this competition: a tiny training dataset (221 rows) mixed with high-dimensional physics data from [WatchDuty](https://www.watchduty.org/).

The architecture simulates a **5-Agent System** acting in phases:

1. **DATA AGENT:** Engineers 30+ physics-grounded features (distance, closing speed, fire growth) and permanently prunes unstable noise.
2. **MODEL AGENT:** Executes a **Multi-Seed Ensemble**. It trains 3 distinct algorithms (LightGBM, Logistic Regression, Random Forest) across 5 different initializations to neutralize single-seed variance.
3. **CRITIC AGENT:** Evaluates models out-of-fold (OOF) to block any unstable or overfitted estimators from passing into the final blend.
4. **VALIDATION AGENT:** Ensures strict OOF-vs-Test distribution alignment to prevent invisible covariate shifting.
5. **MANAGER AGENT:** Enforces isotonic monotonicity (`prob_12h ≤ prob_24h ≤ prob_48h ≤ prob_72h`) and runs the multi-submission strategy.

---

## ⚡ Top-2 Killer Tweaks implemented:

To bridge the gap from "Top 10" to "Top 2", this pipeline utilizes 4 advanced mathematical micro-optimizations:

> [!TIP]
> 1. **Locked Horizon-Specific Blending:** Replaces safe adaptive OOF tuning with mathematically aggressive, locked blend configurations per horizon ($12h \rightarrow 60/40$, $72h \rightarrow 82/18$). This optimizes directly for leaderboard C-Index instead of local Brier scores.
> 2. **Bi-Directional Selective Stretch:** Selectively stretches highly confident predictions ($p > 0.70$) upward and pushes low-confidence predictions ($p < 0.08$) downward, perfectly preserving the middle curve. 
> 3. **Pairwise Rank Sharpening:** Amplifies separation in the rank transform array using an aggressive exponential power ($1.35$) so events are violently separated for the C-Index ranking metric.
> 4. **Fold-wise Calibration:** Uses `StratifiedKFold` Platt scaling to rigorously eliminate subtle leakages usually present in standard OOF probability calibration.

---

## 📂 The 3-Submission Strategy

Because the Kaggle test-set can be unpredictable, the pipeline automatically generates **three highly optimized variants**, dynamically reacting to different risk thresholds:

| File | Strategy | Best For | Logic |
|------|----------|----------|-------|
| `submission_A.csv` ⭐ | **BALANCED** | **Primary Submission** | Uses the aggressive locked blend configurations + the Bi-Directional stretch without any extra risk modifiers. |
| `submission_B.csv` | **AGGRESSIVE** | **Edge Optimization** | Adds an extra global stretch (`power=0.93`). Gambles optimal calibration to secure harder C-index separation points. |
| `submission_C.csv` | **CONSERVATIVE** | **Safe Fallback** | Removes extra stretch and uses a tighter clipping bound (`[0.03, 0.97]`). Safest submission against an unstable leaderboard. |

---

## 📊 Pipeline Validation Metrics

| Horizon | Ensemble AUC | OOF Brier Score |
|---------|-------------|-------------|
| **12h** | 0.973 ± 0.003 | 0.060 |
| **24h** | 0.986 ± 0.002 | 0.028 |
| **48h** | 0.985 ± 0.003 | 0.019 |
| **72h** | 1.000 ± 0.000 | 0.005 |

---

## 🚀 How to Run

1. **Install Dependencies:**
```bash
pip install pandas numpy scikit-learn lightgbm scipy
```

2. **Execute the Pipeline:**
```bash
python pipeline.py
```

This will run all phases automatically, train 75 separate models (to eliminate variance), calibrate them, blend them, and output the 3 final submission files.
