<div align="center">
  <img src="https://img.shields.io/badge/🔥-Wildfire_Survival_Analysis-FF4500?style=for-the-badge" alt="Wildfire">
  
  <h1>WiDS Global Datathon 2026<br>Wildfire Threat Prediction</h1>
  
  <p><em>Predicting the probability of a wildfire threatening an evacuation zone within 12h, 24h, 48h, and 72h using survival analysis on right-censored data.</em></p>

  [![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26)
  [![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
  [![scikit-survival](https://img.shields.io/badge/scikit--survival-GBSA+RSF-F7931E?style=flat-square)](https://scikit-survival.readthedocs.io/)
  [![LightGBM](https://img.shields.io/badge/LightGBM-Boosting-9ACD32?style=flat-square)](https://lightgbm.readthedocs.io/)
  [![CatBoost](https://img.shields.io/badge/CatBoost-Gradient-FFCD00?style=flat-square)](https://catboost.ai/)
  [![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-EC4E20?style=flat-square)](https://xgboost.readthedocs.io/)

  <br>

  **Team:** The Neurons &nbsp;•&nbsp; **Best LB:** 0.96310 &nbsp;•&nbsp; **Rank:** 673 / 1500+

</div>

---

## 🏆 Competition

| | |
|---|---|
| **Host** | [WiDS Global Datathon 2026 — Kaggle](https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26) |
| **Task** | Predict cumulative probabilities of wildfire "hit" at T ∈ {12, 24, 48, 72} hours |
| **Data** | 221 training rows (152 right-censored), 95 test rows |
| **Metric** | `Hybrid = 0.3 × C-index + 0.7 × (1 − Weighted_Brier)` |
| **Brier Weights** | `0.3 × B_24h + 0.4 × B_48h + 0.3 × B_72h` |

> The core challenge: extreme class imbalance, heavy right-censoring, and only **221 samples** — making overfitting the primary enemy.

---

## 📈 Leaderboard Journey

```
Score:   0.957 ───── 0.960 ───── 0.963 ─────────── 0.975 (target)
          │           │           │
         v8          v10         v11
      baseline    6-model    survival
                 ensemble   paradigm
```

| Version | Pipeline | LB Score | Key Innovation |
|---------|----------|----------|----------------|
| v8 | Binary classifiers + IPCW | **0.95669** | Baseline GBDT ensemble |
| v10 | 6-model Brier-optimized | **0.96025** | Horizon-specific objectives (AUC for 12h, Brier for 24/48/72h) |
| v11 | **Survival + Binary blend** | **0.96310** | Native survival models (GBSA + RSF) + binary classifiers |

---

## 🧠 Technical Approach

### Architecture: Pipeline v11 (Current Best)

```
┌─────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING                       │
│  Raw features → Physics-based transforms → 25 final feats  │
│  (log_dist, projected_time, directional_threat, risk_score) │
└──────────────────────┬──────────────────────────────────────┘
                       │
           ┌───────────┴───────────┐
           ▼                       ▼
┌─────────────────────┐  ┌─────────────────────┐
│  SURVIVAL MODELS    │  │  BINARY CLASSIFIERS  │
│                     │  │                      │
│  • GBSA (sksurv)    │  │  • CatBoost          │
│  • RSF  (sksurv)    │  │  • LightGBM          │
│                     │  │  • ExtraTrees         │
│  Learns full S(t)   │  │                      │
│  → extract P(T≤h)   │  │  IPCW-weighted       │
│  Naturally monotone  │  │  Optuna-tuned (50t)  │
└─────────┬───────────┘  └──────────┬───────────┘
          │                         │
          │    10 seeds × 5 folds   │
          └────────┬────────────────┘
                   ▼
        ┌─────────────────────┐
        │  HYBRID ENSEMBLE    │
        │                     │
        │  Softmax-weighted   │
        │  blend (Brier-opt)  │
        │  Min 3% per model   │
        └─────────┬───────────┘
                  ▼
        ┌─────────────────────┐
        │  POST-PROCESSING    │
        │                     │
        │  • CV-Isotonic cal  │
        │  • Base rate align  │
        │  • Rank blend (5%)  │
        │  • Monotonicity     │
        │  • Clip [0.008,0.99]│
        └─────────┬───────────┘
                  ▼
            submission.csv
```

### Key Design Decisions

| Decision | Reason |
|----------|--------|
| **Native survival models** | Learns the full survival curve S(t) instead of 4 independent binary targets — shares information across horizons |
| **No Ridge stacking** | With 221 rows, meta-learners overfit catastrophically |
| **Geometric mean blending** | More robust than arithmetic for probability ensembles |
| **10 seeds × 5 folds** | 50 fits per model per horizon = maximum stability |
| **Brier-optimized Optuna** | 70% of the competition metric is Brier — optimize what matters |
| **Horizon-specific objectives** | 12h → AUC (C-index only), 24/48/72h → Brier (dominates) |

---

## 🔬 Feature Engineering

Built from fire physics — every feature has a physical interpretation:

| Feature | Physical Meaning |
|---------|-----------------|
| `log_dist` | Log-distance to fire (diminishing threat with distance) |
| `projected_time_to_hit` | ETA = distance ÷ closing speed |
| `directional_threat` | Speed × alignment — how aggressively fire approaches |
| `near_miss_margin` | Distance minus fire expansion — will it barely miss or hit? |
| `risk_score` | Composite: proximity (50%) + speed (30%) + alignment (20%) |
| `hazard_ratio_proxy` | Instantaneous hazard rate = speed ÷ distance |
| `danger_zone` | Binary: close AND approaching = imminent threat |
| `night_fire` | Fires behave differently at night (wind patterns change) |

---

## 🚀 Quick Start

### 1. Setup
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

pip install pandas numpy scikit-learn lightgbm xgboost catboost \
            optuna scipy scikit-survival
```

### 2. Run
```bash
# v10: Binary ensemble (6 models, ~15 min)
python pipeline_v10_FINAL.py

# v11: Survival + Binary hybrid (~8 min)
python pipeline_v11_SURVIVAL.py
```

### 3. Output
Both pipelines generate submission CSVs with columns:
```
event_id, prob_12h, prob_24h, prob_48h, prob_72h
```

---

## 📂 Repository Structure

```
WiDS/
├── pipeline_v10_FINAL.py       # 6-model binary ensemble (LB: 0.960)
├── pipeline_v11_SURVIVAL.py    # Survival + binary hybrid (LB: 0.963)
├── BATTLE_PLAN_TOMORROW.md     # Strategy for next iteration
├── Analytics Engine.md                   # Competition rules & constraints
│
├── train.csv                   # Training data (221 rows)
├── test.csv                    # Test data (95 rows)
├── sample_submission.csv       # Submission format
├── metaData.csv                # Feature descriptions
│
├── submission_07.csv           # v8 output  (LB: 0.957)
├── submission_08.csv           # v10 output (LB: 0.960)
├── submission_09.csv           # v11 output (LB: 0.963) ← BEST
│
└── README.md
```

---

## 📊 OOF Validation (v11)

### Survival Models
| Horizon | AUC | Brier |
|---------|-----|-------|
| 12h | 0.9663 | 0.0492 |
| 24h | 0.9809 | 0.0257 |
| 48h | 0.9902 | 0.0135 |

### Binary Classifiers
| Horizon | Model | AUC | Brier |
|---------|-------|-----|-------|
| 12h | CatBoost | 0.9763 | 0.0506 |
| 24h | CatBoost | 0.9874 | 0.0253 |
| 48h | CatBoost | 0.9979 | 0.0130 |
| 72h | CatBoost | 0.9995 | 0.0001 |

---

## 🔮 Next Steps

See [`BATTLE_PLAN_TOMORROW.md`](BATTLE_PLAN_TOMORROW.md) for the detailed roadmap to 0.975+. Key areas:

1. **Feature revolution** — Physics-based TTA, regime-conditional features, rank features
2. **Extreme simplicity** — Single CatBoost with 8 features vs. complex ensemble
3. **Competition metric as loss** — Optimize the exact hybrid score in Optuna
4. **Learn from the community** — Study top-scoring public notebooks

---

<div align="center">
  
  **Built for the [WiDS Global Datathon 2026](https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26)**
  
  *Because every minute matters when wildfires threaten communities.*

</div>
