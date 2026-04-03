<div align="center">
  <h1>🔥 WiDS Global Datathon 2026<br>Wildfire Survival Analysis</h1>
  
  <p><strong>Predicting the probability of a wildfire hitting infrastructure within 12h, 24h, 48h, and 72h using advanced Survival Analysis & Ensemble Learning.</strong></p>

  [![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
  [![LightGBM](https://img.shields.io/badge/LightGBM-Gradient_Boosting-green.svg?style=for-the-badge)](https://lightgbm.readthedocs.io/)
  [![CatBoost](https://img.shields.io/badge/CatBoost-Gradient_Boosting-yellow.svg?style=for-the-badge)](https://catboost.ai/)
  [![Status](https://img.shields.io/badge/Status-Top_100_Ready-success.svg?style=for-the-badge)]()
</div>

---

## 🏆 Competition Overview
- **Host:** [Kaggle — WiDS Global Datathon 2026](https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26)
- **Objective:** Predict cumulative survival probabilities at $T \in \{12, 24, 48, 72\}$ hours.
- **Metric:** `Hybrid Score = 0.3 × C-index + 0.7 × (1 − Weighted Brier Score)`
  - **C-index**: Measures ranking correctness (Crucial for high scores!).
  - **Weighted Brier**: Measures calibration error at 24h, 48h, and 72h.

---

## 🧠 The Evolution of our Winning Pipeline

We transitioned from standard binary classification to a **Native Survival Paradigm**, culminating in the **God Ensemble (v15)**.

### 🌟 Phase 1: Native Survival (v11)
Introduced **Scikit-Survival** (`GradientBoostingSurvivalAnalysis` & `RandomSurvivalForest`) to learn continuous survival curves. This solved the "Monotonicity" problem fundamentally.

### 🪄 Phase 2: Physics Magic (v13-v14)
Implemented **Deterministic Physics Overrides**. If a fire's speed and direction make a strike mathematically inevitable or impossible, the model hard-overrides probabilities to `0.999` or `0.001`. 

### 🛡️ Phase 3: The God Ensemble (v15 - current)
The "Lifesaver" run. We combined the best of all previous paradigms using a **Weighted Geometric Mean Blend**.
- **Robustness**: Blends signals from Survival Logic, Pseudo-Labeling, and Physics.
- **Precautionary Principle**: If *any* model flags an extreme threat, the ensemble elevates the risk (The "Evacuation Siren" logic).
- **Rank Sharpening**: Optimized specifically for the C-index component.

---

## ⚡ Key Technical Innovations

- **IPCW Weighting**: Inverse Probability of Censoring Weighting for binary horizons.
- **Pseudo-Label Distillation**: High-confidence test predictions from top submissions are fed back as "ground truth" to the latest models.
- **Extreme Regularization**: Limited features to exactly 8-10 physics-grounded variables (EPV > 7) to prevent overfitting on the small 221-row training set.
- **Base Rate Calibration**: Logit-space alignment to ensure test distributions match training realities.
- **Monotonic Law Enforcement**: Strict enforcement of $P(12h) \le P(24h) \le P(48h) \le P(72h)$.

---

## 📊 Evolutionary Submission Strategy

| Version | Strategy | Description | LB Score | Target |
| :--- | :--- | :--- | :--- | :--- |
| **v8** | Stacked Binary | Initial ensemble effort | 0.9566 | - |
| **v11** | **Survival Shift** | Scikit-Survival models introduced | 0.9631 | - |
| **v13** | **Magic Bounds** | Physics logic overrides + Distillation | 0.9750+ | - |
| **v15** | **God Ensemble** | Weighted Geometric Mean + Precautionary shift | **0.988++** | 🏆 |

---

## 🚀 Getting Started

### 1. Setup Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 2. Run the Final Pipeline
```bash
python pipeline_v15_LIFESAVER.py
```

### 3. Verify Submission
```bash
python check_sub.py
```

---

## 📂 Project Architecture

```bash
WiDS/
├── pipeline_v15_LIFESAVER.py  # The God Ensemble
├── pipeline_v13_MAGIC.py      # Physics Overrides
├── submission_15.csv          # Recommended Submission
├── train.csv / test.csv       # Dataset
├── check_sub.py               # Monotonicity & Boundary validator
└── eda/                       # Initial data profiling
```

---

<div align="center">
  <p><strong>"The fire is real. Every prediction counts."</strong></p>
  <img src="https://media.giphy.com/media/26AHvXf99A302AOfS/giphy.gif" width="300" alt="Wildfire Animation">
</div>
