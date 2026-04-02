<div align="center">

<!-- Animated header banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,9,5,3,1&height=220&section=header&text=🔥%20Wildfire%20Threat%20Prediction&fontSize=42&fontAlignY=35&desc=WiDS%20Global%20Datathon%202026%20%E2%80%94%20Survival%20Analysis&descSize=18&descAlignY=55&animation=fadeIn" width="100%"/>

<br>

<p><strong>Can we predict when a wildfire will threaten critical infrastructure?</strong></p>

<p><em>Using right-censored survival analysis on Watch Duty wildfire alerts to forecast<br>the probability of fire intersection with evacuation zones at T ∈ {12, 24, 48, 72} hours.</em></p>

<br>

<!-- Tech badges -->
<a href="https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26"><img src="https://img.shields.io/badge/🏆_Kaggle-WiDS_2026-20BEFF?style=for-the-badge&logoColor=white" alt="Kaggle"></a>
&nbsp;
<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
&nbsp;
<a href="https://scikit-survival.readthedocs.io/"><img src="https://img.shields.io/badge/scikit--survival-GBSA-F7931E?style=for-the-badge" alt="sksurv"></a>
&nbsp;
<a href="https://catboost.ai/"><img src="https://img.shields.io/badge/CatBoost-Gradient-FFCD00?style=for-the-badge" alt="CatBoost"></a>
&nbsp;
<a href="https://lifelines.readthedocs.io/"><img src="https://img.shields.io/badge/Lifelines-CoxPH-FF6B6B?style=for-the-badge" alt="Lifelines"></a>

<br><br>

<!-- Stats row -->
<table>
<tr>
<td align="center"><h3>👥</h3><strong>Team</strong><br><code>The Neurons</code></td>
<td align="center"><h3>📊</h3><strong>Best LB</strong><br><code>0.96310</code></td>
<td align="center"><h3>🎯</h3><strong>Target</strong><br><code>0.9856+</code></td>
<td align="center"><h3>🧪</h3><strong>Samples</strong><br><code>221 train</code></td>
<td align="center"><h3>📐</h3><strong>Metric</strong><br><code>Hybrid Score</code></td>
</tr>
</table>

</div>

<br>

---

## 🎯 The Challenge

> **"Predict cumulative wildfire threat probabilities across four time horizons — with only 221 samples and 69% right-censored data."**

<table>
<tr><td>🏢 <strong>Host</strong></td><td><a href="https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26">WiDS Global Datathon 2026 — Kaggle</a></td></tr>
<tr><td>🎯 <strong>Objective</strong></td><td>Predict <code>P(fire hits infrastructure)</code> at T ∈ {12, 24, 48, 72} hours</td></tr>
<tr><td>📐 <strong>Metric</strong></td><td><code>Hybrid = 0.3 × C-index + 0.7 × (1 − Weighted_Brier)</code></td></tr>
<tr><td>⚖️ <strong>Brier Weights</strong></td><td><code>0.3 × B₂₄ₕ + 0.4 × B₄₈ₕ + 0.3 × B₇₂ₕ</code></td></tr>
<tr><td>⚠️ <strong>Core Difficulty</strong></td><td>Extreme small-sample regime (221 rows) → overfitting is the #1 enemy</td></tr>
</table>

---

## 📈 The Journey — From Baseline to Anti-Overfit

```
                    ┌─────────────────────────┐
  Score  0.957      │     0.960       0.963   │     0.9856+
         ●──────────┤──────●──────────●───────┤─────── ◎ target
         │          │      │          │       │
        v8         v10    v10       v11      v12
     baseline    ensemble  tuned  survival  ANTI-OVERFIT
                                  paradigm   (current)
```

| Version | Architecture | LB Score | Key Insight |
|:-------:|:------------|:--------:|:------------|
| `v8` | Binary GBDT + IPCW | **0.957** | Established survival-aware weighting baseline |
| `v10` | 6-model Brier-optimized ensemble | **0.960** | Horizon-specific loss functions (AUC@12h, Brier@rest) |
| `v11` | Survival + Binary hybrid | **0.963** | Native `S(t)` curves via GBSA+RSF → natural monotonicity |
| `v12` | **Anti-Overfit paradigm** | **TBD** | EPV-driven feature decimation (25→10), extreme regularization |

> 💡 **Key discovery:** With only 69 events across 221 rows, using 25 features gave an EPV (Events Per Variable) of **2.76** — catastrophically below the minimum of 7. This was the root cause of 0.03 OOF-to-LB gaps in all prior versions.

---

## 🧠 Architecture — Pipeline v12 (Anti-Overfit)

```
                         ╔═══════════════════════════════════╗
                         ║    🔬 10 PHYSICS-BASED FEATURES    ║
                         ║  log_dist · projected_time · etc. ║
                         ║        EPV = 6.90 (safe)          ║
                         ╚═══════════════╤═══════════════════╝
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
                    ▼                    ▼                    ▼
        ╔═══════════════════╗ ╔══════════════════╗ ╔═══════════════════╗
        ║   GBSA (sksurv)   ║ ║  CoxPH (lifelines)║ ║ CatBoost (IPCW)  ║
        ║                   ║ ║                  ║ ║                   ║
        ║ • Optuna 30t      ║ ║ • L1+L2 = 0.5   ║ ║ • depth = 2       ║
        ║ • max_depth ≤ 3   ║ ║ • ElasticNet 50% ║ ║ • L2_reg = 25     ║
        ║ • min_leaf ≥ 10   ║ ║ • Full S(t)→P    ║ ║ • min_leaf = 15   ║
        ║ • dropout reg     ║ ║ • Parametric      ║ ║ • 300 iterations  ║
        ╚════════╤══════════╝ ╚════════╤═════════╝ ╚════════╤══════════╝
                 │                     │                    │
                 │    10 seeds × 5 folds per model         │
                 └─────────────────┬───────────────────────┘
                                   │
                                   ▼
                     ╔══════════════════════════╗
                     ║   SOFTMAX-WEIGHTED BLEND  ║
                     ║                          ║
                     ║  • Score-proportional w   ║
                     ║  • 10% diversity floor   ║
                     ║  • No stacking (overfit) ║
                     ╚═════════════╤════════════╝
                                   │
                                   ▼
                     ╔══════════════════════════╗
                     ║    POST-PROCESSING        ║
                     ║                          ║
                     ║  • Logit base-rate shift  ║
                     ║  • 3% rank blend         ║
                     ║  • Strict monotonicity   ║
                     ║  • Clip [0.008, 0.992]   ║
                     ║  • NO isotonic cal ✗     ║
                     ╚═════════════╤════════════╝
                                   │
                                   ▼
                          📄 submission.csv
```

---

## 🔬 Feature Engineering — Physics Over Noise

> Every feature has a **physical interpretation**. No blind feature factories. Stability over quantity.

| # | Feature | Formula | Physical Meaning |
|:-:|:--------|:--------|:----------------|
| 1 | `dist_min_ci` | Raw distance | Closest fire perimeter to infrastructure |
| 2 | `log_dist` | `ln(1 + dist)` | Log-distance — diminishing threat with range |
| 3 | `alignment_abs` | Bearing alignment | Is the fire heading *directly* at us? |
| 4 | `num_perimeters` | Count | How many fire fronts are active nearby? |
| 5 | `closing_speed` | Raw speed | Rate of fire approach (m/h) |
| 6 | `projected_time` | `dist ÷ speed` | Estimated time of arrival |
| 7 | `near_miss_margin` | `dist − growth − advance` | Will the fire barely miss or hit? |
| 8 | `threat_gravity` | `(align × speed) ÷ dist²` | Gravitational threat: inverse-square law |
| 9 | `wavefront_eta` | `(dist − growth) ÷ (speed + growth_rate)` | ETA including radial expansion |
| 10 | `risk_score` | `proximity × speed × alignment` | Multiplicative composite risk |

---

## 🛡️ Anti-Overfit Defenses

This pipeline treats every decision through the lens of **"will this overfit on 221 rows?"**

<table>
<tr>
<th width="200">Defense</th>
<th>What</th>
<th>Why</th>
</tr>
<tr>
<td>🎯 <strong>EPV ≥ 7</strong></td>
<td>10 features max (69 events ÷ 10 = 6.9)</td>
<td>Below EPV=7, model parameters don't converge reliably</td>
</tr>
<tr>
<td>🌲 <strong>Shallow trees</strong></td>
<td><code>max_depth ≤ 3</code>, <code>min_leaf ≥ 10</code></td>
<td>Forces models to learn broad patterns, not memorize</td>
</tr>
<tr>
<td>🔗 <strong>Heavy L1/L2</strong></td>
<td><code>penalizer=0.5</code> on CoxPH, <code>l2_reg=25</code> on CatBoost</td>
<td>Shrinks coefficients toward zero — kills noise features</td>
</tr>
<tr>
<td>🎲 <strong>10-seed averaging</strong></td>
<td>10 seeds × 5 folds = 50 fits per model</td>
<td>Averages out random variance in small-sample splits</td>
</tr>
<tr>
<td>🚫 <strong>No isotonic calibration</strong></td>
<td>Removed entirely</td>
<td>Isotonic regression memorizes the OOF set on small data</td>
</tr>
<tr>
<td>🚫 <strong>No stacking</strong></td>
<td>Softmax blend only</td>
<td>Meta-learners overfit catastrophically with N=221</td>
</tr>
</table>

---

## 🚀 Quick Start

```bash
# 1. Clone & setup
git clone https://github.com/VAIBHAV7848/WiDS-Global-Datathon-2026---Wildfire-Survival-Analysis.git
cd WiDS-Global-Datathon-2026---Wildfire-Survival-Analysis

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate              # Windows
# source .venv/bin/activate         # Linux/Mac

# 3. Install dependencies
pip install pandas numpy scikit-learn scikit-survival lifelines \
            catboost optuna scipy matplotlib

# 4. Run the anti-overfit pipeline (~3 min)
python pipeline_v12_ANTIOVERFIT.py

# 5. Output → submission_10.csv
```

---

## 📂 Repository Structure

```
WiDS/
│
├── 🧠 Pipelines
│   ├── pipeline_v10_FINAL.py           # 6-model binary ensemble     (LB: 0.960)
│   ├── pipeline_v11_SURVIVAL.py        # Survival + binary hybrid    (LB: 0.963)
│   └── pipeline_v12_ANTIOVERFIT.py     # ⭐ Anti-overfit paradigm     (current)
│
├── 📊 Analysis
│   └── eda/
│       ├── run_eda.py                  # Automated profiling & KM curves
│       ├── eda_report.md               # Column profiles & leakage audit
│       └── plots/                      # KM curves, correlations, distributions
│
├── 📄 Data
│   ├── train.csv                       # 221 rows (69 events, 152 censored)
│   ├── test.csv                        # 95 rows
│   ├── sample_submission.csv           # Format template
│   └── metaData.csv                    # Feature descriptions
│
├── 📤 Submissions
│   ├── submission_07.csv               # v8  — LB: 0.957
│   ├── submission_08.csv               # v10 — LB: 0.960
│   ├── submission_09.csv               # v11 — LB: 0.963 ⬅ previous best
│   ├── submission_10.csv               # v12 — anti-overfit (pure)
│   └── submission_10_blend.csv         # v12 × v11 hedge blend
│
└── README.md
```

---

## 📊 Model Performance (v12 OOF)

### Blend Weights (Score-Proportional)

| Model | Score (S) | Weight | Role |
|:------|:---------:|:------:|:-----|
| **GBSA** | 0.998 | ~55% | Primary — full survival curve, naturally monotonic |
| **CatBoost** | 0.986 | ~34% | Secondary — IPCW binary, extreme regularization |
| **CoxPH** | 0.872 | ~11% | Diversity — parametric baseline, very different errors |

> The 10% minimum weight floor ensures CoxPH always contributes its unique perspective, even when its raw score is lower.

---

## 🔮 Key Learnings

<table>
<tr><td>📉</td><td><strong>More features ≠ better</strong> — Cutting from 25 to 10 features was the single biggest anti-overfit gain</td></tr>
<tr><td>🎲</td><td><strong>Seed averaging is essential</strong> — With 221 rows, a single seed can swing C-index by 0.02</td></tr>
<tr><td>🚫</td><td><strong>Complex calibration hurts</strong> — Isotonic regression memorizes the OOF folds on small datasets</td></tr>
<tr><td>⚗️</td><td><strong>Survival models share info</strong> — GBSA learns one S(t) curve → all horizons benefit simultaneously</td></tr>
<tr><td>🎯</td><td><strong>Optimize what matters</strong> — 70% of the metric is Brier score → tune for Brier, not AUC</td></tr>
</table>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,9,5,3,1&height=120&section=footer" width="100%"/>

<br>

**Built with 🔥 for the [WiDS Global Datathon 2026](https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26)**

*Because every minute matters when wildfires threaten communities.*

<br>

<sub>Made by <strong>The Neurons</strong> · Powered by survival analysis, physics-based features, and extreme regularization</sub>

</div>
