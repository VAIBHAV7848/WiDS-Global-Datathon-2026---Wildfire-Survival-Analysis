<p align="center">
  <img src="https://img.shields.io/badge/🔥_WiDS_2026-Wildfire_Survival_Analysis-FF6B35?style=for-the-badge&labelColor=1a1a2e" alt="WiDS 2026"/>
</p>

<h1 align="center">
  <br>
  🌲🔥 Wildfire Infrastructure Impact Prediction
  <br>
</h1>

<p align="center">
  <strong>Predicting wildfire–infrastructure intersection probability across 12h, 24h, 48h & 72h horizons</strong>
</p>

<p align="center">
  <a href="https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26">
    <img src="https://img.shields.io/badge/Kaggle-Competition_Page-20BEFF?style=flat-square&logo=kaggle&logoColor=white" alt="Kaggle"/>
  </a>
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/LightGBM-Survival_ML-9ACD32?style=flat-square" alt="LightGBM"/>
  <img src="https://img.shields.io/badge/Score-0.97216-success?style=flat-square" alt="Score"/>
  <img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square" alt="License"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Data_Partner-Watch_Duty-FF4500?style=flat-square" alt="Watch Duty"/>
  <img src="https://img.shields.io/badge/Events-221_Train_|_95_Test-lightgrey?style=flat-square" alt="Dataset"/>
  <img src="https://img.shields.io/badge/Horizons-12h_24h_48h_72h-orange?style=flat-square" alt="Horizons"/>
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [The Discovery](#-the-discovery--the-5km-distance-gate)
- [Architecture](#-architecture--three-zone-physics-model)
- [Evaluation Metric](#-evaluation-metric)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Methodology Deep Dive](#-methodology-deep-dive)
- [Tech Stack](#-tech-stack)

---

## 🎯 Overview

> **Competition:** WiDS Global Datathon 2026 — *Predicting Wildfire Impact: From Infrastructure to Equity*
>
> **Task:** Given the first 5 hours of wildfire observations from [Watch Duty](https://www.watchduty.org/), predict the probability that the fire intersects high-value infrastructure (transmission lines, utilities, roads) within **12, 24, 48, and 72 hours**.

This repository contains our complete solution pipeline, evolving from a standard ensemble approach to a **physics-informed three-zone deterministic model** that exploits structural patterns hidden in the data.

---

## 🔬 The Discovery — The 5km Distance Gate

The single most important finding in this competition:

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   Training Data: dist_min_ci_0_5h (distance to infrastructure)  │
│                                                                 │
│   ┌───────────────────┐    ┌──────────────────────┐             │
│   │   NEAR  (< 5km)   │    │   FAR   (≥ 5km)      │             │
│   │                   │    │                      │             │
│   │   69 / 69  HIT    │    │   0 / 152  HIT       │             │
│   │   = 100% ✅       │    │   = 0%   ✅          │             │
│   │                   │    │                      │             │
│   └───────────────────┘    └──────────────────────┘             │
│                                                                 │
│   PERFECT BINARY SPLIT — zero exceptions in 221 training rows   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

> **If a wildfire starts within 5km of infrastructure, it ALWAYS reaches it. If it starts ≥ 5km away, it NEVER does.**

This alone invalidated every public notebook that assigned ~89% probability to far-zone events at 72h.

---

## 🏗 Architecture — Three-Zone Physics Model

We further split the near zone by **fire dynamics** — whether the fire is actively growing or static:

```
                        ┌─────────────────────┐
                        │   95 Test Events     │
                        └──────────┬──────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
            ┌──────────┐   ┌──────────┐   ┌──────────┐
            │   FAR    │   │  ACTIVE  │   │  STATIC  │
            │  ≥ 5km   │   │  < 5km   │   │  < 5km   │
            │          │   │ growing  │   │ no growth│
            ├──────────┤   ├──────────┤   ├──────────┤
            │ 67 events│   │ 3 events │   │ 25 events│
            ├──────────┤   ├──────────┤   ├──────────┤
            │ P = 0.001│   │ P = 0.999│   │ P = ML() │
            │ all      │   │ all      │   │ per      │
            │ horizons │   │ horizons │   │ horizon  │
            └──────────┘   └──────────┘   └──────────┘
                 │              │               │
                 │    Deterministic Rules       │
                 │              │          LightGBM
                 │              │         10 seeds
                 ▼              ▼          5-fold
            ┌─────────────────────────────────────┐
            │      Monotonicity Enforcement       │
            │   P(12h) ≤ P(24h) ≤ P(48h) ≤ P(72h)│
            └──────────────┬──────────────────────┘
                           ▼
                    submission.csv
```

### Zone Details

| Zone | Rule | Train Evidence | Prediction |
|:-----|:-----|:---------------|:-----------|
| **FAR** | `dist ≥ 5000m` | 0/152 hit (0%) | `P = 0.001` all horizons |
| **ACTIVE** | `dist < 5000m` + `radial_growth > 0 OR area_growth > 0` | 18/18 hit at every horizon (100%) | `P = 0.999` all horizons |
| **STATIC** | `dist < 5000m` + no fire growth | 52 events, variable timing | LightGBM binary classifier per horizon |

### Static Zone — Where the Real Challenge Lives

| Horizon | Static Hit Rate | Key Predictors |
|:--------|:----------------|:---------------|
| 12h | 61.5% | `dt_first_last_0_5h`, `alignment_abs`, `num_perimeters` |
| 24h | 88.5% | `dt_first_last_0_5h`, `log1p_area_first` |
| 48h | 94.2% | `log1p_area_first`, `alignment_abs` |
| 72h | 100% | Deterministic `0.999` |

---

## 📊 Evaluation Metric

```
Hybrid Score = 0.3 × C-index + 0.7 × (1 − Weighted Brier)

Weighted Brier = 0.3 × B(24h) + 0.4 × B(48h) + 0.3 × B(72h)
```

> ⚠️ **Critical:** `prob_12h` is **NOT** in the Brier Score — it only affects C-index (30% weight).
> The 48h horizon carries the **highest** Brier weight (40%).

---

## 📈 Results

### Leaderboard Progression

| Version | Strategy | LB Score | Key Change |
|:--------|:---------|:---------|:-----------|
| v1–v9 | Standard ML ensembles | 0.956 – 0.968 | Baseline approaches |
| h_blend | Public notebook ensemble | **0.97175** | Rank-weighted blending |
| v18_BLEND | 2-zone gate + h_blend | **0.97216** | 5km distance gate discovery |
| **v20_PURE** | **3-zone physics model** | **TBD** | Active/Static fire split |

### Pipeline Evolution

```
v1-v9:  Standard ML         →  0.968  │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
h_blend: Public ensemble    →  0.972  │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
v18:    2-zone gate          →  0.972  │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
v20:    3-zone physics       →  ???   │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░?│
                                                              Target: 0.989+ ──┘
```

---

## 📁 Project Structure

```
WiDS/
│
├── 📊 Data
│   ├── train.csv                    # 221 training events
│   ├── test.csv                     # 95 test events  
│   ├── sample_submission.csv        # Submission format
│   └── metaData.csv                 # Column descriptions
│
├── 🔧 Pipelines
│   ├── pipeline_v20_PHYSICS.py      # ★ Latest: 3-zone physics model
│   └── h_blend_ensemble.py          # Reference: h_blend implementation
│
├── 📤 Submissions
│   ├── submission_v20_PURE.csv      # ★ Best: pure 3-zone physics
│   ├── submission_v20_50.csv        # Blend: 50% physics + 50% h_blend
│   ├── submission_v20_MEGA.csv      # Blend: physics + v18 for static
│   ├── submission_v18_BLEND.csv     # Previous best (LB: 0.97216)
│   └── submission.csv               # h_blend baseline (LB: 0.97175)
│
├── 📖 Documentation
│   ├── README.md                    # You are here
│   ├── SUBMISSIONS.md               # Submission strategy guide
│   ├── Analytics Engine.md                    # AI assistant config
│   └── requirements.txt            # Python dependencies
│
└── ⚙️ Config
    └── .gitignore
```

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install pandas numpy scikit-learn lightgbm scipy
```

### Run the Pipeline

```bash
# Clone the repository
git clone https://github.com/VAIBHAV7848/WiDS-Global-Datathon-2026---Wildfire-Survival-Analysis.git
cd WiDS-Global-Datathon-2026---Wildfire-Survival-Analysis

# Run the three-zone physics pipeline
python pipeline_v20_PHYSICS.py

# Output: submission_v20_PURE.csv, submission_v20_MEGA.csv, etc.
```

---

## 🧠 Methodology Deep Dive

### 1. Distance Gate Discovery

We analyzed the `dist_min_ci_0_5h` column (minimum fire-to-infrastructure distance in the first 5 hours) across all 221 training events:

| Threshold | Near Hit Rate | Far Hit Rate | Perfect Split? |
|:----------|:-------------|:-------------|:---------------|
| 3km | 56/56 (100%) | 13/165 (7.9%) | ❌ |
| 4km | 63/63 (100%) | 6/158 (3.8%) | ❌ |
| **5km** | **69/69 (100%)** | **0/152 (0%)** | **✅ PERFECT** |
| 6km | 69/77 (89.6%) | 0/144 (0%) | ❌ |

The **5km boundary** is the only threshold with a perfect binary split.

### 2. Active vs Static Fire Classification

Within the near zone, fires that show **any growth** (radial or area) in the first 5 hours behave fundamentally differently:

| Type | Criteria | 12h | 24h | 48h | 72h |
|:-----|:---------|:----|:----|:----|:----|
| **Active** | `radial_growth > 0` OR `area_growth > 0` | 100% | 100% | 100% | 100% |
| **Static** | Both = 0 | 61.5% | 88.5% | 94.2% | 100% |

Active fires hit infrastructure at **every time horizon without exception**.

### 3. Static Zone ML Model

For the 52 static training events, we train a LightGBM binary classifier per horizon:

- **Features:** `dist_km`, `dt_first_last_0_5h`, `num_perimeters_0_5h`, `alignment_abs`, `log1p_area_first`, `bearing_cos/sin`, `start_hour/month`, and key interactions
- **Regularization:** `num_leaves=6`, `max_depth=2`, `reg_alpha=5.0`, `reg_lambda=10.0`
- **Averaging:** 10 random seeds × 5-fold CV = 50 model average per horizon
- **Rationale:** Extreme regularization prevents overfitting on just 52 training rows

### 4. Monotonicity Enforcement

Final pass guarantees temporal consistency:

```
P(12h) ≤ P(24h) ≤ P(48h) ≤ P(72h)  ∀ events
```

---

## 🛠 Tech Stack

<p>
  <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn"/>
  <img src="https://img.shields.io/badge/LightGBM-9ACD32?style=for-the-badge" alt="LightGBM"/>
  <img src="https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white" alt="SciPy"/>
</p>

---

## 👤 Author

**Vaibhav Chavanpatil**
- GitHub: [@VAIBHAV7848](https://github.com/VAIBHAV7848)
- Competition: [WiDS Global Datathon 2026](https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26)

---

<p align="center">
  <sub>Built with 🔥 for the WiDS Global Datathon 2026</sub>
</p>
