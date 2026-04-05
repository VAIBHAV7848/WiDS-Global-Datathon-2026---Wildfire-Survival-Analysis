<div align="center">

<!-- ═══════════════════════════════════════════════════════════════ -->
<!-- HERO SECTION -->
<!-- ═══════════════════════════════════════════════════════════════ -->

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,9,5&height=220&section=header&text=🔥%20APEX%20Engine&fontSize=52&fontAlignY=35&desc=Wildfire%20×%20Infrastructure%20Collision%20Predictor&descSize=18&descAlignY=55&animation=fadeIn&fontColor=ffffff" width="100%"/>

<br>

[![Kaggle Competition](https://img.shields.io/badge/WiDS_Global_Datathon-2026-FF6F00?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26)
&nbsp;
![Best Score](https://img.shields.io/badge/Best_LB_Score-0.97216-00C853?style=for-the-badge&logo=target&logoColor=white)
&nbsp;
[![Author](https://img.shields.io/badge/by-Vaibhav_Chavanpatil-8B5CF6?style=for-the-badge&logo=github&logoColor=white)](https://github.com/VAIBHAV7848)

<br>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&pause=1000&color=FF6F00&center=true&vCenter=true&multiline=true&repeat=false&width=700&height=80&lines=Right-censored+survival+analysis;powered+by+physics-constrained+ML+ensemble" alt="Typing SVG" />

<br>

<img src="https://image.pollinations.ai/prompt/dramatic%20wildfire%20approaching%20electrical%20power%20transmission%20lines%20at%20night%20cinematic%20orange%20glow%20dark%20background%20realistic?width=1200&height=300&nologo=true" width="100%" style="border-radius: 16px;" alt="Wildfire approaching infrastructure"/>

</div>

<br>

## 🎯 The Problem

> **Predict** the probability that an active wildfire intersects high-value grid infrastructure (transmission lines, substations, roads) at **12h, 24h, 48h, and 72h** time horizons.

This is a **right-censored survival analysis** problem — fires that haven't hit infrastructure *yet* aren't necessarily safe. Standard binary classification fails here. APEX combines **hard physics constraints** with **multi-model ML ensembles** to solve it.

<br>

## 📊 Competition Metric

```
Hybrid Score = 0.3 × C-index  +  0.7 × (1 − Weighted Brier)

Weighted Brier = 0.3 × B(24h)  +  0.4 × B(48h)  +  0.3 × B(72h)
```

> **70% of the score is calibration accuracy** — getting probability values right matters 2.3× more than ranking events correctly.

<br>

## ⚡ The APEX Strategy — 3-Zone Gate

Training data reveals a **perfect deterministic split** that most competitors miss:

<div align="center">

```mermaid
flowchart TD
    A["🔥 WILDFIRE EVENT"] --> B{"📏 Distance to Infrastructure"}
    
    B -->|"≥ 5 km"| C["🌍 FAR ZONE"]
    B -->|"< 5 km"| D{"📡 Fire Status"}
    
    D -->|"Growing / Active"| E["🔥 ACTIVE ZONE"]
    D -->|"Static / Still"| F["🤖 STATIC ZONE"]
    
    C --> G["✅ P = 0.001\n0/152 hit in training"]
    E --> H["⚠️ P = 0.999\n100% hit in training"]
    F --> I["🧠 ML Ensemble\n5 models × 10 seeds × 5 folds"]

    style A fill:#1a1a2e,stroke:#FF6F00,color:#FF6F00,stroke-width:3px
    style B fill:#16213e,stroke:#00BCD4,color:#E0E0E0,stroke-width:2px
    style C fill:#1B5E20,stroke:#4CAF50,color:#C8E6C9,stroke-width:2px
    style D fill:#16213e,stroke:#FF9800,color:#E0E0E0,stroke-width:2px
    style E fill:#B71C1C,stroke:#FF5252,color:#FFCDD2,stroke-width:2px
    style F fill:#4A148C,stroke:#CE93D8,color:#E1BEE7,stroke-width:2px
    style G fill:#2E7D32,stroke:#66BB6A,color:#E8F5E9,stroke-width:2px
    style H fill:#D84315,stroke:#FF7043,color:#FBE9E7,stroke-width:2px
    style I fill:#6A1B9A,stroke:#AB47BC,color:#F3E5F5,stroke-width:2px
```

</div>

<div align="center">
<table>
<tr>
<td align="center" width="33%">

### 🌍 FAR Zone
**Distance ≥ 5km**

Training: **0 / 152** hit

```
P = 0.001
```
*Physics says: impossible*

</td>
<td align="center" width="33%">

### 🔥 ACTIVE Zone
**< 5km + Growing**

Training: **100%** hit

```
P = 0.999
```
*Physics says: certain*

</td>
<td align="center" width="33%">

### 🤖 STATIC Zone
**< 5km + Still**

Training: **uncertain**

```
P = ML Ensemble
```
*The real battleground*

</td>
</tr>
</table>
</div>

<br>

## 🧠 ML Architecture (Static Zone)

The **~26 uncertain events** pass through a heavy-compute ensemble:

```mermaid
flowchart TD
    A["📥 STATIC ZONE INPUT\n26 uncertain fire events"] --> B["⚙️ FEATURE ENGINEERING\n15 physics-grounded features"]
    
    B --> C["LightGBM"]
    B --> D["GradientBoosting"]
    B --> E["LogisticRegression"]
    B --> F["CatBoost"]
    B --> G["RandomForest"]
    
    C --> H["🔄 ENSEMBLE AVERAGING\n10 seeds × 5 folds = 250 models\n+ h_blend ranking signal"]
    D --> H
    E --> H
    F --> H
    G --> H
    
    H --> I["📐 MONOTONICITY ENFORCEMENT\nP 12h ≤ P 24h ≤ P 48h ≤ P 72h"]
    I --> J["✂️ CLIP to 0.001 — 0.999"]
    J --> K["📤 FINAL PREDICTION"]

    style A fill:#4A148C,stroke:#CE93D8,color:#E1BEE7,stroke-width:2px
    style B fill:#1A237E,stroke:#5C6BC0,color:#C5CAE9,stroke-width:2px
    style C fill:#E65100,stroke:#FF9800,color:#FFF3E0,stroke-width:2px
    style D fill:#1B5E20,stroke:#66BB6A,color:#E8F5E9,stroke-width:2px
    style E fill:#0D47A1,stroke:#42A5F5,color:#E3F2FD,stroke-width:2px
    style F fill:#F9A825,stroke:#FFEE58,color:#1a1a2e,stroke-width:2px
    style G fill:#BF360C,stroke:#FF7043,color:#FBE9E7,stroke-width:2px
    style H fill:#311B92,stroke:#7C4DFF,color:#EDE7F6,stroke-width:2px
    style I fill:#006064,stroke:#26C6DA,color:#E0F7FA,stroke-width:2px
    style J fill:#880E4F,stroke:#F06292,color:#FCE4EC,stroke-width:2px
    style K fill:#2E7D32,stroke:#66BB6A,color:#E8F5E9,stroke-width:3px
```

<br>

## 📈 Score Progression

<div align="center">

```
Score    File                     What Changed
─────    ────────────────────     ─────────────────────────────────
0.9576   submission_06.csv        v7.2 — 700 model geometric blend
0.9593   submission_07.csv        submission_07
0.9596   submission_13.csv        Upgrade iteration
0.9603   submission_08.csv        v10 — 6 model + pseudo-labels
0.9610   submission_10_blend      v8 — CBSA+RSF+AFT+LGBM
0.9631   submission_09.csv        v11 — GBSA+RSF+CB+LGBM survival
─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─    ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
0.9718   submission.csv           h_blend of 4 public notebooks ← BIG JUMP
0.9719   v18_SAFE.csv             + 5km distance gate
0.9722   v18_BLEND.csv       ★   + 50% ML signal → BEST SCORE
```

</div>

> **Key insight**: The +0.009 jump from 0.963 → 0.972 came from **blending top public Kaggle kernels** via the h_blend algorithm — not from building a better solo model.

<br>

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/VAIBHAV7848/WiDS-Global-Datathon-2026---Wildfire-Survival-Analysis.git
cd WiDS-Global-Datathon-2026---Wildfire-Survival-Analysis

# Install
pip install -r requirements.txt

# Run the pipeline (~5 min on CPU)
python pipeline_v21_ORACLE.py
```

<br>

## 📂 Repository Structure

```
.
├── 🔧 Pipelines
│   ├── pipeline_v22_APEX.py          # Latest experimental engine
│   ├── pipeline_v21_ORACLE.py        # 3-zone gate + multi-model ensemble
│   ├── pipeline_v20_PHYSICS.py       # Physics constraint testbed
│   └── h_blend_ensemble.py           # h_blend replication (LB = 0.97175)
│
├── 📤 Submissions
│   ├── submission_v21_C.csv          # 3-zone + 40%ML + 60%h_blend
│   ├── submission_v21_A.csv          # 3-zone + pure ML
│   ├── submission_v20_MEGA.csv       # 3-zone + alt calibration
│   ├── submission_v18_BLEND.csv      # ★ Best LB = 0.97216
│   └── submission.csv                # h_blend baseline (LB = 0.97175)
│
├── 📊 Data
│   ├── train.csv                     # 221 events (69 hits, 152 censored)
│   ├── test.csv                      # 95 events to predict
│   ├── metaData.csv                  # Feature descriptions
│   └── sample_submission.csv         # Submission format
│
└── 📖 Docs
    ├── README.md                     # You are here
    ├── SUBMISSIONS.md                # Today's submission plan
    └── Analytics Engine.md                     # Automated workflow configuration
```

<br>

## 🔑 Key Discoveries

<table>
<tr>
<td>

**📐 The 5km Perfect Split**

In 221 training events, every fire within 5km hit infrastructure. Every fire beyond 5km did not. Zero exceptions.

</td>
<td>

**📊 Metric is 70% Brier**

Calibration matters 2.3× more than ranking. Getting confident when you should be confident is rewarded heavily.

</td>
</tr>
<tr>
<td>

**🤝 Ensemble > Solo Model**

Our best solo ML scored 0.963. Blending 4 public notebooks via h_blend scored 0.972. Humility > engineering.

</td>
<td>

**🎯 The Gate is Worth +0.002**

Simply assigning 0.005 to far-zone events improved the h_blend score from 0.97175 → 0.97194. Free points.

</td>
</tr>
</table>

<br>

## 🛠️ Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.x-9ACD32?style=flat-square)
![CatBoost](https://img.shields.io/badge/CatBoost-1.x-FFCC00?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?style=flat-square&logo=numpy&logoColor=white)

</div>

<br>

## 👤 Author

<div align="center">

**Vaibhav Chavanpatil**

[![GitHub](https://img.shields.io/badge/GitHub-@VAIBHAV7848-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/VAIBHAV7848)
[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/vaibhavchavanpatil)

</div>

<br>

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,9,5&height=120&section=footer&animation=fadeIn" width="100%"/>

</div>
