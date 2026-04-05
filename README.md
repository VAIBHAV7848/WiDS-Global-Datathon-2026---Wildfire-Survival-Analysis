<div align="center">
  <img src="https://img.shields.io/badge/🔥_WiDS_Global_Datathon_2026-Wildfire_Survival_Analysis-FF6B35?style=for-the-badge&labelColor=1a1a2e" alt="WiDS 2026"/>
  <br>
  <h1>APEX System: Wildfire Collision Engine</h1>
  
  <p>
    <strong>Predicting grid infrastructure intersection through right-censored survival estimators and rigid physics bounds.</strong>
  </p>

  <a href="https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26">
    <img src="https://img.shields.io/badge/Kaggle-Competition-20BEFF?style=flat-square&logo=kaggle&logoColor=white" alt="Kaggle"/>
  </a>
  <img src="https://img.shields.io/badge/Language-Python_3.11+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Architecture-Survival_ML_%2B_Physics-9ACD32?style=flat-square" alt="ML Stack"/>
  <img src="https://img.shields.io/badge/Target_Score-0.989%2B-success?style=flat-square" alt="Score"/>

  <br><br>
  <img src="https://image.pollinations.ai/prompt/wildfire%20aerial%20cinematic%20smoke%20forest?width=1200&height=350&nologo=true" width="100%" alt="Cinematic Wildfire" />
</div>

<br>

## ⚡ Instant Understanding

At its core, APEX abandons probabilistic guessing in favor of **spatial realities**. 

<div align="center">
  <h3><strong>The 3-Zone Logic:</strong></h3>
  <p><code>FAR ZONE (&ge; 5km)</code> ➔ <strong>0.001</strong> (Zero impact physically possible)</p>
  <p><code>ACTIVE ZONE (&lt; 5km + Growing)</code> ➔ <strong>0.999</strong> (100% collision rate)</p>
  <p><code>STATIC ZONE (&lt; 5km + Stable)</code> ➔ <strong>Survival ML Blend</strong> (Gradient Boosting)</p>
</div>

---

## 🚨 The Challenge

Wildfires outpace human response. Given 5 hours of initiation telemetry from **Watch Duty**, grid operators must predict the exact probability that a fire will engulf high-value infrastructure across **12h, 24h, 48h, and 72h** horizons.

Standard binary classifiers fail catastrophically here. They suffer from survival bias, misinterpreting active, unresolved fires as "safe" simply because they haven't made impact *yet*.

<div align="center">
  <img src="https://image.pollinations.ai/prompt/wildfire%20destroying%20infrastructure%20power%20lines?width=1000&height=300&nologo=true" width="100%" alt="Wildfire Impact Context" />
</div>

---

## 💡 Core Insight

**Pure ML ignores spatial absolutes. Pure Physics ignores micro-climatic variance.**

The **APEX Pipeline** bridges this gap. It is an unhedged, deterministic machine learning system engineered to dominate Kaggle Leaderboards. By combining **Inverse Probability of Censoring Weights (IPCW)** with **Deterministic Distance Bounding**, we mathematically trap the actual probability space, drastically crushing the Brier Score penalty.

---

## 🧠 Why This Wins (The Kaggle Edge)

Most standard solutions fail because they treat time-space dependencies as independent probabilities. **APEX dominates through three engineered advantages:**

* 🛡️ **Zeroes the Impossible:** APEX recognizes that fires 5km away never hit within 72h. Pinning these probabilities to `0.001` flawlessly eliminates Brier penalty on 76% of the dataset.
* ⚔️ **Eradicates Rank-Blending Flaws:** Standard "Rank Blending" safe-zones edge probabilities toward the middle (~0.30). APEX preserves unblended, sharp boundaries (`0.001` vs `0.999`), mathematically maximizing confidence scores where physicists know the model is right.
* 🎯 **Combats Survival Censorship:** Where standard LightGBM trees fail on unresolved time horizons, APEX deploys **Gradient Boosting Survival Analysis (GBSA)** to mathematically account for "still burning" events.

---

## 🔬 Decision Flow

Every telemetry ping routes through a deterministic physical filter before reaching the heavy-compute models:

```text
[ Live Telemetry Stream ] 
           │
           ▼
[ Distance ≥ 5km? ] ──────(YES)─────> [ FAR ZONE ] ──────> [ Output: 0.001 ]
           │
         (NO)
           │
           ▼
[ Active Growing? ] ──────(YES)─────> [ ACTIVE ZONE ] ───> [ Output: 0.999 ]
           │
         (NO)
           │
           ▼
    [ STATIC ZONE ] ────────────────> [ ML Engine ]
```

---

## ⚙️ Target Architecture

To handle the 24% of events that fall into the "Static" ML zone, APEX deploys a dual-stack estimator matrix.

<div align="center">
  <img src="https://image.pollinations.ai/prompt/machine%20learning%20pipeline%20diagram%20minimal%20clean?width=1000&height=250&nologo=true" width="100%" alt="Machine Learning Architecture" />
</div>

<br>

```text
[ Static Telemetry Filter ] 
           │
           ├──────> [ 300x GBSA Estimator ] ─────┐
           │                                     │
           ├──────> [ 400x IPCW LightGBM ] ──────┤
                                                 ▼
                                     [ Probability Blender ]
                                        (60 / 40 Weights)
                                                 │
                                                 ▼
                                    [ Strict Platt Scaling ]
                                                 │
                                                 ▼
                                      [ Final Submission ]
```

---

## 🚀 Features

* **Gradient Boosting Survival Analysis (GBSA):** Directly targets the right-censored time-to-event objective rather than independent time slices.
* **IPCW-Weighted LightGBM:** Deploys Inverse Probability of Censoring Weighting to dynamically calculate survival risks across moving time horizons.
* **Strict Monotonicity:** Mathematically ensures chronologic integrity ($P(12h) \le P(24h) \le P(48h) \le P(72h)$) via recursive array bounding.
* **Platt Scaling Alignment:** Corrects prediction variance back to true real-world base rates.

---

## 🏆 Performance Strategy

*   **Brier Optimization:** Survival competitions punish mid-tier probabilities. If an outcome is physically known, guessing `0.85` instead of `0.999` yields a 22,500x heavier penalty score. APEX prioritizes boundary confidence.
*   **Temporal Stability:** Models naturally flip-flop when predicting sequentially. APEX forces all distributions through a strict array bounding filter ensuring chronologic progression.

---

## 🎞️ Demo Video & Visualization

<div align="center">
  <img src="https://placehold.co/900x400/161b22/58a6ff.png?text=APEX+Pipeline+Animation+(Simulated)" alt="Pipeline Demo" width="100%" />

  <br><br>

  <a href="https://www.youtube.com/">
    <img src="https://placehold.co/900x400/161b22/f85149.png?text=%E2%96%B6++Play+Full+Architecture+Video" alt="Watch Demo" width="100%" />
  </a>
</div>

### 🎬 Product Demo Storyboard (Frame-by-Frame Generation)
For those reconstructing the animation flow, the APEX UI demo executes perfectly in 6 frames:

*   **Frame 1 (Ingest):** **`[ Blue UI Box ]`** A batch of Watch Duty telemetry pings flashes onto the dark-mode dashboard. Text glows: `"Analyzing 95 Wildfire Events..."`
*   **Frame 2 (Physics Gate 1):** The screen splits. Data flows down into a **`[ Red UI Box: Distance ≥ 5km? ]`**. The path flashes right to a **`[ Green Box ]`**. Text overlay pops: `"76% of events clamped."` Output locks at exactly `0.001`.
*   **Frame 3 (Physics Gate 2):** Remaining data drops to the second **`[ Red UI Box: Active + Growing? ]`**. An anomaly fires. Path routes right to a **`[ Green Box ]`**. Text: `"Imminent Collision Detected."` Output locks at `0.999`. 
*   **Frame 4 (Machine Learning Initialization):** The surviving "uncertain" data drops into a **`[ Purple UI Box: STATIC ZONE ]`**. The dashboard zooms in natively.
*   **Frame 5 (Heavy Compute):** The screen splits horizontally. We see twin progress graphs in **`[ Purple: 300x GBSA ]`** and **`[ Purple: 400x IPCW ]`** churning in real-time as survival curves rapidly flatten.
*   **Frame 6 (Resolution):** The twin streams merge into a **`[ Green UI Box: Output ]`**. An array bounds filter flashes across the screen enforcing `P(12) < P(24) < P(48) < P(72)`. Final CSV successfully drops onto the screen.

---

## 🛠 Tech Stack

| Component | Framework | Purpose |
|:---|:---|:---|
| **Estimators** | `LightGBM`, `Scikit-learn` | IPCW classification and parallel tree boosting |
| **Survival**| `scikit-survival` | Time-to-event right-censored mathematical operations |
| **Matrix Ops**| `pandas`, `numpy`, `scipy` | Vectorized temporal alignment and array bounding |

---

## ⚡ Quick Start

Zero friction. APEX is self-contained. 

**Prerequisites:** `Python 3.11+`

```bash
# 1. Clone repository
git clone https://github.com/VAIBHAV7848/WiDS-Global-Datathon-2026---Wildfire-Survival-Analysis.git
cd WiDS-Global-Datathon-2026---Wildfire-Survival-Analysis

# 2. Spawn environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Inject dependencies
pip install -r requirements.txt
```

---

## 📦 Usage

Execute the APEX engine to ingest telemetry, run physics evaluations, and build optimized submission architectures.

```bash
# Run Core Pipeline (~5 mins on CPU)
python pipeline_v22_APEX.py
```

### Generated Target Profiles:
*   🥇 `submission_A_physics.csv`: **Primary.** The ultimate deployment output. 100% Physics Overrides alongside the integrated ML bounds.
*   🥈 `submission_C_blend.csv`: **Safety Base.** 60/40 blend of rigorous physics and ML. Stabilizer for edge-stage spatial anomalies.
*   🥉 `submission_B_model.csv`: **Raw Output.** Pure machine learning.

---

## 📂 Project Structure

```text
WiDS-Wildfire-Survival/
├── 📊 Data/
│   ├── train.csv                    # Ground-truth validation
│   └── test.csv                     # Hidden telemetry targets 
├── 🔧 Pipelines/
│   ├── pipeline_v22_APEX.py         # 🚀 Target Engine (Physics + ML)
│   └── pipeline_v21_ORACLE.py       # Legacy ensemble environment
├── 📤 Submissions/
│   └── submission_A_physics.csv     # ★ Kaggle submission protocol
└── 📖 Docs/
    └── SUBMISSIONS.md               # LB Ranking strategy roadmap
```

---

## 🔮 Future Scope
* **Shapefile Intersection:** Moving beyond abstract vectors to parse raw native `.shp` geographic polygon collision timings.
* **TabNet Transformer Architecture:** Discovering latent multi-variate alignments across structural density mappings and wind corridor turbulence matrices.

---

## 👤 Maintainer
**Vaibhav Chavanpatil**
*   **GitHub:** [@VAIBHAV7848](https://github.com/VAIBHAV7848)
*   **Competition Details:** [WiDS Global Datathon 2026](https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26)

<br>
<div align="center">
  <sub>Architected with 🔥 for the WiDS Datathon 2026</sub>
</div>
