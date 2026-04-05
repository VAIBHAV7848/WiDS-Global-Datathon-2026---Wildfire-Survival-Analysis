<div align="center">

<img src="https://img.shields.io/badge/WiDS_Global_Datathon_2026-Wildfire_Survival_Analysis-FF4500?style=for-the-badge&logoColor=white" alt="WiDS 2026" />

<br>

<h1 style="border-bottom: none; margin-bottom: 0;">APEX: Wildfire Collision Engine</h1>

<p align="center" style="font-size: 1.2rem; color: #8b949e;">
  <strong>Right-censored survival estimators paired with rigid physics bounds to predict grid infrastructure intersection.</strong>
</p>

<p align="center">
<a href="https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26">
  <img src="https://img.shields.io/badge/Kaggle_Rank-Target_Top_100-20BEFF?style=flat-square&logo=kaggle&logoColor=white" alt="Kaggle Track" />
</a>
<a href="https://github.com/VAIBHAV7848">
  <img src="https://img.shields.io/badge/Author-Vaibhav-blueviolet?style=flat-square&logo=github&logoColor=white" alt="Author" />
</a>
<img src="https://img.shields.io/badge/Pipeline-Deterministic%20ML-success?style=flat-square&logo=git&logoColor=white" alt="Methodology" />
</p>

<img src="https://image.pollinations.ai/prompt/cinematic%20wildfire%20approaching%20power%20lines%20dark%20mode%20technology?width=1200&height=350&nologo=true" width="100%" style="border-radius: 12px; box-shadow: 0px 4px 20px rgba(255, 69, 0, 0.2);" alt="Hero Wildfire" />

</div>

<br>

## ✦ The Challenge

Given 5 hours of initiation telemetry from **Watch Duty**, we must predict the exact probability a wildfire will hit high-value grid infrastructure (power lines, substations) over **12h, 24h, 48h, and 72h** horizons.

Because this is a **right-censored survival problem**, standard binary classifiers fail. They mistake active, growing fires for "safe" simply because they haven't made impact *yet*. 

<br>

## ✦ Why APEX Wins (The Leaderboard Strategy)

Most Kaggle solutions blindly feed geographic coordinates into a LightGBM regressor. **APEX does not guess what physics already knows.** By imposing absolute spatial constraints, we mathematically trap the penalty margins in the Brier Score.

<div align="center">
<table>
  <tr>
    <td align="center" width="33%">
      <h3>🌍 The FAR Zone</h3>
      <p>Fires <b>&ge; 5km</b> away physically cannot bridge the gap to infrastructure in 72h.</p>
      <img src="https://img.shields.io/badge/Output-Fixed_at_0.001-3FB950?style=for-the-badge" />
    </td>
    <td align="center" width="33%">
      <h3>🔥 The ACTIVE Zone</h3>
      <p>Fires <b>&lt; 5km</b> that are actively growing represent an imminent crisis.</p>
      <img src="https://img.shields.io/badge/Output-Fixed_at_0.999-FF4500?style=for-the-badge" />
    </td>
    <td align="center" width="33%">
      <h3>🤖 The STATIC Zone</h3>
      <p>The uncertain 24%. These are routed to the heavy-compute ML stack.</p>
      <img src="https://img.shields.io/badge/Output-Survival_ML_Blend-D2A8FF?style=for-the-badge" />
    </td>
  </tr>
</table>
</div>

<br>

## ✦ System Architecture

APEX isolates the uncertain `STATIC` events and processes them identically through a dual-stack estimator matrix. 

<div align="center">
  <img src="https://image.pollinations.ai/prompt/abstract%20machine%20learning%20pipeline%20diagram%20dark%20modern%20ui%20glow?width=1000&height=250&nologo=true" width="100%" style="border-radius: 12px;" alt="Architecture Flow" />
</div>

<br>

> **GBSA (Gradient Boosting Survival Analysis):** Directly models the time-to-event objective curve.  
> **IPCW LightGBM:** Computes Inverse Probability of Censoring Weights to mathematically account for structurally unresolved time horizons.

Both models are strictly bounded by a chronologic monotonicity filter ensuring `P(12h) < P(24h) < P(48h) < P(72h)`, permanently eliminating sequential logic errors.

<br>

## ✦ Demonstration Run

Watch the pipeline seamlessly filter, route, and predict. 

<div align="center">
  <img src="https://placehold.co/900x400/0d1117/3fb950.png?text=APEX+Pipeline+Visualizer+(Simulated)" width="100%" style="border-radius: 8px;" alt="Pipeline Animation" />
</div>

<br>

<details>
<summary><b>🎬 View the Step-by-Step Logic (Click to Expand)</b></summary>

<br>

1. **Ingest Pipeline:** Batches 95 Watch Duty telemetry pings.
2. **Gate 1 (Distance &ge; 5km):** 76% of events immediately clamped to `0.001`.
3. **Gate 2 (Active/Growing):** Imminent collision anomalies flagged to `0.999`.
4. **Machine Learning Array:** The remaining "uncertain" data runs through 300 GBSA trees and 400 IPCW iterations.
5. **Array Bounding Check:** Sequential chronologic logic is mathematically enforced.
6. **Deploy:** Final predictions merged securely into `submission_A_physics.csv`.
</details>

<br>

## ✦ Quick Start

Deploy the APEX engine to your local hardware. Zero friction, completely deterministic.

```bash
# 1. Clone the environment
git clone https://github.com/VAIBHAV7848/WiDS-Global-Datathon-2026---Wildfire-Survival-Analysis.git
cd WiDS-Global-Datathon-2026---Wildfire-Survival-Analysis

# 2. Spawn dependencies
pip install -r requirements.txt

# 3. Ignite the pipeline (~5 min CPU)
python pipeline_v22_APEX.py
```

<br>

## ✦ Generated Target Profiles

APEX auto-generates a three-tiered submission matrix based on your risk tolerance on the leaderboard.

*   🥇 `submission_A_physics.csv`: **Primary Deployment.** 100% Physics Overrides merged with the integrated ML bounds. The lowest Brier penalty mathematically possible.
*   🥈 `submission_C_blend.csv`: **Conservative Base.** 60/40 blend of precise physics and machine learning arrays. Protects against edge-stage spatial anomalies in the test set.
*   🥉 `submission_B_model.csv`: **Baseline Output.** Pure gradient boosted machine learning with zero deterministic spatial gates.

<br>

## ✦ Repository Layout

```text
WiDS-Wildfire-Survival/
├── 📊 Data/
│   ├── train.csv                    # Ground-truth telemetry
│   └── test.csv                     # Hidden Kaggle targets 
├── 🔧 Pipelines/
│   ├── pipeline_v22_APEX.py         # 🚀 Target Engine (Active)
│   ├── pipeline_v21_ORACLE.py       # Legacy ensemble environment
│   └── pipeline_v20_PHYSICS.py      # Spatial constraint testbed
├── 📤 Submissions/
│   └── submission_A_physics.csv     # ★ Kaggle submission protocol
└── 📖 Docs/
    └── SUBMISSIONS.md               # Leaderboard ranking analytics
```

<br>

## ✦ Let's Connect

Architected and maintained by **Vaibhav Chavanpatil**.  
Want to discuss machine learning, quantitative survival analysis, or the WiDS Kaggle strategy? 

*   **GitHub:** [@VAIBHAV7848](https://github.com/VAIBHAV7848)

<div align="center">
  <br>
  <img src="https://img.shields.io/badge/Designed_for_WiDS_Global_Datathon-2026-1a1a2e?style=for-the-badge&logoColor=white" />
</div>
