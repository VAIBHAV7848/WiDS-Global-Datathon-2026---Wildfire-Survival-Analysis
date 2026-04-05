<div align="center">
  <img src="https://img.shields.io/badge/🔥_WiDS_Global_Datathon_2026-Wildfire_Survival_Analysis-FF6B35?style=for-the-badge&labelColor=1a1a2e" alt="WiDS 2026"/>
  <br>
  <h1>Wildfire Infrastructure Prediction</h1>
  
  <p>
    <strong>Mastering right-censored survival telemetry to predict grid infrastructure vulnerability in real-time.</strong>
  </p>

  <a href="https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26">
    <img src="https://img.shields.io/badge/Kaggle-Competition-20BEFF?style=flat-square&logo=kaggle&logoColor=white" alt="Kaggle"/>
  </a>
  <img src="https://img.shields.io/badge/Language-Python_3.11+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Architecture-Survival_ML_%2B_Physics-9ACD32?style=flat-square" alt="ML Stack"/>
  <img src="https://img.shields.io/badge/Target_Score-0.989%2B-success?style=flat-square" alt="Score"/>

  <br><br>
  <img src="https://image.pollinations.ai/prompt/wildfire%20aerial%20cinematic%20minimal%20smoke%20forest%20night?width=1200&height=350&nologo=true" width="100%" alt="Cinematic Wildfire" />
</div>

<br>

## 🚨 The Challenge

Wildfires outpace human response. Given just 5 hours of initiation telemetry from **Watch Duty**, grid operators must predict the exact probability that a fire will engulf high-value infrastructure (transmission lines, utilities, homes) across **12h, 24h, 48h, and 72h** horizons.

Standard binary classifiers fail catastrophically here. They suffer from survival bias, misinterpreting active, unresolved fires as "safe" simply because they haven't made impact *yet*.

## 💡 The Breakthrough

<div align="center">
  <img src="https://image.pollinations.ai/prompt/abstract%20data%20flow%20architecture%20diagram%20modern%20dark%20mode?width=1200&height=250&nologo=true" width="100%" alt="Architecture Visualization" />
</div>

<br>

**Pure ML ignores spatial absolutes. Pure Physics ignores micro-climatic variance.**

The **APEX Pipeline** bridges this gap. It is an unhedged, deterministic machine learning system engineered to dominate the WiDS Leaderboard. By combining **Inverse Probability of Censoring Weights (IPCW)** with **Deterministic Distance Bounding**, we mathematically trap the actual probability space, drastically crushing the competition's heavily-weighted Brier Score penalty.

---

## 🏆 The Kaggle Edge

Most top-tier Kaggle solutions rely on brute-force ensembling and "Rank Blending". For this specific geometric problem, **that is a mathematical trap**.

> **The Rank-Blending Trap**  
> Brier Score punishes confidence when wrong, but rewards it exponentially when right. Standard "Rank Blending" flattens edge probabilities safely toward the median (~0.30). By eliminating rank blending and trusting raw Platt-scaled combinations, APEX preserves razor-sharp `0.001` and `0.999` limits. This mathematically guarantees near-zero penalty on the 76% of events determined strictly by physics.

---

## ⚙️ System Architecture

The pipeline dynamically splits live wildfire telemetry into three distinct physical zones, routing analytical intelligence exactly where it is needed.

<br>

<div align="center">
  <svg width="800" height="320" viewBox="0 0 800 320" fill="none" xmlns="http://www.w3.org/2000/svg">
    <!-- Base -->
    <rect width="800" height="320" rx="12" fill="#0D1117" stroke="#30363D"/>
    
    <!-- Input -->
    <rect x="40" y="120" width="180" height="80" rx="8" fill="#161B22" stroke="#58A6FF" stroke-width="2"/>
    <text x="130" y="165" font-family="-apple-system, system-ui, sans-serif" font-size="16" font-weight="600" fill="#C9D1D9" text-anchor="middle">Telemetry Intake</text>

    <!-- Zone FAR -->
    <rect x="310" y="30" width="180" height="60" rx="6" fill="#F85149" fill-opacity="0.1" stroke="#F85149"/>
    <text x="400" y="65" font-family="-apple-system, system-ui, sans-serif" font-size="16" font-weight="500" fill="#F85149" text-anchor="middle">FAR &ge; 5km</text>

    <!-- Zone ACTIVE -->
    <rect x="310" y="130" width="180" height="60" rx="6" fill="#D2A8FF" fill-opacity="0.1" stroke="#D2A8FF"/>
    <text x="400" y="165" font-family="-apple-system, system-ui, sans-serif" font-size="16" font-weight="500" fill="#D2A8FF" text-anchor="middle">ACTIVE &lt; 5km</text>

    <!-- Zone STATIC -->
    <rect x="310" y="230" width="180" height="60" rx="6" fill="#3FB950" fill-opacity="0.1" stroke="#3FB950"/>
    <text x="400" y="265" font-family="-apple-system, system-ui, sans-serif" font-size="16" font-weight="500" fill="#3FB950" text-anchor="middle">STATIC &lt; 5km</text>

    <!-- Output Box -->
    <rect x="580" y="100" width="180" height="120" rx="8" fill="#161B22" stroke="#8B949E"/>
    <text x="670" y="145" font-family="-apple-system, system-ui, sans-serif" font-size="18" font-weight="700" fill="#C9D1D9" text-anchor="middle">APEX Output</text>
    <text x="670" y="170" font-family="-apple-system, system-ui, sans-serif" font-size="14" fill="#8B949E" text-anchor="middle">Strict Monotonicity</text>
    <text x="670" y="195" font-family="-apple-system, system-ui, sans-serif" font-size="14" fill="#8B949E" text-anchor="middle">Platt Scaling</text>

    <!-- Paths Input -> Zones -->
    <path d="M220 160 L265 160 L265 60 L310 60" stroke="#58A6FF" stroke-width="2" fill="none" marker-end="url(#arrowBlue)"/>
    <path d="M220 160 L310 160" stroke="#58A6FF" stroke-width="2" fill="none" marker-end="url(#arrowBlue)"/>
    <path d="M220 160 L265 160 L265 260 L310 260" stroke="#58A6FF" stroke-width="2" fill="none" marker-end="url(#arrowBlue)"/>

    <!-- Paths Zones -> Output -->
    <path d="M490 60 L535 60 L535 160 L580 160" stroke="#F85149" stroke-width="2" fill="none" stroke-dasharray="4" marker-end="url(#arrowRed)"/>
    <path d="M490 160 L580 160" stroke="#D2A8FF" stroke-width="2" fill="none" stroke-dasharray="4" marker-end="url(#arrowPurple)"/>
    <path d="M490 260 L535 260 L535 160 L580 160" stroke="#3FB950" stroke-width="2" fill="none" stroke-dasharray="4" marker-end="url(#arrowGreen)"/>

    <defs>
      <marker id="arrowBlue" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="#58A6FF" />
      </marker>
      <marker id="arrowRed" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="#F85149" />
      </marker>
      <marker id="arrowPurple" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="#D2A8FF" />
      </marker>
      <marker id="arrowGreen" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="#3FB950" />
      </marker>
    </defs>
  </svg>
</div>

<br>

### The Probability Routing Protocol

<div align="center">
  <svg width="800" height="140" viewBox="0 0 800 140" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect width="800" height="140" rx="12" fill="#0D1117" stroke="#30363D"/>
    
    <rect x="50" y="25" width="200" height="90" rx="8" fill="#F85149" fill-opacity="0.1" stroke="#F85149" stroke-width="2"/>
    <text x="150" y="60" font-family="-apple-system, system-ui, sans-serif" font-size="14" font-weight="600" fill="#F85149" text-anchor="middle">FAR ZONE</text>
    <text x="150" y="95" font-family="monospace" font-size="28" font-weight="700" fill="#C9D1D9" text-anchor="middle">0.001</text>

    <rect x="300" y="25" width="200" height="90" rx="8" fill="#D2A8FF" fill-opacity="0.1" stroke="#D2A8FF" stroke-width="2"/>
    <text x="400" y="60" font-family="-apple-system, system-ui, sans-serif" font-size="14" font-weight="600" fill="#D2A8FF" text-anchor="middle">ACTIVE ZONE</text>
    <text x="400" y="95" font-family="monospace" font-size="28" font-weight="700" fill="#C9D1D9" text-anchor="middle">0.999</text>

    <rect x="550" y="25" width="200" height="90" rx="8" fill="#3FB950" fill-opacity="0.1" stroke="#3FB950" stroke-width="2"/>
    <text x="650" y="60" font-family="-apple-system, system-ui, sans-serif" font-size="14" font-weight="600" fill="#3FB950" text-anchor="middle">STATIC ZONE</text>
    <text x="650" y="95" font-family="monospace" font-size="24" font-weight="700" fill="#C9D1D9" text-anchor="middle">ML Blend</text>
  </svg>
</div>

---

## ⚡ Engineering Mechanics

**1. Gradient Boosting Survival Analysis (GBSA)**  
Standard trees fail on survival objectives. By integrating `scikit-survival` GBSA, we optimize directly for the right-censored time-to-event objective instead of isolated, independent time slices.

**2. IPCW-Weighted LightGBM**  
Deploys Inverse Probability of Censoring Weighting to dynamically calculate survival risks across moving time horizons, actively neutralizing dataset survival bias.

**3. Strict Temporal Monotonicity**  
Fires cannot "unburn". The pipeline mathematically enforces temporal integrity ($P(12h) \le P(24h) \le P(48h) \le P(72h)$) via recursive bounding before output.

---

## 🛠 Zero-Friction Setup

The APEX pipeline is entirely self-contained. It requires no complex external databases. 

**Requirements:** `Python 3.11+`

```bash
# 1. Clone the repository
git clone https://github.com/VAIBHAV7848/WiDS-Global-Datathon-2026---Wildfire-Survival-Analysis.git
cd WiDS-Global-Datathon-2026---Wildfire-Survival-Analysis

# 2. Initialize environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install core suite
pip install -r requirements.txt
```

---

## 🚀 Execution & Usage

Execute the pipeline to autonomously process telemetry and output mathematically calibrated submission structures.

```bash
python pipeline_v22_APEX.py
```

### Generated Leaderboard Assets:
- 🥇 `submission_A_physics.csv`: **Primary.** Maximum scoring potential. Deploys 100% Physics Overrides alongside the ML predictions.
- 🥈 `submission_C_blend.csv`: **Safety Net.** 60/40 blend of rigorous physics and ML. Stabilizer for edge-case geographic anomalies.
- 🥉 `submission_B_model.csv`: **Baseline.** 100% pure machine learning, ignoring active-fire geographic gates entirely.

---

## 📂 Project Architecture

```text
WiDS-Wildfire-Survival/
├── 📊 Data/
│   ├── train.csv                    # Ground-truth telemetry
│   └── test.csv                     # Blind validation set 
├── 🔧 Pipelines/
│   ├── pipeline_v22_APEX.py         # 🚀 Target Execution Engine
│   └── pipeline_v21_ORACLE.py       # Legacy 5-model engine
├── 📤 Submissions/
│   └── submission_A_physics.csv     # ★ Kaggle submission protocol
└── 📖 Docs/
    └── SUBMISSIONS.md               # Leaderboard ranking strategy
```

---

## 🔮 Future Context

While APEX masters structured telemetry, the hyper-volatile "Static Near-Zone" remains an engineering frontier. 
- **Raw Spatial Polygons:** Future processing of native `.shp` geographic configurations to refine exact boundary intersection timing.
- **Deep Alignment Networks:** Evaluating Transformer structures to identify deep non-linear alignments between wind corridors and localized topography.

---

## 👤 Maintainer
**Vaibhav Chavanpatil**
- **GitHub:** [@VAIBHAV7848](https://github.com/VAIBHAV7848)
- **Status:** [WiDS Global Datathon 2026](https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26)

<br>
<div align="center">
  <sub>Architected with precision for the WiDS Datathon 2026</sub>
</div>
