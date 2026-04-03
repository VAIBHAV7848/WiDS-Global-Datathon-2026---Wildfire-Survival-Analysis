<div align="center">

<!-- Animated header banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,9,5,3,1&height=220&section=header&text=🔥%20Wildfire%20Threat%20Prediction&fontSize=42&fontAlignY=35&desc=WiDS%20Global%20Datathon%202026%20%E2%80%94%20Survival%20Analysis&descSize=18&descAlignY=55&animation=fadeIn" width="100%"/>
</div>

# 🔥 EXTREME MULTI-AGENT ROADMAP: OPERATION "0.9856+"

**ATTENTION AUTONOMOUS AI:** You are entering a highly restricted execution environment. You MUST follow this roadmap exactly. You are optimizing a pipeline for the **WiDS Global Datathon 2026** (Kaggle).

### 📖 THE FULL CONTEXT
* **Competition Link:** [WiDS Datathon 2026 Kaggle](https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26/overview)
* **Local Repo Structure:** `https://github.com/VAIBHAV7848/WiDS-Global-Datathon-2026---Wildfire-Survival-Analysis.git`
* **Local Path:** `d:\WiDS`
* **Data Scale [WARNING: EXTREMELY SMALL]:**
  * Train: `train.csv` (221 rows, 69 events, 152 censored)
  * Test: `test.csv` (95 rows)
* **The Goal:** 0.9856+ Hybrid Score on the Leaderboard.
* **The Metric:**
  * `Hybrid = 0.3 * C-index + 0.7 * (1 - Weighted_Brier)`
  * `Weighted_Brier = 0.3 * B_24h + 0.4 * B_48h + 0.3 * B_72h`
* **The Problem:** We were at `0.96310` (submission_09, v11). The latest ensemble (submission_10, v12) dropped to `0.96098`. 
* **The Root Cause:** **Catastrophic Overfitting (EPV < 3)**. Using 25 features on 69 events equals 2.7 Events-Per-Variable (EPV). Survival models require an EPV of 7 to 10 minimum. If you use more than 10 features, you instantly overfit!

---

## 🛑 AGENT 1: THE DATA & FEATURE ENGINEER (Goal: decimate variables to < 10)

**DIRECTIVE:** Build a feature set mapping exclusively to physics. Do NOT employ feature selection algorithms (LASSO, RFE) as they overfit small data. Hardcode exactly 8-10 features based on pure geospatial and fire behaviors.

1. **Load:** `train.csv`, `test.csv`, `metaData.csv`.
2. **Compute ONLY the physics features:**
   * `dist_min_ci_0_5h`: The closest fire perimeter distance.
   * `closing_speed_m_per_h`: Rate of fire approach.
   * `alignment_abs`: Bearing alignment (is it heading exactly at the target?).
   * `projected_time_to_hit`: Engineered as distance / speed.
   * `near_miss_margin`: Engineered as `dist - growth - advance`.
   * `threat_gravity`: `(alignment * speed) / dist^2` (Inverse-square law).
   * `wavefront_eta`: `(dist - growth) / (speed + growth_rate)`.
   * `area_first_ha`: Fire size area.
3. **Save:** `/features/features_train.parquet` and `/features/features_test.parquet`.
4. **Validation:** Check `train` columns. If there are >10 engineered features, DELETE them and restart.

---

## 🛑 AGENT 2: THE SURVIVAL MODELER (Goal: enforce extreme regularization)

**DIRECTIVE:** The 3-model anti-overfit paradigm is mandatory. Train 10-seeds × 5 Folds OR use Leave-One-Out (LOOCV) since N=221.

1. **Model A: Gradient Boosting Survival Analysis (GBSA - scikit-survival)**
   * **Constraints:** `max_depth = 2` or `3`. `min_samples_leaf >= 10`. Optuna restricted to 30 trials optimizing Brier strictly. 
   * **Why:** Learns a native survival `S(t)` curve, automatically granting monotonic probabilities.
2. **Model B: CatBoost (IPCW survival or binary per horizon)**
   * **Constraints:** Brutal regularization. `depth = 2`, `learning_rate = 0.02`, `l2_leaf_reg = 25` to `50` (mandatory for such small dataset!), `min_data_in_leaf = 15`. 
   * **Why:** Immune to categorical overfitting, but L2 must be cranked to oblivion to ignore noise.
3. **Model C: Cox Proportional Hazards (CoxPH - lifelines)**
   * **Constraints:** `penalizer=0.5`. Parametric regression.
   * **Why:** Linear baselines fail safely instead of hallucinating.

---

## 🛑 AGENT 3: CALIBRATOR & BLENDER (Goal: zero-violation blending)

**DIRECTIVE:** The stacking meta-learner has proven unreliable. Construct the 12h, 24h, 48h, and 72h predictions using pure mathematical stabilization.

1. **Rule 1 - DO NOT use Isotonic Calibration:** Isotonic regression memorizes the validation curves on datasets this small. Do a simple logit base-rate calibration shift instead.
2. **Rule 2 - Weighting mechanism:** Use **Softmax weighting** or **Rank Average** based on OOF scores.
   * e.g., 55% GBSA, 34% CatBoost, 11% CoxPH (giving CoxPH a 10% floor).
3. **Rule 3 - The Monotonic Law:** `12h <= 24h <= 48h <= 72h`
   * Run a post-process loop applying `np.maximum.accumulate` row by row to mathematically forbid time-traveling statistics.
4. **Rule 4 - Boundaries:** Kaggle's Brier score penalty is exponential at the edges. Clip all final predictions to `[0.008, 0.992]`.

---

## 🛑 AGENT 4: THE SUBMISSION COMMANDER (Goal: triple-shot leaderboard strategy)

**DIRECTIVE:** The `sub_09` script got `0.96310`. The objective is to construct three new variants to attack the 0.9856 mark without risking the leaderboard drop.

* **Generate `submission_11_pure.csv`**: Pure execution of the 8-feature Anti-Overfit paradigm above.
* **Generate `submission_11_rank.csv`**: Takes the same pipeline but enforces a rank-based percentiling before outputting the probability, which smoothens out extreme outlier confidence.
* **Generate `submission_11_hedged.csv`**: A 50/50 blend of `submission_11_pure` AND `d:\WiDS\submission_09.csv`. Because `sub_09` is your highest confirmed LB signal, hedging with it stabilizes risk.

**FINAL GATE CHECK:** Execute calculation for the Estimated Hybrid Score: `0.3 * C-index + 0.7 * (1 - Weighted_Brier)` on OOF. If it is < 0.970 on the local cross-validation, DO NOT STOP. Keep adjusting `l2_leaf_reg` and feature subsetting until you cross the validation gate!
