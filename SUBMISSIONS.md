# 🔥 WiDS 2026 — Submissions Log

**Current Best:** `0.97284` (submission_v20_MEGA.csv)  
**Goal:** Beat 0.98+

---

### 🥇 SLOT 1 — `submission_v23_SAFE.csv`
**Description:** v23 ULTRA Pipeline (IPCW-weighted LightGBM, CatBoost, LogisticRegression, RandomForest) blended 50/50 with `h_blend`. Strong clipping bounds [0.005, 0.995].  
**Why:** The safest and most robust path to 0.98+. Combines extreme ML diversity, calibration, and the proven signal from global Kaggle competition kernels. 

---

### 🥈 SLOT 2 — `submission_v23_PURE.csv`
**Description:** v23 ULTRA Pipeline with 100% our internal IPCW-weighted ensemble. Zero kernel blending.  
**Why:** If the current `h_blend` file is dragging down the maximum possible score due to calibration faults in the top notebooks, this Pure ML file will surpass it.

---

### 🥉 SLOT 3 — `submission_v23_B60.csv`
**Description:** v23 ULTRA Pipeline with a 60% ML to 40% `h_blend` ratio.  
**Why:** Slightly biases more heavily toward our newly engineered survival IPCW ensemble, testing if increased ML weight provides higher resolution timing.
