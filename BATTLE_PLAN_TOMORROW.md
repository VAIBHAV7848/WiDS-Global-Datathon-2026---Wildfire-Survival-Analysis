# 🔥 WiDS 2026 — Battle Plan for Tomorrow

## Current Situation (Honest Assessment)

| Submission | LB Score | Rank | What Changed |
|-----------|----------|------|--------------|
| submission_07 | 0.95669 | ~943 | v8 binary pipeline |
| submission_08 | 0.96025 | — | v10: 6-model Brier-optimized ensemble |
| submission_09 | 0.96310 | 673 | v11: survival models + binary blend |

**Gap to target:** 0.975 − 0.963 = **0.012** (this is HUGE in a competitive LB)

**Root cause:** OOF scores show 0.98+ but LB gives 0.963. This is **overfitting** — our models memorize the 221 training rows instead of learning the true signal.

---

## 🧠 Why Previous Approaches Hit a Wall

> **IMPORTANT:** Every approach so far treats the SAME features with the SAME modeling paradigm using MORE complexity. More models, more seeds, more trials — but all fitting the same 221 rows. **Adding complexity to an overfit system makes it worse, not better.**

| What We Tried | Why It Didn't Bridge the Gap |
|---------------|------------------------------|
| 6 models | All overfit the same 221 rows similarly |
| 100 Optuna trials | Finds params that overfit train, not params that generalize |
| Pseudo-labeling | Reinforces the model's own biased predictions |
| Complex calibration (Isotonic) | Isotonic with 221 rows = memorization |
| Survival models (GBSA/RSF) | Right direction (+0.003) but still overfit features |

---

## 📋 Tomorrow's Plan (4 Phases)

### Phase 1: Intelligence (30 min) — DO THIS FIRST

**Goal:** Find what top teams are ACTUALLY doing differently.

1. **Browse EVERY public notebook** with score > 0.97:
   - Go to: `kaggle.com/competitions/WiDSWorldWide_GlobalDathon26/code`
   - Sort by "Most Votes" and "Best Score"
   - **Read the actual code**, not just the descriptions
   - Look for: features you don't have, model tricks, calibration methods

2. **Read EVERY discussion post:**
   - Go to: `kaggle.com/competitions/WiDSWorldWide_GlobalDathon26/discussion`
   - Look for hints about: feature engineering, data leaks, metadata usage, external data

3. **Study the metadata file (`metaData.csv`):**
   - You have this file but may not be fully exploiting it
   - It may contain feature descriptions that hint at better transformations
   - Check if any features have special meaning you're not using

4. **Write down** the top 3 insights you find before coding anything.

---

### Phase 2: Feature Revolution (1 hour)

> **CAUTION:** This is probably where the real score gap lives. Top teams likely have 2-3 killer features you don't.

#### A. Physics-Based Time-to-Arrival (TTA) Feature
```
Instead of: projected_time = dist / speed

Try: Account for fire spread geometry
  - TTA_radial = (dist - fire_radius) / (closing_speed + radial_growth_rate)
  - TTA_worst_case = (dist - fire_radius - projected_advance) / max(closing_speed, 0.001)
  - For each horizon h: threat_h = sigmoid(-(TTA - h) / h * 5)
```
This gives a **smooth, horizon-aware threat score** instead of a binary danger flag.

#### B. Conditional Features (Regime-Based)
```
The fire behaves DIFFERENTLY depending on conditions:
  - approaching_features = features * is_approaching (mask non-approaching fires)  
  - close_features = features * is_close (amplify signal for nearby fires)
  - night_threat = features * night_fire (different dynamics at night)
```

#### C. Rank-Based Features
```
For each raw feature, add its rank (percentile) version:
  - rank_dist = rankdata(dist) / n
  - rank_speed = rankdata(closing_speed) / n
These are MORE ROBUST to outliers than raw values
```

#### D. Ratio Features (proven powerful in survival)
```
  - dist_to_growth_ratio = dist / (area_growth_rate + 1)
  - speed_to_dist_ratio = closing_speed / (dist + 100)  
  - growth_momentum = area_growth_rate * radial_growth_rate
```

#### E. Feature Selection: LESS IS MORE
```
Try training with ONLY these 8 features:
  1. dist_min_ci_0_5h
  2. closing_speed_m_per_h
  3. alignment_abs
  4. projected_time_to_hit (engineered)
  5. near_miss_margin (engineered)
  6. risk_score (engineered)
  7. hazard_ratio_proxy (engineered)
  8. area_first_ha

If 8-feature model scores BETTER on LB than 25-feature model,
then you've been overfitting through feature space.
```

---

### Phase 3: Anti-Overfit Modeling (1.5 hours)

> **WARNING:** The #1 priority is **reducing the OOF-LB gap**, not increasing OOF score. A model with OOF=0.970 and LB=0.970 beats a model with OOF=0.990 and LB=0.963.

#### Strategy A: EXTREME Simplicity
```python
# Try a SINGLE CatBoost with brutal regularization
CatBoostClassifier(
    depth=2,                    # VERY shallow
    learning_rate=0.02,        # VERY slow
    l2_leaf_reg=50,            # EXTREME regularization
    min_data_in_leaf=25,       # HUGE minimum (out of 221!)
    iterations=200,            # FEW trees
    subsample=0.6,             # Heavy subsampling
    rsm=0.4,                   # Only see 40% of features per split
)
```
Sometimes a single well-regularized model beats a complex ensemble.

#### Strategy B: Leave-One-Out Cross-Validation (LOOCV)
```
With only 221 rows, 5-fold CV wastes 20% for validation.
LOOCV uses 220/221 for training = maximum data usage.
It's slow but with a simple model it's feasible.
```

#### Strategy C: Bayesian Target Encoding (careful!)
```python
# Instead of raw features, encode using target statistics
# But with LOO smoothing to prevent leakage:
for feature in categorical_candidates:
    global_mean = y.mean()
    group_means = train.groupby(feature)['event'].transform('mean')
    group_counts = train.groupby(feature)['event'].transform('count')
    smoothing = 1 / (1 + np.exp(-(group_counts - 5) / 1))
    encoded = smoothing * group_means + (1 - smoothing) * global_mean
```

#### Strategy D: Direct Competition Metric Optimization
```
Our Optuna optimizes Brier per-fold, but the LB uses WEIGHTED Brier:
  Weighted_Brier = 0.3 x B_24h + 0.4 x B_48h + 0.3 x B_72h

Try optimizing the EXACT competition metric as Optuna objective:
  objective = 0.3 * c_index + 0.7 * (1 - weighted_brier)
This aligns training with the exact evaluation.
```

---

### Phase 4: Smart Submission Strategy (30 min)

#### The Blending Ladder
You now have 3 LB-scored submissions. Use them strategically:

```
Your submissions and LB scores:
  sub_07: 0.95669
  sub_08: 0.96025  
  sub_09: 0.96310  <-- BEST

If your new v12 pipeline produces sub_10:
  
  Option A: Submit pure sub_10
  Option B: Blend 50% sub_10 + 50% sub_09 (hedge)
  Option C: Blend 40% sub_10 + 30% sub_09 + 30% sub_08 (max diversity)
  
ALWAYS try the pure new submission first.
If it scores WORSE than sub_09, try blending.
If blend also fails, your new pipeline is not better.
```

#### The Hill-Climbing Trick
```
If sub_10 scores X on LB:
  - X > 0.970  -->  Great! Try more aggressive variants
  - X = 0.963  -->  No improvement, need fundamentally different approach  
  - X < 0.960  -->  New pipeline overfits MORE, reduce complexity

Each submission gives you information. Use it.
```

---

## 🎯 Specific Things to Try Tomorrow (Priority Order)

### Try #1: Minimal Feature CatBoost (fastest — 20 min)
- Use ONLY 8 features listed above
- Single CatBoost, depth=2, heavy regularization
- 10-seed average for stability
- NO calibration, NO rank blend, NO isotonic
- **Expected:** if LB > 0.963, feature reduction helps

### Try #2: Survival GBSA with Optuna (if Try #1 shows promise — 40 min)
- Tune GBSA hyperparameters with Optuna (almost nobody does this!)
- Focus on: `n_estimators`, `max_depth`, `min_samples_leaf`, `learning_rate`, `subsample`
- Extract P(T<=h) from survival curves
- Blend with Try #1

### Try #3: Competition Metric as Loss (advanced — 30 min)
- Custom Optuna objective = exact competition metric
- Weight horizons by their contribution: 0.3 x B24 + 0.4 x B48 + 0.3 x B72
- This is THE most direct path to high score

### Try #4: Study and Implement a Top Notebook (if stuck — 1 hour)
- Find a public notebook with score > 0.97
- Understand their approach
- Implement their key ideas in your framework
- This is NOT cheating — it's learning from the community

---

## Timeline for Tomorrow

| Time | Task | Goal |
|------|------|------|
| First 30 min | Phase 1: Intelligence | Find 3 key insights from top notebooks |
| Next 30 min | Try #1: Minimal CatBoost | Test if fewer features = better LB |
| Next 30 min | Submit and analyze | Learn from LB feedback |
| Next 60 min | Try #2 or #3 (based on results) | Push toward 0.975 |
| Final 30 min | Smart blending | Maximize LB with blend ladder |

---

## Key Mindset Shifts for Tomorrow

1. **Stop adding complexity.** You went from 3 models to 6 models and gained only +0.003. The answer is NOT 12 models.

2. **The gap is in FEATURES, not models.** Top teams likely have 2-3 features that capture the fire physics better than yours.

3. **Trust the LB more than OOF.** If OOF says 0.98 but LB says 0.963, your OOF is lying. Optimize for LB stability.

4. **Read top notebooks.** The fastest way to improve is to learn from people who already solved it.

5. **Simple + robust > complex + fragile.** With 221 rows, a well-tuned CatBoost with 8 features might beat a 6-model ensemble with 25 features.

---

## Progress So Far

```
Day 1:  0.957 ----||--------------------------- 0.975
Day 1:  0.960 ------||------------------------- 0.975
Day 1:  0.963 --------||----------------------- 0.975
Day 2:  0.975 ---------------------------------|| TARGET
```

**Your progress: 0.957 -> 0.963 in one day.** That is real. You learned what works
and what doesn't. Tomorrow, focus on the RIGHT changes, not MORE changes.

Good luck tomorrow! You've got this. 🔥
