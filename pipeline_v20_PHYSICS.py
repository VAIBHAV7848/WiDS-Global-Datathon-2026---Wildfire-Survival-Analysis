"""
Pipeline v20 PHYSICS — The 0.989+ Submission
=============================================
KEY INSIGHTS FROM DATA ANALYSIS:
1. 5km gate: dist >= 5km -> NEVER hits (0/152 in training)
2. Near + Active fire (growing): ALWAYS hits at ALL horizons (18/18)
3. Near + Static fire (not growing): hits with time-dependent probability
4. Best predictors for static fire timing: dt_first_last, num_perimeters, alignment, log1p_area
5. Weighted Brier = 0.3*B_24h + 0.4*B_48h + 0.3*B_72h (prob_12h NOT in Brier!)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import brier_score_loss
from scipy.stats import rankdata
import warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb

print("=" * 70)
print("  PIPELINE v20 PHYSICS — Three-Zone Deterministic Model")
print("=" * 70)

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
hblend = pd.read_csv("submission.csv")  # proven LB 0.97175

HORIZONS = [12, 24, 48, 72]
HORIZON_COLS = [f"prob_{h}h" for h in HORIZONS]

# =============================================
# ZONE CLASSIFICATION
# =============================================
def classify_zones(df):
    """Three zones: FAR, NEAR_ACTIVE, NEAR_STATIC"""
    far = df.dist_min_ci_0_5h >= 5000
    near = ~far
    active = near & (
        (df.radial_growth_rate_m_per_h > 0) | 
        (df.area_growth_rate_ha_per_h > 0)
    )
    static = near & ~active
    return far, active, static

train_far, train_active, train_static = classify_zones(train)
test_far, test_active, test_static = classify_zones(test)

print(f"Train: {train_far.sum()} far, {train_active.sum()} active, {train_static.sum()} static")
print(f"Test:  {test_far.sum()} far, {test_active.sum()} active, {test_static.sum()} static")

# =============================================
# ZONE A: FAR (>= 5km) — assign near-zero
# =============================================
# Training truth: 0/152 hit. Probability = 0.001
FAR_PROB = 0.001

# =============================================
# ZONE B: NEAR ACTIVE (< 5km, fire growing) — assign near-one
# =============================================
# Training truth: 18/18 hit at ALL horizons including 12h
ACTIVE_PROBS = {12: 0.999, 24: 0.999, 48: 0.999, 72: 0.999}

# =============================================
# ZONE C: NEAR STATIC (< 5km, fire NOT growing) — ML model
# =============================================
# This is the ONLY zone where prediction is non-trivial
# Training: 51 static events with different hit timing

static_train = train[train_static].copy()
static_test = test[test_static].copy()

print(f"\nStatic zone: {len(static_train)} train, {len(static_test)} test")

# Features that actually predict timing for static fires
def make_static_features(df):
    f = pd.DataFrame(index=df.index)
    f["dist_km"] = df.dist_min_ci_0_5h / 1000
    f["log_dist"] = np.log1p(f.dist_km)
    f["dt_first_last"] = df.dt_first_last_0_5h
    f["num_perimeters"] = df.num_perimeters_0_5h
    f["alignment"] = df.alignment_abs
    f["log1p_area"] = df.log1p_area_first
    f["bearing_cos"] = df.spread_bearing_cos
    f["bearing_sin"] = df.spread_bearing_sin
    f["closing_speed_abs"] = df.closing_speed_abs_m_per_h  
    f["centroid_speed"] = df.centroid_speed_m_per_h
    f["dist_slope"] = df.dist_slope_ci_0_5h
    f["start_hour"] = df.event_start_hour
    f["start_month"] = df.event_start_month
    # Interactions
    f["align_x_dt"] = f.alignment * f.dt_first_last
    f["dist_x_align"] = f.dist_km * f.alignment
    return f

X_static_train = make_static_features(static_train)
X_static_test = make_static_features(static_test)
feat_names = X_static_train.columns.tolist()

print(f"Features: {len(feat_names)}")

# Train per-horizon models with extreme multi-seed averaging
SEEDS = [42, 123, 2024, 7, 999, 314, 1337, 555, 888, 2025]
N_FOLDS = 5

static_oof = {}
static_test_preds = {}

for h in HORIZONS:
    y = (static_train.time_to_hit_hours <= h).astype(int).values
    pos_rate = y.mean()
    print(f"\n  {h}h: {y.sum()}/{len(y)} positive ({pos_rate:.3f})")
    
    if pos_rate >= 0.99:
        # All hit by this horizon — just assign high probability
        static_oof[h] = np.ones(len(static_train)) * 0.999
        static_test_preds[h] = np.ones(len(static_test)) * 0.999
        print(f"    -> All positive, assigning 0.999")
        continue
    
    X = X_static_train.values
    X_te = X_static_test.values
    
    oof = np.zeros(len(static_train))
    oof_counts = np.zeros(len(static_train))
    test_accum = np.zeros(len(static_test))
    fold_scores = []
    
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]
            
            dtrain = lgb.Dataset(X_tr, y_tr)
            dval = lgb.Dataset(X_va, y_va, reference=dtrain)
            
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "verbosity": -1,
                "seed": seed * 100 + fold,
                "num_leaves": 6,        # Very small trees
                "max_depth": 2,         # Extremely shallow
                "min_child_samples": 5,
                "reg_alpha": 5.0,       # Heavy L1
                "reg_lambda": 10.0,     # Heavy L2
                "colsample_bytree": 0.5,
                "subsample": 0.6,
                "learning_rate": 0.01,  # Very slow learning
                "n_jobs": -1,
            }
            
            model = lgb.train(
                params, dtrain, num_boost_round=1000,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            
            p_va = model.predict(X_va)
            p_te = model.predict(X_te)
            
            oof[va_idx] += p_va
            oof_counts[va_idx] += 1
            test_accum += p_te
            fold_scores.append(brier_score_loss(y_va, p_va))
    
    oof /= oof_counts  
    test_accum /= (len(SEEDS) * N_FOLDS)
    
    static_oof[h] = oof
    static_test_preds[h] = test_accum
    
    bs = np.mean(fold_scores)
    print(f"    CV Brier: {bs:.5f} +/- {np.std(fold_scores):.4f}")

# =============================================
# ASSEMBLE THREE-ZONE SUBMISSION
# =============================================
print("\n" + "=" * 70)
print("ASSEMBLING FINAL SUBMISSION")
print("=" * 70)

submission = pd.DataFrame({"event_id": test.event_id})

for h in HORIZONS:
    col = f"prob_{h}h"
    submission[col] = 0.0
    
    # Zone A: FAR
    submission.loc[test_far.values, col] = FAR_PROB
    
    # Zone B: NEAR ACTIVE
    submission.loc[test_active.values, col] = ACTIVE_PROBS[h]
    
    # Zone C: NEAR STATIC (ML model)
    submission.loc[test_static.values, col] = static_test_preds[h]

# =============================================
# BLEND WITH H_BLEND FOR STATIC ZONE ONLY
# =============================================
# The h_blend has useful ranking signal for timing within static zone
# But its FAR zone predictions are wrong (0.894), so DON'T use those
# For ACTIVE zone, our deterministic 0.999 is definitely correct

# Try different blend weights
for blend_name, our_weight in [("PURE", 1.0), ("B60", 0.6), ("B40", 0.4)]:
    sub = pd.DataFrame({"event_id": test.event_id})
    for h in HORIZONS:
        col = f"prob_{h}h"
        sub[col] = 0.0
        sub.loc[test_far.values, col] = FAR_PROB
        sub.loc[test_active.values, col] = ACTIVE_PROBS[h]
        
        ours = static_test_preds[h]
        hb = hblend.loc[test_static.values, col].values
        blended = our_weight * ours + (1 - our_weight) * hb
        sub.loc[test_static.values, col] = blended
    
    # Enforce monotonicity
    for idx in range(len(sub)):
        prev = 0
        for col in HORIZON_COLS:
            if sub.loc[idx, col] < prev:
                sub.loc[idx, col] = prev
            prev = sub.loc[idx, col]
    
    # Clip
    for col in HORIZON_COLS:
        sub[col] = sub[col].clip(0.001, 0.999)
    
    fname = f"submission_v20_{blend_name}.csv"
    sub.to_csv(fname, index=False)
    
    mono = sum(1 for i in range(95) if not all(
        sub.iloc[i][HORIZON_COLS[j]] <= sub.iloc[i][HORIZON_COLS[j+1]] for j in range(3)))
    
    print(f"\n{fname}: mono={mono}")
    print(f"  FAR:    {sub.loc[test_far.values, HORIZON_COLS].mean().to_dict()}")
    print(f"  ACTIVE: {sub.loc[test_active.values, HORIZON_COLS].mean().to_dict()}")
    print(f"  STATIC: {sub.loc[test_static.values, HORIZON_COLS].mean().to_dict()}")

# =============================================
# ALSO: Create version blending with v18_BLEND (our current best LB)
# =============================================
v18b = pd.read_csv("submission_v18_BLEND.csv")

sub_v20_mega = pd.DataFrame({"event_id": test.event_id})
for h in HORIZONS:
    col = f"prob_{h}h"
    sub_v20_mega[col] = 0.0
    sub_v20_mega.loc[test_far.values, col] = FAR_PROB
    sub_v20_mega.loc[test_active.values, col] = ACTIVE_PROBS[h]
    
    # For static: 50% our model + 50% v18_BLEND (best LB)
    ours = static_test_preds[h]
    v18_static = v18b.loc[test_static.values, col].values
    sub_v20_mega.loc[test_static.values, col] = 0.5 * ours + 0.5 * v18_static

for idx in range(len(sub_v20_mega)):
    prev = 0
    for col in HORIZON_COLS:
        if sub_v20_mega.loc[idx, col] < prev:
            sub_v20_mega.loc[idx, col] = prev
        prev = sub_v20_mega.loc[idx, col]

for col in HORIZON_COLS:
    sub_v20_mega[col] = sub_v20_mega[col].clip(0.001, 0.999)

sub_v20_mega.to_csv("submission_v20_MEGA.csv", index=False)
print(f"\nsubmission_v20_MEGA.csv (3-zone + v18_BLEND static blend):")
print(f"  STATIC: {sub_v20_mega.loc[test_static.values, HORIZON_COLS].mean().to_dict()}")

print("\n" + "=" * 70)
print("DONE. Three-zone physics model with ML-corrected static predictions.")
print("=" * 70)
