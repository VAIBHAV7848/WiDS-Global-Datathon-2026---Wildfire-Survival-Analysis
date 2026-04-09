"""
Pipeline v25 ULTIMATE -- Maximum Score Submission
==================================================
KEY INSIGHT: Score is determined ENTIRELY by 25 STATIC test events.
FAR (67 events) = 0.001 and ACTIVE (3 events) = 0.999 are already perfect.

Strategy:
1. Train ONLY on 52 STATIC training events (all event=1)
2. Ultra-regularized models with 10-seed averaging  
3. Focus on 12h/24h discrimination (where v20 has most error)
4. For 72h: all static = 0.999 (100% hit rate in training)
5. Smart blend weights per-horizon with v20
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss
from sklearn.neighbors import KNeighborsClassifier
from lifelines.utils import concordance_index
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("=" * 70)
print("  PIPELINE v25 ULTIMATE -- Maximum Score Submission")
print("=" * 70)

# === LOAD DATA ===
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
v20 = pd.read_csv("submission_v20_MEGA.csv")

# === ZONE CLASSIFICATION ===
def classify_zones(df):
    far = df['dist_min_ci_0_5h'] >= 5000
    near = ~far
    active = near & (
        (df['radial_growth_rate_m_per_h'] > 0) |
        (df['area_growth_rate_ha_per_h'] > 0)
    )
    static = near & ~active
    return far, active, static

train_far, train_active, train_static = classify_zones(train)
test_far, test_active, test_static = classify_zones(test)

static_train = train[train_static].reset_index(drop=True)
static_test = test[test_static].reset_index(drop=True)

print(f"\nStatic train: {len(static_train)} events (all event=1)")
print(f"Static test:  {len(static_test)} events")

HORIZONS = [12, 24, 48, 72]

# === FEATURE ENGINEERING (static-focused) ===
def make_static_features(df):
    """Features specifically chosen for static zone timing prediction."""
    f = pd.DataFrame(index=df.index)
    
    # Distance features (strongest physics signal)
    f['dist_km'] = df['dist_min_ci_0_5h'] / 1000.0
    f['log_dist'] = np.log1p(f['dist_km'])
    f['dist_sq'] = f['dist_km'] ** 2
    
    # Observation quality features (r=-0.35 with time_to_hit)
    f['dt_first_last'] = df['dt_first_last_0_5h']
    f['num_perimeters'] = df['num_perimeters_0_5h']
    f['low_temporal'] = df['low_temporal_resolution_0_5h']
    
    # Alignment features (r=-0.28 with time_to_hit)
    f['alignment'] = df['alignment_abs']
    f['alignment_cos'] = df['alignment_cos']
    
    # Fire/area features
    f['log1p_area'] = df['log1p_area_first']
    f['area_first'] = df['area_first_ha']
    
    # Bearing
    f['bearing_cos'] = df['spread_bearing_cos']
    f['bearing_sin'] = df['spread_bearing_sin']
    
    # Closing dynamics
    f['closing_speed'] = df['closing_speed_m_per_h'].fillna(0)
    f['closing_speed_abs'] = df['closing_speed_abs_m_per_h'].fillna(0)
    f['centroid_speed'] = df['centroid_speed_m_per_h'].fillna(0)
    f['dist_slope'] = df['dist_slope_ci_0_5h'].fillna(0)
    
    # Time features (cyclical)
    f['hour_sin'] = np.sin(2 * np.pi * df['event_start_hour'] / 24)
    f['hour_cos'] = np.cos(2 * np.pi * df['event_start_hour'] / 24)
    f['month_sin'] = np.sin(2 * np.pi * df['event_start_month'] / 12)
    f['month_cos'] = np.cos(2 * np.pi * df['event_start_month'] / 12)
    
    # Key interactions  
    f['align_x_dt'] = f['alignment'] * f['dt_first_last']
    f['dist_x_align'] = f['dist_km'] * f['alignment']
    f['dist_x_nperim'] = f['dist_km'] * f['num_perimeters']
    
    # Along-track dynamics
    f['along_track'] = df['along_track_speed'].fillna(0)
    f['cross_track'] = df['cross_track_component'].fillna(0)
    
    return f

X_static_train = make_static_features(static_train)
X_static_test = make_static_features(static_test)
feat_names = X_static_train.columns.tolist()

print(f"Features: {len(feat_names)}")

Xtr = X_static_train.values.astype(np.float64)
Xte = X_static_test.values.astype(np.float64)

for arr in [Xtr, Xte]:
    arr[np.isnan(arr)] = 0
    arr[np.isinf(arr)] = 0

# === TARGETS ===
y_targets = {}
for h in HORIZONS:
    if h == 72:
        # All static events hit by 72h in training
        y_targets[h] = static_train['event'].values
    else:
        y_targets[h] = (static_train['time_to_hit_hours'] <= h).astype(int).values

for h in HORIZONS:
    print(f"  y_{h}h: {y_targets[h].sum()}/{len(y_targets[h])} ({y_targets[h].mean():.3f})")

# === MODEL TRAINING: Ultra-Regularized Multi-Seed Ensemble ===
print("\n" + "=" * 70)
print("TRAINING: Ultra-Regularized Multi-Seed Ensemble")
print("=" * 70)

SEEDS = [42, 123, 2024, 7, 999, 314, 1337, 555, 888, 2025]
N_FOLDS = 5

static_oof = {}
static_test_preds = {}

for h in HORIZONS:
    y = y_targets[h]
    pos_rate = y.mean()
    print(f"\n  {h}h: {y.sum()}/{len(y)} positive ({pos_rate:.3f})")
    
    # Special case: 72h is ALL positive -> predict 0.999
    if pos_rate >= 0.98:
        static_oof[h] = np.ones(len(static_train)) * 0.999
        static_test_preds[h] = np.ones(len(static_test)) * 0.999
        print(f"    -> All positive, assigning 0.999")
        continue
    
    # Special case: 48h is 94% positive -> very high base rate
    # Need good discrimination for the 3 events that don't hit by 48h
    
    # Collect predictions from ALL models x ALL seeds
    all_oof = []
    all_test = []
    all_brier = []
    
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        
        # === LightGBM (ultra-regularized) ===
        oof_lgb = np.zeros(len(static_train))
        oof_cnt = np.zeros(len(static_train))
        te_lgb = np.zeros(len(static_test))
        
        for fold, (tr_idx, va_idx) in enumerate(skf.split(Xtr, y)):
            dt = lgb.Dataset(Xtr[tr_idx], y[tr_idx])
            dv = lgb.Dataset(Xtr[va_idx], y[va_idx], reference=dt)
            params = {
                "objective": "binary", "metric": "binary_logloss",
                "verbosity": -1, "seed": seed * 100 + fold,
                "num_leaves": 4, "max_depth": 2,
                "min_child_samples": 3,
                "reg_alpha": 10.0, "reg_lambda": 20.0,
                "colsample_bytree": 0.5, "subsample": 0.5,
                "learning_rate": 0.01, "n_jobs": -1,
            }
            m = lgb.train(params, dt, num_boost_round=1500,
                valid_sets=[dv], callbacks=[lgb.early_stopping(50, verbose=False)])
            oof_lgb[va_idx] += m.predict(Xtr[va_idx])
            oof_cnt[va_idx] += 1
            te_lgb += m.predict(Xte)
        
        oof_lgb /= np.maximum(oof_cnt, 1)
        te_lgb /= (N_FOLDS)
        all_oof.append(oof_lgb)
        all_test.append(te_lgb)
        all_brier.append(brier_score_loss(y, oof_lgb))
        
        # === XGBoost (ultra-regularized) ===
        oof_xgb = np.zeros(len(static_train))
        oof_cnt = np.zeros(len(static_train))
        te_xgb = np.zeros(len(static_test))
        
        for fold, (tr_idx, va_idx) in enumerate(skf.split(Xtr, y)):
            dt = xgb.DMatrix(Xtr[tr_idx], label=y[tr_idx])
            dv = xgb.DMatrix(Xtr[va_idx], label=y[va_idx])
            params = {
                "objective": "binary:logistic", "eval_metric": "logloss",
                "seed": seed * 100 + fold, "max_depth": 2,
                "min_child_weight": 3,
                "reg_alpha": 10.0, "reg_lambda": 20.0,
                "colsample_bytree": 0.5, "subsample": 0.5,
                "learning_rate": 0.01, "tree_method": "hist",
            }
            m = xgb.train(params, dt, num_boost_round=1500,
                evals=[(dv, 'val')], early_stopping_rounds=50, verbose_eval=False)
            oof_xgb[va_idx] += m.predict(xgb.DMatrix(Xtr[va_idx]))
            oof_cnt[va_idx] += 1
            te_xgb += m.predict(xgb.DMatrix(Xte))
        
        oof_xgb /= np.maximum(oof_cnt, 1)
        te_xgb /= N_FOLDS
        all_oof.append(oof_xgb)
        all_test.append(te_xgb)
        all_brier.append(brier_score_loss(y, oof_xgb))
        
        # === CatBoost (ultra-regularized) ===
        oof_cb = np.zeros(len(static_train))
        oof_cnt = np.zeros(len(static_train))
        te_cb = np.zeros(len(static_test))
        
        for fold, (tr_idx, va_idx) in enumerate(skf.split(Xtr, y)):
            m = cb.CatBoostClassifier(
                loss_function='Logloss', iterations=1500, depth=2,
                learning_rate=0.01, l2_leaf_reg=20.0,
                random_seed=seed * 100 + fold, verbose=0,
                early_stopping_rounds=50)
            m.fit(Xtr[tr_idx], y[tr_idx], eval_set=(Xtr[va_idx], y[va_idx]), verbose=0)
            oof_cb[va_idx] += m.predict_proba(Xtr[va_idx])[:, 1]
            oof_cnt[va_idx] += 1
            te_cb += m.predict_proba(Xte)[:, 1]
        
        oof_cb /= np.maximum(oof_cnt, 1)
        te_cb /= N_FOLDS
        all_oof.append(oof_cb)
        all_test.append(te_cb)
        all_brier.append(brier_score_loss(y, oof_cb))
        
        # === RandomForest (good for small data) ===
        oof_rf = np.zeros(len(static_train))
        oof_cnt = np.zeros(len(static_train))
        te_rf = np.zeros(len(static_test))
        
        for fold, (tr_idx, va_idx) in enumerate(skf.split(Xtr, y)):
            m = RandomForestClassifier(n_estimators=500, max_depth=3,
                min_samples_leaf=3, random_state=seed * 100 + fold, n_jobs=-1)
            m.fit(Xtr[tr_idx], y[tr_idx])
            oof_rf[va_idx] += m.predict_proba(Xtr[va_idx])[:, 1]
            oof_cnt[va_idx] += 1
            te_rf += m.predict_proba(Xte)[:, 1]
        
        oof_rf /= np.maximum(oof_cnt, 1)
        te_rf /= N_FOLDS
        all_oof.append(oof_rf)
        all_test.append(te_rf)
        all_brier.append(brier_score_loss(y, oof_rf))
        
        # === LogisticRegression (strong baseline for small data) ===
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xte_s = sc.transform(Xte)
        
        oof_lr = np.zeros(len(static_train))
        oof_cnt = np.zeros(len(static_train))
        te_lr = np.zeros(len(static_test))
        
        for fold, (tr_idx, va_idx) in enumerate(skf.split(Xtr_s, y)):
            m = LogisticRegression(C=0.1, max_iter=2000, random_state=seed * 100 + fold)
            m.fit(Xtr_s[tr_idx], y[tr_idx])
            oof_lr[va_idx] += m.predict_proba(Xtr_s[va_idx])[:, 1]
            oof_cnt[va_idx] += 1
            te_lr += m.predict_proba(Xte_s)[:, 1]
        
        oof_lr /= np.maximum(oof_cnt, 1)
        te_lr /= N_FOLDS
        all_oof.append(oof_lr)
        all_test.append(te_lr)
        all_brier.append(brier_score_loss(y, oof_lr))
    
    # Grand average across ALL models x ALL seeds (10 seeds x 5 models = 50 predictors)
    oof_avg = np.mean(all_oof, axis=0)
    te_avg = np.mean(all_test, axis=0)
    
    static_oof[h] = oof_avg
    static_test_preds[h] = te_avg
    
    oof_brier = brier_score_loss(y, oof_avg)
    print(f"    Ensemble OOF Brier: {oof_brier:.5f} (from {len(all_oof)} predictors)")
    print(f"    Individual model Brier range: [{min(all_brier):.5f}, {max(all_brier):.5f}]")
    print(f"    Test pred range: [{te_avg.min():.4f}, {te_avg.max():.4f}]")

# === LOO (Leave-One-Out) VALIDATION on static zone ===
print("\n" + "=" * 70)
print("LOO VALIDATION (more reliable on 52 samples)")
print("=" * 70)

for h in [12, 24, 48]:
    y = y_targets[h]
    if y.mean() >= 0.98:
        continue
    loo_preds = np.zeros(len(static_train))
    
    for i in range(len(static_train)):
        tr_mask = np.ones(len(static_train), dtype=bool)
        tr_mask[i] = False
        
        # Simple average of LR + RF for LOO (fast)
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr[tr_mask])
        Xva_s = sc.transform(Xtr[i:i+1])
        
        lr = LogisticRegression(C=0.1, max_iter=2000, random_state=42)
        lr.fit(Xtr_s, y[tr_mask])
        p_lr = lr.predict_proba(Xva_s)[:, 1][0]
        
        rf = RandomForestClassifier(n_estimators=300, max_depth=3, min_samples_leaf=3, random_state=42)
        rf.fit(Xtr[tr_mask], y[tr_mask])
        p_rf = rf.predict_proba(Xtr[i:i+1])[:, 1][0]
        
        loo_preds[i] = 0.5 * p_lr + 0.5 * p_rf
    
    loo_brier = brier_score_loss(y, loo_preds)
    print(f"  {h}h LOO Brier: {loo_brier:.5f}")

# === ASSEMBLE FINAL SUBMISSIONS ===
print("\n" + "=" * 70)
print("ASSEMBLING SUBMISSIONS")
print("=" * 70)

v20_indexed = v20.set_index('event_id')

# Map test events to positions
test_event_ids = test['event_id'].values
static_test_ids = static_test['event_id'].values

def assemble_submission(static_preds, name, blend_w_12=1.0, blend_w_24=1.0, 
                        blend_w_48=1.0, blend_w_72=1.0):
    """Create zone-aware submission with per-horizon v20 blend weights."""
    sub = pd.DataFrame({'event_id': test['event_id']})
    
    for h in HORIZONS:
        col = f"prob_{h}h"
        sub[col] = 0.0
        
        # FAR: deterministic near-zero
        sub.loc[test_far.values, col] = 0.001
        
        # ACTIVE: deterministic near-one
        sub.loc[test_active.values, col] = 0.999
        
        # STATIC: blend our model with v20
        w_map = {12: blend_w_12, 24: blend_w_24, 48: blend_w_48, 72: blend_w_72}
        w = w_map[h]
        
        our_preds = static_preds[h]
        v20_preds = v20.loc[test_static.values, col].values
        blended = w * our_preds + (1 - w) * v20_preds
        sub.loc[test_static.values, col] = blended
    
    # Enforce monotonicity
    for idx in sub.index:
        prev = 0
        for col in ['prob_12h', 'prob_24h', 'prob_48h', 'prob_72h']:
            if sub.loc[idx, col] < prev:
                sub.loc[idx, col] = prev
            prev = sub.loc[idx, col]
    
    # Clip
    for col in ['prob_12h', 'prob_24h', 'prob_48h', 'prob_72h']:
        sub[col] = sub[col].clip(0.001, 0.999)
    
    sub.to_csv(f"submission_{name}.csv", index=False)
    
    static_means = sub.loc[test_static.values, ['prob_12h','prob_24h','prob_48h','prob_72h']].mean()
    print(f"\n  {name}:")
    print(f"    p12={static_means['prob_12h']:.4f} p24={static_means['prob_24h']:.4f} "
          f"p48={static_means['prob_48h']:.4f} p72={static_means['prob_72h']:.4f}")
    return sub

# === STRATEGY 1: Pure new model (no v20) ===
s1 = assemble_submission(static_test_preds, "v25_ULTIMATE_PURE", 1.0, 1.0, 1.0, 1.0)

# === STRATEGY 2: v20-dominant with new model correction ===
# v20 has too-uniform 12h predictions. Our model should improve discrimination.
# Blend: 40% new for 12h (most improvement), less for others
s2 = assemble_submission(static_test_preds, "v25_ULTIMATE_SMART", 0.40, 0.30, 0.20, 0.0)

# === STRATEGY 3: Conservative (mostly v20) ===
s3 = assemble_submission(static_test_preds, "v25_ULTIMATE_SAFE", 0.25, 0.20, 0.15, 0.0)

# === STRATEGY 4: Aggressive (mostly new) ===
s4 = assemble_submission(static_test_preds, "v25_ULTIMATE_BOLD", 0.70, 0.60, 0.50, 0.0)

# === STRATEGY 5: Per-horizon optimized based on OOF analysis ===
# Where does our model beat v20?
# For 12h: v20 range is 0.54-0.75 (low discrimination). Our model may have wider range.
# For 24h: v20 range is 0.83-0.92 (decent). 
# For 48h: v20 range is 0.89-0.96 (good).
# For 72h: v20 = 0.999 (perfect).
# -> Most improvement potential at 12h, some at 24h, little at 48h, none at 72h
s5 = assemble_submission(static_test_preds, "v25_ULTIMATE_PERH", 0.50, 0.35, 0.20, 0.0)

# === STRATEGY 6: Clip new model predictions more aggressively ===
# Force static predictions closer to training base rates
clipped_preds = {}
for h in HORIZONS:
    base_rate = y_targets[h].mean()  # training base rate for this horizon
    p = static_test_preds[h].copy()
    # Shrink toward base rate (Bayesian shrinkage on tiny data)
    shrinkage = 0.3  # 30% shrinkage toward base rate
    p = (1 - shrinkage) * p + shrinkage * base_rate
    clipped_preds[h] = p

s6 = assemble_submission(clipped_preds, "v25_ULTIMATE_SHRINK", 0.50, 0.35, 0.20, 0.0)

# === FINAL COMPARISON ===
print("\n" + "=" * 70)
print("COMPARISON: OOF STATIC BRIER SCORES")
print("=" * 70)

# Compute static-zone OOF Brier for our model vs what we can compute for v20
for h in [12, 24, 48]:
    y = y_targets[h]
    oof = static_oof[h]
    our_brier = brier_score_loss(y, oof)
    
    # v20 predictions for static training events (using v20 predictions mapped from test)
    # Can't directly compare -- v20 only has test predictions, not train OOF
    print(f"  {h}h: our OOF Brier = {our_brier:.5f}, test pred range = [{static_test_preds[h].min():.3f}, {static_test_preds[h].max():.3f}]")

print(f"\n  v20 test pred ranges for STATIC:")
for h in HORIZONS:
    col = f"prob_{h}h"
    v20_st = v20.loc[test_static.values, col].values
    print(f"    {h}h: [{v20_st.min():.3f}, {v20_st.max():.3f}], range={v20_st.max()-v20_st.min():.3f}")

print(f"\n  Our test pred ranges for STATIC:")
for h in HORIZONS:
    st = static_test_preds[h]
    print(f"    {h}h: [{st.min():.3f}, {st.max():.3f}], range={st.max()-st.min():.3f}")

# === RECOMMENDATION ===
print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)
print("""
  SUBMIT IN THIS ORDER:
  
  1. submission_v25_ULTIMATE_SMART.csv
     (40% new at 12h, 30% at 24h, 20% at 48h, 0% at 72h)
     -> Best balance: improves 12h discrimination where v20 is weakest
     
  2. submission_v25_ULTIMATE_PERH.csv
     (50% new at 12h, 35% at 24h, 20% at 48h, 0% at 72h)
     -> Slightly more aggressive at 12h
     
  3. submission_v25_ULTIMATE_SAFE.csv
     (25% new at 12h, 20% at 24h, 15% at 48h, 0% at 72h)
     -> Conservative: minimal risk of regression
""")

print("=" * 70)
print("DONE. All submissions saved.")
print("=" * 70)
