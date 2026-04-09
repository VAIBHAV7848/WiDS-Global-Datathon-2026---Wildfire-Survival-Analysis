"""
Pipeline v25b FULLCV -- Full-CV Stacked Ensemble + Zone-Aware v20 Blend
=======================================================================
Key insight: With only 222 train rows, the 80/10/10 split leaves too
few rows (22-23) for reliable calibration/blending. Instead:

Strategy:
  1) Use ALL training data in 5-fold CV for base models
  2) Stack with LogisticRegression (also via nested CV to avoid leakage)
  3) Blend with v20_MEGA using zone-aware logic:
     - FAR zone (dist >= 5km): Trust v20's 0.001 (100% correct in training)
     - ACTIVE zone (near + growing): Trust v20's 0.999 (100% correct in training)  
     - STATIC zone (near + not growing): Blend v20 with new model predictions
  4) Multiple blend variants for submission

No isotonic calibration (not enough data for a separate calibration set).
Focus on maximizing diversity and zone-aware blending.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss
from lifelines.utils import concordance_index
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("=" * 70)
print("  PIPELINE v25b FULLCV -- Zone-Aware Stacked Ensemble")
print("=" * 70)

# === METRIC ===
def weighted_brier(y12, y24, y48, y72, p12, p24, p48, p72):
    return 0.25 * (np.mean((y12-p12)**2) + np.mean((y24-p24)**2) +
                   np.mean((y48-p48)**2) + np.mean((y72-p72)**2))

def hybrid_score(yt, ye, y12, y24, y48, y72, p12, p24, p48, p72):
    ci = concordance_index(yt, -p72, ye)
    wb = weighted_brier(y12, y24, y48, y72, p12, p24, p48, p72)
    return 0.5 * ci + 0.5 * (1 - wb)

# === LOAD DATA ===
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
v20 = pd.read_csv("submission_v20_MEGA.csv")

print(f"\nTrain: {train.shape}, Test: {test.shape}")
print(f"Events: {train.event.sum()}/{len(train)} ({train.event.mean():.3f})")

HORIZONS = [12, 24, 48, 72]
HORIZON_COLS = [f"prob_{h}h" for h in HORIZONS]

# === ZONE CLASSIFICATION ===
def classify_zones(df):
    """Deterministic three-zone classification based on physics."""
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

print(f"\nTrain zones: {train_far.sum()} far, {train_active.sum()} active, {train_static.sum()} static")
print(f"Test zones:  {test_far.sum()} far, {test_active.sum()} active, {test_static.sum()} static")

# Verify zone correctness on training data
print("\n--- Zone Verification ---")
for zone_name, mask in [("FAR", train_far), ("ACTIVE", train_active), ("STATIC", train_static)]:
    subset = train[mask]
    for h in HORIZONS:
        if h < 72:
            hit = (subset.time_to_hit_hours <= h).sum()
        else:
            hit = subset.event.sum()
        print(f"  {zone_name:8s} {h:2d}h: {hit:3d}/{len(subset)} hit ({hit/max(len(subset),1):.3f})")

# === TARGETS ===
def make_targets(df):
    return {
        12: (df['time_to_hit_hours'] <= 12).astype(int).values,
        24: (df['time_to_hit_hours'] <= 24).astype(int).values,
        48: (df['time_to_hit_hours'] <= 48).astype(int).values,
        72: df['event'].values,
    }

y_all = make_targets(train)

# === FEATURE ENGINEERING ===
print("\n" + "=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)

def engineer_features(df):
    """Physics-informed feature engineering."""
    df = df.copy()
    
    cs = df['closing_speed_m_per_h'].fillna(0)
    cs_abs = df['closing_speed_abs_m_per_h'].fillna(0)
    align_abs = df['alignment_abs'].fillna(0)
    rgr = df['radial_growth_rate_m_per_h'].fillna(0)
    area_first = df['area_first_ha'].fillna(0)
    cspeed = df['centroid_speed_m_per_h'].fillna(0)
    dist_min = df['dist_min_ci_0_5h'].fillna(0)
    agr = df['area_growth_rate_ha_per_h'].fillna(0)
    
    # Core interactions (feat1-feat8 as specified)
    df['feat1'] = cs * align_abs
    df['feat2'] = rgr * (1 - align_abs)
    df['feat3'] = np.log1p(area_first) * cspeed
    df['feat4'] = dist_min * (cs / (cs_abs + 1e-6))
    df['feat5'] = np.sin(2 * np.pi * df['event_start_month'] / 12)
    df['feat6'] = np.cos(2 * np.pi * df['event_start_month'] / 12)
    df['feat7'] = np.sin(2 * np.pi * df['event_start_hour'] / 24)
    df['feat8'] = np.cos(2 * np.pi * df['event_start_hour'] / 24)
    
    # Physics proxies
    df['dist_km'] = dist_min / 1000.0
    df['log_dist'] = np.log1p(df['dist_km'])
    df['is_active'] = ((rgr > 0) | (agr > 0)).astype(int)
    df['is_near'] = (dist_min < 5000).astype(int)
    df['near_active'] = df['is_near'] * df['is_active']
    df['align_dist'] = align_abs * df['dist_km']
    
    # Extra informative features
    df['closing_ratio'] = cs / (cs_abs + 1e-6)
    df['growth_intensity'] = agr * rgr
    df['speed_dist_ratio'] = cspeed / (dist_min + 1)
    df['area_dist_ratio'] = np.log1p(area_first) / (df['dist_km'] + 0.1)
    df['dt_quality'] = df['dt_first_last_0_5h'] / (df['num_perimeters_0_5h'] + 1)
    df['along_closing'] = df['along_track_speed'].fillna(0) * align_abs
    
    return df

train_fe = engineer_features(train)
test_fe = engineer_features(test)

exclude_cols = ['event_id', 'time_to_hit_hours', 'event']
feature_cols = [c for c in train_fe.columns if c not in exclude_cols]
print(f"  Features: {len(feature_cols)}")

X_train = train_fe[feature_cols].values.astype(np.float64)
X_test = test_fe[feature_cols].values.astype(np.float64)

for arr in [X_train, X_test]:
    arr[np.isnan(arr)] = 0
    arr[np.isinf(arr)] = 0

# === BASE MODELS: 5-Fold CV ===
print("\n" + "=" * 70)
print("BASE MODELS (6 models x 4 horizons, 5-fold CV)")
print("=" * 70)

N_FOLDS = 5
SEED = 42
MODEL_NAMES = ['lgbm', 'xgb', 'catboost', 'rf', 'et', 'lr']

oof_preds = {h: {} for h in HORIZONS}
test_preds = {h: {} for h in HORIZONS}

for h in HORIZONS:
    y = y_all[h]
    pos_rate = y.mean()
    print(f"\n  Horizon {h}h: {y.sum()}/{len(y)} positive ({pos_rate:.3f})")
    
    for mname in MODEL_NAMES:
        oof = np.zeros(len(train))
        te_acc = np.zeros(len(test))
        fold_brier = []
        
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        
        for fi, (tr_idx, va_idx) in enumerate(skf.split(X_train, y)):
            Xtr, Xva = X_train[tr_idx], X_train[va_idx]
            ytr, yva = y[tr_idx], y[va_idx]
            
            if mname == 'lgbm':
                dt = lgb.Dataset(Xtr, ytr)
                dv = lgb.Dataset(Xva, yva, reference=dt)
                params = {
                    "objective": "binary", "metric": "binary_logloss",
                    "verbosity": -1, "seed": SEED + fi,
                    "num_leaves": 8, "max_depth": 3, "min_child_samples": 5,
                    "reg_alpha": 3.0, "reg_lambda": 5.0,
                    "colsample_bytree": 0.6, "subsample": 0.7,
                    "learning_rate": 0.02, "n_jobs": -1,
                }
                m = lgb.train(params, dt, num_boost_round=2000,
                    valid_sets=[dv], callbacks=[lgb.early_stopping(50, verbose=False)])
                pva = m.predict(Xva)
                pte = m.predict(X_test)
                
            elif mname == 'xgb':
                dt = xgb.DMatrix(Xtr, label=ytr)
                dv = xgb.DMatrix(Xva, label=yva)
                params = {
                    "objective": "binary:logistic", "eval_metric": "logloss",
                    "seed": SEED + fi, "max_depth": 3, "min_child_weight": 5,
                    "reg_alpha": 3.0, "reg_lambda": 5.0,
                    "colsample_bytree": 0.6, "subsample": 0.7,
                    "learning_rate": 0.02, "tree_method": "hist",
                }
                m = xgb.train(params, dt, num_boost_round=2000,
                    evals=[(dv, 'val')], early_stopping_rounds=50, verbose_eval=False)
                pva = m.predict(xgb.DMatrix(Xva))
                pte = m.predict(xgb.DMatrix(X_test))
                
            elif mname == 'catboost':
                m = cb.CatBoostClassifier(
                    loss_function='Logloss', iterations=2000, depth=3,
                    learning_rate=0.02, l2_leaf_reg=5.0,
                    random_seed=SEED + fi, verbose=0, early_stopping_rounds=50)
                m.fit(Xtr, ytr, eval_set=(Xva, yva), verbose=0)
                pva = m.predict_proba(Xva)[:, 1]
                pte = m.predict_proba(X_test)[:, 1]
                
            elif mname == 'rf':
                m = RandomForestClassifier(n_estimators=300, max_depth=4,
                    min_samples_leaf=5, random_state=SEED + fi, n_jobs=-1)
                m.fit(Xtr, ytr)
                pva = m.predict_proba(Xva)[:, 1]
                pte = m.predict_proba(X_test)[:, 1]
                
            elif mname == 'et':
                m = ExtraTreesClassifier(n_estimators=300, max_depth=4,
                    min_samples_leaf=5, random_state=SEED + fi, n_jobs=-1)
                m.fit(Xtr, ytr)
                pva = m.predict_proba(Xva)[:, 1]
                pte = m.predict_proba(X_test)[:, 1]
                
            elif mname == 'lr':
                sc = StandardScaler()
                Xtr_s = sc.fit_transform(Xtr)
                Xva_s = sc.transform(Xva)
                Xte_s = sc.transform(X_test)
                m = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED + fi)
                m.fit(Xtr_s, ytr)
                pva = m.predict_proba(Xva_s)[:, 1]
                pte = m.predict_proba(Xte_s)[:, 1]
            
            oof[va_idx] = pva
            te_acc += pte / N_FOLDS
            fold_brier.append(brier_score_loss(yva, pva))
        
        oof_preds[h][mname] = oof
        test_preds[h][mname] = te_acc
        
        bm = np.mean(fold_brier)
        print(f"    {mname:10s}: Brier={bm:.5f} +/- {np.std(fold_brier):.4f}")

# === STACKING ===
print("\n" + "=" * 70)
print("STACKING (LogisticRegression meta-model)")
print("=" * 70)

stacked_oof = {}
stacked_test = {}

for h in HORIZONS:
    meta_tr = np.column_stack([oof_preds[h][m] for m in MODEL_NAMES])
    meta_te = np.column_stack([test_preds[h][m] for m in MODEL_NAMES])
    y = y_all[h]
    
    # Use nested CV for OOF stacked predictions (avoid leakage)
    stacked_oof_h = np.zeros(len(train))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fi, (tr_idx, va_idx) in enumerate(skf.split(meta_tr, y)):
        meta_lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
        meta_lr.fit(meta_tr[tr_idx], y[tr_idx])
        stacked_oof_h[va_idx] = meta_lr.predict_proba(meta_tr[va_idx])[:, 1]
    
    # Final meta-model on all data for test predictions
    meta_lr_final = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    meta_lr_final.fit(meta_tr, y)
    stacked_test_h = meta_lr_final.predict_proba(meta_te)[:, 1]
    
    stacked_oof[h] = stacked_oof_h
    stacked_test[h] = stacked_test_h
    
    bs = brier_score_loss(y, stacked_oof_h)
    print(f"  {h}h: Stacked OOF Brier = {bs:.5f}")

# === CV HYBRID SCORE ===
print("\n" + "=" * 70)
print("CV VALIDATION (on full train OOF)")
print("=" * 70)

cv_score = hybrid_score(
    train['time_to_hit_hours'].values, train['event'].values,
    y_all[12], y_all[24], y_all[48], y_all[72],
    stacked_oof[12], stacked_oof[24], stacked_oof[48], stacked_oof[72]
)
cv_cidx = concordance_index(
    train['time_to_hit_hours'].values, -stacked_oof[72], train['event'].values
)
cv_brier = weighted_brier(
    y_all[12], y_all[24], y_all[48], y_all[72],
    stacked_oof[12], stacked_oof[24], stacked_oof[48], stacked_oof[72]
)

print(f"  CV Hybrid Score: {cv_score:.5f}")
print(f"  CV C-index:      {cv_cidx:.5f}")
print(f"  CV W-Brier:      {cv_brier:.5f}")

# Per-horizon OOF Brier
for h in HORIZONS:
    bs = brier_score_loss(y_all[h], stacked_oof[h])
    print(f"  {h}h Brier: {bs:.5f}")

# === ZONE-AWARE ASSEMBLY ===
print("\n" + "=" * 70)
print("ZONE-AWARE ASSEMBLY + v20 BLENDING")
print("=" * 70)

# Build zone-aware submission combining:
# - FAR zone: v20's 0.001 (physics certain)
# - ACTIVE zone: v20's 0.999 (physics certain)
# - STATIC zone: blend of stacked model + v20

v20_indexed = v20.set_index('event_id')

# Also build a "simple average" ensemble without stacking for diversity
simple_avg_test = {}
for h in HORIZONS:
    simple_avg_test[h] = np.mean([test_preds[h][m] for m in MODEL_NAMES], axis=0)

# Blend weights to try for the STATIC zone predictions
BLEND_CONFIGS = [
    ("PURE_STACK", 1.0, stacked_test),       # 100% new stacked model
    ("B70_STACK", 0.7, stacked_test),         # 70% stack + 30% v20
    ("B50_STACK", 0.5, stacked_test),         # 50/50
    ("B30_STACK", 0.3, stacked_test),         # 30% stack + 70% v20
    ("PURE_AVG", 1.0, simple_avg_test),       # 100% simple average
    ("B50_AVG", 0.5, simple_avg_test),        # 50% avg + 50% v20
]

for config_name, w_new, pred_dict in BLEND_CONFIGS:
    sub = pd.DataFrame({'event_id': test['event_id']})
    
    for h in HORIZONS:
        col = f"prob_{h}h"
        sub[col] = 0.0
        
        # FAR zone: deterministic 0.001
        sub.loc[test_far.values, col] = 0.001
        
        # ACTIVE zone: deterministic 0.999
        sub.loc[test_active.values, col] = 0.999
        
        # STATIC zone: blend
        v20_static = v20.loc[test_static.values, col].values
        new_static = pred_dict[h][test_static.values]
        blended_static = w_new * new_static + (1 - w_new) * v20_static
        sub.loc[test_static.values, col] = blended_static
    
    # Enforce monotonicity
    for idx in sub.index:
        prev = 0
        for col in HORIZON_COLS:
            if sub.loc[idx, col] < prev:
                sub.loc[idx, col] = prev
            prev = sub.loc[idx, col]
    
    # Clip
    for col in HORIZON_COLS:
        sub[col] = sub[col].clip(0.001, 0.999)
    
    fname = f"submission_v25b_{config_name}.csv"
    sub.to_csv(fname, index=False)
    
    # Statistics for STATIC zone
    static_means = sub.loc[test_static.values, HORIZON_COLS].mean()
    print(f"\n  {fname}:")
    print(f"    STATIC mean: {dict(zip(HORIZON_COLS, [f'{v:.4f}' for v in static_means]))}")

# === ALSO: Direct rank-average ensemble ===
print("\n" + "-" * 70)
print("RANK AVERAGE ENSEMBLE")
print("-" * 70)

from scipy.stats import rankdata

# Rank average all base models + stacked
sub_rank = pd.DataFrame({'event_id': test['event_id']})
for h in HORIZONS:
    col = f"prob_{h}h"
    all_preds = [test_preds[h][m] for m in MODEL_NAMES] + [stacked_test[h]]
    ranks = np.mean([rankdata(p) / len(p) for p in all_preds], axis=0)
    sub_rank[col] = ranks

# Apply zone-aware overrides
for h in HORIZONS:
    col = f"prob_{h}h"
    sub_rank.loc[test_far.values, col] = 0.001
    sub_rank.loc[test_active.values, col] = 0.999

# Monotonicity + clip
for idx in sub_rank.index:
    prev = 0
    for col in HORIZON_COLS:
        if sub_rank.loc[idx, col] < prev:
            sub_rank.loc[idx, col] = prev
        prev = sub_rank.loc[idx, col]

for col in HORIZON_COLS:
    sub_rank[col] = sub_rank[col].clip(0.001, 0.999)

sub_rank.to_csv("submission_v25b_RANK.csv", index=False)
print(f"  Saved submission_v25b_RANK.csv")
static_means = sub_rank.loc[test_static.values, HORIZON_COLS].mean()
print(f"    STATIC mean: {dict(zip(HORIZON_COLS, [f'{v:.4f}' for v in static_means]))}")

# === RANK-AVERAGE BLEND WITH v20 ===
for rw_name, rw in [("RB50", 0.5), ("RB30", 0.3)]:
    sub_rb = pd.DataFrame({'event_id': test['event_id']})
    for h in HORIZONS:
        col = f"prob_{h}h"
        # Blend rank-avg static with v20 static
        sub_rb[col] = 0.0
        sub_rb.loc[test_far.values, col] = 0.001
        sub_rb.loc[test_active.values, col] = 0.999
        
        rank_static = sub_rank.loc[test_static.values, col].values
        v20_static = v20.loc[test_static.values, col].values
        sub_rb.loc[test_static.values, col] = rw * rank_static + (1 - rw) * v20_static
    
    for idx in sub_rb.index:
        prev = 0
        for col in HORIZON_COLS:
            if sub_rb.loc[idx, col] < prev:
                sub_rb.loc[idx, col] = prev
            prev = sub_rb.loc[idx, col]
    
    for col in HORIZON_COLS:
        sub_rb[col] = sub_rb[col].clip(0.001, 0.999)
    
    sub_rb.to_csv(f"submission_v25b_{rw_name}.csv", index=False)
    print(f"  Saved submission_v25b_{rw_name}.csv")

# === MULTI-SEED AVERAGING VARIANT ===
print("\n" + "-" * 70)
print("MULTI-SEED AVERAGING (3 seeds x all models)")
print("-" * 70)

SEEDS = [42, 123, 2024]
ms_test_preds = {h: np.zeros(len(test)) for h in HORIZONS}
ms_oof_preds = {h: np.zeros(len(train)) for h in HORIZONS}

for seed in SEEDS:
    for h in HORIZONS:
        y = y_all[h]
        oof_seed = np.zeros(len(train))
        te_seed = np.zeros(len(test))
        n_models = 0
        
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        
        for mname in ['lgbm', 'xgb', 'catboost']:
            oof_m = np.zeros(len(train))
            te_m = np.zeros(len(test))
            
            for fi, (tr_idx, va_idx) in enumerate(skf.split(X_train, y)):
                Xtr, Xva = X_train[tr_idx], X_train[va_idx]
                ytr, yva = y[tr_idx], y[va_idx]
                
                if mname == 'lgbm':
                    dt = lgb.Dataset(Xtr, ytr)
                    dv = lgb.Dataset(Xva, yva, reference=dt)
                    params = {
                        "objective": "binary", "metric": "binary_logloss",
                        "verbosity": -1, "seed": seed * 100 + fi,
                        "num_leaves": 8, "max_depth": 3, "min_child_samples": 5,
                        "reg_alpha": 3.0, "reg_lambda": 5.0,
                        "colsample_bytree": 0.6, "subsample": 0.7,
                        "learning_rate": 0.02, "n_jobs": -1,
                    }
                    m = lgb.train(params, dt, num_boost_round=2000,
                        valid_sets=[dv], callbacks=[lgb.early_stopping(50, verbose=False)])
                    oof_m[va_idx] = m.predict(Xva)
                    te_m += m.predict(X_test) / N_FOLDS
                    
                elif mname == 'xgb':
                    dt = xgb.DMatrix(Xtr, label=ytr)
                    dv = xgb.DMatrix(Xva, label=yva)
                    params = {
                        "objective": "binary:logistic", "eval_metric": "logloss",
                        "seed": seed * 100 + fi, "max_depth": 3, "min_child_weight": 5,
                        "reg_alpha": 3.0, "reg_lambda": 5.0,
                        "colsample_bytree": 0.6, "subsample": 0.7,
                        "learning_rate": 0.02, "tree_method": "hist",
                    }
                    m = xgb.train(params, dt, num_boost_round=2000,
                        evals=[(dv, 'val')], early_stopping_rounds=50, verbose_eval=False)
                    oof_m[va_idx] = m.predict(xgb.DMatrix(Xva))
                    te_m += m.predict(xgb.DMatrix(X_test)) / N_FOLDS
                    
                elif mname == 'catboost':
                    m = cb.CatBoostClassifier(
                        loss_function='Logloss', iterations=2000, depth=3,
                        learning_rate=0.02, l2_leaf_reg=5.0,
                        random_seed=seed * 100 + fi, verbose=0, early_stopping_rounds=50)
                    m.fit(Xtr, ytr, eval_set=(Xva, yva), verbose=0)
                    oof_m[va_idx] = m.predict_proba(Xva)[:, 1]
                    te_m += m.predict_proba(X_test)[:, 1] / N_FOLDS
            
            oof_seed += oof_m
            te_seed += te_m
            n_models += 1
        
        ms_oof_preds[h] += oof_seed / n_models
        ms_test_preds[h] += te_seed / n_models

# Average across seeds
for h in HORIZONS:
    ms_oof_preds[h] /= len(SEEDS)
    ms_test_preds[h] /= len(SEEDS)

# CV score for multi-seed
ms_score = hybrid_score(
    train['time_to_hit_hours'].values, train['event'].values,
    y_all[12], y_all[24], y_all[48], y_all[72],
    ms_oof_preds[12], ms_oof_preds[24], ms_oof_preds[48], ms_oof_preds[72]
)
print(f"  Multi-seed OOF Hybrid: {ms_score:.5f}")

# Create multi-seed zone-aware submission
for ms_w_name, ms_w in [("MS_PURE", 1.0), ("MS_B50", 0.5), ("MS_B30", 0.3)]:
    sub_ms = pd.DataFrame({'event_id': test['event_id']})
    for h in HORIZONS:
        col = f"prob_{h}h"
        sub_ms[col] = 0.0
        sub_ms.loc[test_far.values, col] = 0.001
        sub_ms.loc[test_active.values, col] = 0.999
        
        new_s = ms_test_preds[h][test_static.values]
        v20_s = v20.loc[test_static.values, col].values
        sub_ms.loc[test_static.values, col] = ms_w * new_s + (1 - ms_w) * v20_s
    
    for idx in sub_ms.index:
        prev = 0
        for col in HORIZON_COLS:
            if sub_ms.loc[idx, col] < prev:
                sub_ms.loc[idx, col] = prev
            prev = sub_ms.loc[idx, col]
    
    for col in HORIZON_COLS:
        sub_ms[col] = sub_ms[col].clip(0.001, 0.999)
    
    sub_ms.to_csv(f"submission_v25b_{ms_w_name}.csv", index=False)
    print(f"  Saved submission_v25b_{ms_w_name}.csv")

# === FINAL SUMMARY ===
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"\n  Stacked CV Hybrid:    {cv_score:.5f}")
print(f"  Multi-Seed CV Hybrid: {ms_score:.5f}")
print(f"  Baseline LB (v20):   0.97284")
print(f"\n  Submissions generated:")
print(f"  - submission_v25b_PURE_STACK.csv  (100% stacked model)")
print(f"  - submission_v25b_B70_STACK.csv   (70% stack + 30% v20)")
print(f"  - submission_v25b_B50_STACK.csv   (50% stack + 50% v20)")
print(f"  - submission_v25b_B30_STACK.csv   (30% stack + 70% v20)")
print(f"  - submission_v25b_RANK.csv        (rank average ensemble)")
print(f"  - submission_v25b_RB50.csv        (50% rank + 50% v20)")
print(f"  - submission_v25b_RB30.csv        (30% rank + 70% v20)")
print(f"  - submission_v25b_MS_PURE.csv     (multi-seed pure)")
print(f"  - submission_v25b_MS_B50.csv      (multi-seed 50/50 v20)")
print(f"  - submission_v25b_MS_B30.csv      (multi-seed 30/70 v20)")
print(f"\n  Best strategy: Submit B50_STACK and MS_B50 first.")
print(f"  Zone gating ensures FAR/ACTIVE are correct; only STATIC zone varies.")
print("\n" + "=" * 70)
print("DONE.")
print("=" * 70)
