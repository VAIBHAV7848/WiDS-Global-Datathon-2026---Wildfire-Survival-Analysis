"""
Pipeline v25c CORRECT -- Corrected Targets + Zone-Aware Stacked Ensemble
========================================================================
CRITICAL BUG FIX: Previous pipelines used y_h = (time_to_hit_hours <= h)
which incorrectly labels censored events (event=0) as positive when their
censoring time is <= h hours.

Correct target construction:
  - event=1 + time_to_hit <= h  -> label = 1 (hit within h hours)
  - event=1 + time_to_hit > h   -> label = 0 (hit but after h hours)
  - event=0 + time_to_hit > h   -> label = 0 (censored after h, so no hit by h)
  - event=0 + time_to_hit <= h  -> EXCLUDE (censored before h, unknown outcome)

For y_72h we use event directly (as specified in the competition).

This pipeline also uses:
  - Multiple seed averaging for stability
  - Zone-aware assembly (FAR/ACTIVE/STATIC)
  - v20 blending on static zone only
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

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("=" * 70)
print("  PIPELINE v25c CORRECT -- Bug-Fixed Zone-Aware Ensemble")
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

# === CORRECT TARGET CONSTRUCTION ===
print("\n--- Correct Target Construction ---")

def make_correct_targets(df):
    """
    Correct binary targets for survival-aware classification.
    For h < 72: event=1 AND time_to_hit <= h => label=1, else 0.
    For h = 72: label = event (raw 0/1).
    Censored events with time_to_hit <= h are excluded from training.
    """
    targets = {}
    masks = {}  # valid samples mask for each horizon
    for h in [12, 24, 48]:
        # Valid: event=1 (always valid) OR event=0 with time > h (known alive through h)
        valid = (df['event'] == 1) | (df['time_to_hit_hours'] > h)
        y = ((df['time_to_hit_hours'] <= h) & (df['event'] == 1)).astype(int)
        targets[h] = y.values
        masks[h] = valid.values
    
    # 72h: just use event directly, all samples valid
    targets[72] = df['event'].values
    masks[72] = np.ones(len(df), dtype=bool)
    
    return targets, masks

y_all, y_masks = make_correct_targets(train)

for h in HORIZONS:
    n_valid = y_masks[h].sum()
    n_pos = (y_all[h][y_masks[h]]).sum()
    print(f"  y_{h}h: {n_pos}/{n_valid} positive ({n_pos/n_valid:.3f}), {len(train)-n_valid} excluded")

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

print(f"\nTrain zones: {train_far.sum()} far, {train_active.sum()} active, {train_static.sum()} static")
print(f"Test zones:  {test_far.sum()} far, {test_active.sum()} active, {test_static.sum()} static")

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
    
    df['feat1'] = cs * align_abs
    df['feat2'] = rgr * (1 - align_abs)
    df['feat3'] = np.log1p(area_first) * cspeed
    df['feat4'] = dist_min * (cs / (cs_abs + 1e-6))
    df['feat5'] = np.sin(2 * np.pi * df['event_start_month'] / 12)
    df['feat6'] = np.cos(2 * np.pi * df['event_start_month'] / 12)
    df['feat7'] = np.sin(2 * np.pi * df['event_start_hour'] / 24)
    df['feat8'] = np.cos(2 * np.pi * df['event_start_hour'] / 24)
    
    df['dist_km'] = dist_min / 1000.0
    df['log_dist'] = np.log1p(df['dist_km'])
    df['is_active'] = ((rgr > 0) | (agr > 0)).astype(int)
    df['is_near'] = (dist_min < 5000).astype(int)
    df['near_active'] = df['is_near'] * df['is_active']
    df['align_dist'] = align_abs * df['dist_km']
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

X_all = train_fe[feature_cols].values.astype(np.float64)
X_test = test_fe[feature_cols].values.astype(np.float64)

for arr in [X_all, X_test]:
    arr[np.isnan(arr)] = 0
    arr[np.isinf(arr)] = 0

# === BASE MODELS: Multi-Seed 5-Fold CV ===
print("\n" + "=" * 70)
print("BASE MODELS (6 models x 4 horizons, multi-seed 5-fold CV)")
print("=" * 70)

N_FOLDS = 5
SEEDS = [42, 123, 2024]
MODEL_NAMES = ['lgbm', 'xgb', 'catboost', 'rf', 'et', 'lr']

oof_preds = {h: {} for h in HORIZONS}
test_preds = {h: {} for h in HORIZONS}

for h in HORIZONS:
    y = y_all[h]
    mask = y_masks[h]
    X_h = X_all[mask]
    y_h = y[mask]
    
    pos_rate = y_h.mean()
    print(f"\n  Horizon {h}h: {y_h.sum()}/{len(y_h)} positive ({pos_rate:.3f})")
    
    for mname in MODEL_NAMES:
        oof = np.zeros(len(y_h))
        te_acc = np.zeros(len(test))
        fold_brier = []
        n_total = 0
        
        for seed in SEEDS:
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
            oof_seed = np.zeros(len(y_h))
            oof_count = np.zeros(len(y_h))
            
            for fi, (tr_idx, va_idx) in enumerate(skf.split(X_h, y_h)):
                Xtr, Xva = X_h[tr_idx], X_h[va_idx]
                ytr, yva = y_h[tr_idx], y_h[va_idx]
                
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
                    pva = m.predict(Xva)
                    pte = m.predict(X_test)
                    
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
                    pva = m.predict(xgb.DMatrix(Xva))
                    pte = m.predict(xgb.DMatrix(X_test))
                    
                elif mname == 'catboost':
                    m = cb.CatBoostClassifier(
                        loss_function='Logloss', iterations=2000, depth=3,
                        learning_rate=0.02, l2_leaf_reg=5.0,
                        random_seed=seed * 100 + fi, verbose=0, early_stopping_rounds=50)
                    m.fit(Xtr, ytr, eval_set=(Xva, yva), verbose=0)
                    pva = m.predict_proba(Xva)[:, 1]
                    pte = m.predict_proba(X_test)[:, 1]
                    
                elif mname == 'rf':
                    m = RandomForestClassifier(n_estimators=300, max_depth=4,
                        min_samples_leaf=5, random_state=seed * 100 + fi, n_jobs=-1)
                    m.fit(Xtr, ytr)
                    pva = m.predict_proba(Xva)[:, 1]
                    pte = m.predict_proba(X_test)[:, 1]
                    
                elif mname == 'et':
                    m = ExtraTreesClassifier(n_estimators=300, max_depth=4,
                        min_samples_leaf=5, random_state=seed * 100 + fi, n_jobs=-1)
                    m.fit(Xtr, ytr)
                    pva = m.predict_proba(Xva)[:, 1]
                    pte = m.predict_proba(X_test)[:, 1]
                    
                elif mname == 'lr':
                    sc = StandardScaler()
                    Xtr_s = sc.fit_transform(Xtr)
                    Xva_s = sc.transform(Xva)
                    Xte_s = sc.transform(X_test)
                    m = LogisticRegression(C=1.0, max_iter=1000, random_state=seed * 100 + fi)
                    m.fit(Xtr_s, ytr)
                    pva = m.predict_proba(Xva_s)[:, 1]
                    pte = m.predict_proba(Xte_s)[:, 1]
                
                oof_seed[va_idx] += pva
                oof_count[va_idx] += 1
                te_acc += pte
                fold_brier.append(brier_score_loss(yva, pva))
                n_total += 1
            
            oof_seed /= np.maximum(oof_count, 1)
            oof += oof_seed
        
        oof /= len(SEEDS)
        te_acc /= n_total
        
        # Map OOF back to full train array
        oof_full = np.full(len(train), 0.5)  # default for excluded samples
        oof_full[mask] = oof
        
        oof_preds[h][mname] = oof_full
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
    mask = y_masks[h]
    meta_tr = np.column_stack([oof_preds[h][m][mask] for m in MODEL_NAMES])
    meta_te = np.column_stack([test_preds[h][m] for m in MODEL_NAMES])
    y_h = y_all[h][mask]
    
    # Nested CV for OOF stacked predictions
    stacked_oof_h = np.zeros(mask.sum())
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for fi, (tr_idx, va_idx) in enumerate(skf.split(meta_tr, y_h)):
        meta_lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
        meta_lr.fit(meta_tr[tr_idx], y_h[tr_idx])
        stacked_oof_h[va_idx] = meta_lr.predict_proba(meta_tr[va_idx])[:, 1]
    
    # Final meta-model for test
    meta_lr_final = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    meta_lr_final.fit(meta_tr, y_h)
    stacked_test_h = meta_lr_final.predict_proba(meta_te)[:, 1]
    
    # Map back to full
    stacked_oof_full = np.full(len(train), 0.5)
    stacked_oof_full[mask] = stacked_oof_h
    
    stacked_oof[h] = stacked_oof_full
    stacked_test[h] = stacked_test_h
    
    bs = brier_score_loss(y_h, stacked_oof_h)
    print(f"  {h}h: Stacked OOF Brier = {bs:.5f}")

# === Simple average for diversity ===
simple_avg_test = {}
for h in HORIZONS:
    simple_avg_test[h] = np.mean([test_preds[h][m] for m in MODEL_NAMES], axis=0)

# === CV VALIDATION ===
print("\n" + "=" * 70)
print("CV VALIDATION")
print("=" * 70)

# For hybrid score, use only valid samples where all horizons are valid
all_valid = y_masks[12] & y_masks[24] & y_masks[48] & y_masks[72]
print(f"  Samples valid for all horizons: {all_valid.sum()}/{len(train)}")

if all_valid.sum() > 10:
    cv_score = hybrid_score(
        train['time_to_hit_hours'].values[all_valid],
        train['event'].values[all_valid],
        y_all[12][all_valid], y_all[24][all_valid],
        y_all[48][all_valid], y_all[72][all_valid],
        stacked_oof[12][all_valid], stacked_oof[24][all_valid],
        stacked_oof[48][all_valid], stacked_oof[72][all_valid]
    )
    cv_cidx = concordance_index(
        train['time_to_hit_hours'].values[all_valid],
        -stacked_oof[72][all_valid],
        train['event'].values[all_valid]
    )
    cv_brier = weighted_brier(
        y_all[12][all_valid], y_all[24][all_valid],
        y_all[48][all_valid], y_all[72][all_valid],
        stacked_oof[12][all_valid], stacked_oof[24][all_valid],
        stacked_oof[48][all_valid], stacked_oof[72][all_valid]
    )
    print(f"  CV Hybrid Score: {cv_score:.5f}")
    print(f"  CV C-index:      {cv_cidx:.5f}")
    print(f"  CV W-Brier:      {cv_brier:.5f}")

# === ZONE-AWARE ASSEMBLY ===
print("\n" + "=" * 70)
print("ZONE-AWARE ASSEMBLY + v20 BLENDING")
print("=" * 70)

BLEND_CONFIGS = [
    ("PURE_STACK", 1.0, stacked_test),
    ("B70_STACK", 0.7, stacked_test),
    ("B50_STACK", 0.5, stacked_test),
    ("B30_STACK", 0.3, stacked_test),
    ("PURE_AVG", 1.0, simple_avg_test),
    ("B50_AVG", 0.5, simple_avg_test),
    ("B30_AVG", 0.3, simple_avg_test),
]

for config_name, w_new, pred_dict in BLEND_CONFIGS:
    sub = pd.DataFrame({'event_id': test['event_id']})
    
    for h in HORIZONS:
        col = f"prob_{h}h"
        sub[col] = 0.0
        sub.loc[test_far.values, col] = 0.001
        sub.loc[test_active.values, col] = 0.999
        
        v20_static = v20.loc[test_static.values, col].values
        new_static = pred_dict[h][test_static.values]
        blended_static = w_new * new_static + (1 - w_new) * v20_static
        sub.loc[test_static.values, col] = blended_static
    
    # Monotonicity
    for idx in sub.index:
        prev = 0
        for col in HORIZON_COLS:
            if sub.loc[idx, col] < prev:
                sub.loc[idx, col] = prev
            prev = sub.loc[idx, col]
    
    for col in HORIZON_COLS:
        sub[col] = sub[col].clip(0.001, 0.999)
    
    fname = f"submission_v25c_{config_name}.csv"
    sub.to_csv(fname, index=False)
    
    static_m = sub.loc[test_static.values, HORIZON_COLS].mean()
    print(f"  {fname}:")
    print(f"    STATIC: p12={static_m['prob_12h']:.4f} p24={static_m['prob_24h']:.4f} "
          f"p48={static_m['prob_48h']:.4f} p72={static_m['prob_72h']:.4f}")

# === RANK-AVERAGE VARIANTS ===
print("\n--- Rank Average Variants ---")
from scipy.stats import rankdata

sub_rank = pd.DataFrame({'event_id': test['event_id']})
for h in HORIZONS:
    col = f"prob_{h}h"
    all_p = [test_preds[h][m] for m in MODEL_NAMES] + [stacked_test[h]]
    ranks = np.mean([rankdata(p) / len(p) for p in all_p], axis=0)
    sub_rank[col] = ranks

for h in HORIZONS:
    col = f"prob_{h}h"
    sub_rank.loc[test_far.values, col] = 0.001
    sub_rank.loc[test_active.values, col] = 0.999

for idx in sub_rank.index:
    prev = 0
    for col in HORIZON_COLS:
        if sub_rank.loc[idx, col] < prev:
            sub_rank.loc[idx, col] = prev
        prev = sub_rank.loc[idx, col]

for col in HORIZON_COLS:
    sub_rank[col] = sub_rank[col].clip(0.001, 0.999)

sub_rank.to_csv("submission_v25c_RANK.csv", index=False)
print(f"  Saved submission_v25c_RANK.csv")

for rw_name, rw in [("RB50", 0.5), ("RB30", 0.3)]:
    sub_rb = pd.DataFrame({'event_id': test['event_id']})
    for h in HORIZONS:
        col = f"prob_{h}h"
        sub_rb[col] = 0.0
        sub_rb.loc[test_far.values, col] = 0.001
        sub_rb.loc[test_active.values, col] = 0.999
        rank_s = sub_rank.loc[test_static.values, col].values
        v20_s = v20.loc[test_static.values, col].values
        sub_rb.loc[test_static.values, col] = rw * rank_s + (1 - rw) * v20_s
    
    for idx in sub_rb.index:
        prev = 0
        for col in HORIZON_COLS:
            if sub_rb.loc[idx, col] < prev:
                sub_rb.loc[idx, col] = prev
            prev = sub_rb.loc[idx, col]
    
    for col in HORIZON_COLS:
        sub_rb[col] = sub_rb[col].clip(0.001, 0.999)
    
    sub_rb.to_csv(f"submission_v25c_{rw_name}.csv", index=False)
    print(f"  Saved submission_v25c_{rw_name}.csv")

# === BEST CANDIDATE: Copy to submission_final.csv ===
# B50_STACK is likely the best balance between new model and v20 prior
import shutil
shutil.copy2("submission_v25c_B50_STACK.csv", "submission_final.csv")

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"\n  Baseline LB (v20):  0.97284")
print(f"\n  Submissions generated:")
for cfg_name, _, _ in BLEND_CONFIGS:
    print(f"    - submission_v25c_{cfg_name}.csv")
print(f"    - submission_v25c_RANK.csv")
print(f"    - submission_v25c_RB50.csv")
print(f"    - submission_v25c_RB30.csv")
print(f"\n  submission_final.csv = copy of B50_STACK")
print(f"\n  Recommended submission order:")
print(f"    1. submission_v25c_B50_STACK.csv (balanced blend)")
print(f"    2. submission_v25c_B30_STACK.csv (v20-dominant)")  
print(f"    3. submission_v25c_B50_AVG.csv   (simpler model, diversified)")
print("\n" + "=" * 70)
print("DONE.")
print("=" * 70)
