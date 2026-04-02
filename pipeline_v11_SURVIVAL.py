"""
pipeline_v11_SURVIVAL.py — PARADIGM SHIFT: Native Survival Models
WiDS Global Datathon 2026: Wildfire Survival Analysis

KEY INSIGHT: Top teams use PROPER survival models (scikit-survival),
NOT independent binary classifiers for each horizon.

STRATEGY:
  1. NATIVE SURVIVAL: GradientBoostingSurvivalAnalysis + RandomSurvivalForest
     → learns ONE survival curve → extract P(T≤t) at 12/24/48/72h
     → naturally monotonic, naturally handles censoring
  2. BINARY ENSEMBLE: CatBoost + LightGBM (best binary classifiers from v10)
  3. SMART BLEND: Survival models + Binary classifiers with Brier-optimized weights
  4. ULTRA-SIMPLE features (only top 15) to prevent overfit on 221 rows
"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.special import logit, expit
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Survival-specific imports
from sksurv.ensemble import (
    GradientBoostingSurvivalAnalysis,
    RandomSurvivalForest
)
from sksurv.metrics import concordance_index_censored
import time

start_time = time.time()

# ============================================================
# CONFIGURATION
# ============================================================
TIME_HORIZONS = [12, 24, 48, 72]
SEEDS = [42, 52, 62, 72, 82, 92, 102, 112, 122, 132]  # 10 seeds
N_SPLITS = 5
MONOTONICITY_INCREMENT = 0.003

# ============================================================
# DATA LOADING
# ============================================================
print("=" * 70)
print("WiDS 2026 — Pipeline v11 SURVIVAL (Paradigm Shift)")
print("=" * 70)

train = pd.read_csv('d:/WiDS/train.csv')
test = pd.read_csv('d:/WiDS/test.csv')
sample = pd.read_csv('d:/WiDS/sample_submission.csv')

print(f"Train: {train.shape}, Test: {test.shape}")

# ============================================================
# PHASE 1 — FEATURE ENGINEERING (LEAN: only proven features)
# ============================================================
print("\n" + "=" * 70)
print("PHASE 1: Feature Engineering (LEAN — max 20 features)")
print("=" * 70)

def engineer_features(df):
    out = pd.DataFrame(index=df.index)
    
    # Raw features (keep all originals)
    for col in df.columns:
        if col not in ['event_id', 'event', 'time_to_hit_hours']:
            out[col] = df[col]

    # Top proven transforms
    out['log_dist'] = np.log1p(df['dist_min_ci_0_5h'].clip(lower=0))
    out['inv_dist'] = 1.0 / (df['dist_min_ci_0_5h'].clip(lower=100) + 1)
    
    safe_closing = df['closing_speed_m_per_h'].clip(lower=0.001)
    out['projected_time_to_hit'] = (df['dist_min_ci_0_5h'] / safe_closing).clip(0, 500)
    out['log_time_projected'] = np.log1p(out['projected_time_to_hit'])
    
    out['directional_threat'] = df['closing_speed_m_per_h'] * df['alignment_abs']
    out['proximity_threat'] = df['closing_speed_m_per_h'] / (df['dist_min_ci_0_5h'] + 100)
    
    out['near_miss_margin'] = (
        df['dist_min_ci_0_5h'] - df['radial_growth_m']
        - df['projected_advance_m'].clip(lower=0)
    )
    
    dist_max = df['dist_min_ci_0_5h'].max() + 1
    speed_max = df['closing_speed_m_per_h'].abs().max() + 1
    align_max = df['alignment_abs'].max() + 1
    out['risk_score'] = (
        (1 - df['dist_min_ci_0_5h'] / dist_max) * 0.5
        + (df['closing_speed_m_per_h'] / speed_max).clip(0, 1) * 0.3
        + (df['alignment_abs'] / align_max) * 0.2
    )
    
    out['hazard_ratio_proxy'] = df['closing_speed_m_per_h'] / (df['dist_min_ci_0_5h'] + 10)
    out['is_close'] = (df['dist_min_ci_0_5h'] < 5000).astype(float)
    out['is_approaching'] = (df['closing_speed_m_per_h'] > 0).astype(float)
    out['danger_zone'] = (
        (df['dist_min_ci_0_5h'] < 5000) & (df['closing_speed_m_per_h'] > 0)
    ).astype(float)
    
    out['hour_sin'] = np.sin(2 * np.pi * df['event_start_hour'] / 24)
    out['hour_cos'] = np.cos(2 * np.pi * df['event_start_hour'] / 24)
    
    out['growth_factor_prox'] = df['radial_growth_rate_m_per_h'] / (df['dist_min_ci_0_5h'] + 1)
    out['log_area'] = np.log1p(df['area_first_ha'].clip(lower=0))
    
    out['night_fire'] = (
        (df['event_start_hour'] >= 20) | (df['event_start_hour'] <= 6)
    ).astype(float)

    return out

train_fe = engineer_features(train)
test_fe = engineer_features(test)

# Feature selection — keep only features with meaningful variance
candidate_features = [c for c in train_fe.columns
                      if c not in ['event_id', 'event', 'time_to_hit_hours']]
train_std = train_fe[candidate_features].std()
candidate_features = [f for f in candidate_features if train_std[f] > 1e-10]

# Remove highly correlated (>0.95)
PROTECTED = {
    'dist_min_ci_0_5h', 'closing_speed_m_per_h', 'alignment_abs',
    'log_dist', 'projected_time_to_hit', 'risk_score',
    'hazard_ratio_proxy', 'directional_threat', 'danger_zone',
    'near_miss_margin', 'is_close', 'night_fire'
}

corr_matrix = train_fe[candidate_features].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = set()
for col in upper.columns:
    high_corr = upper.index[upper[col] > 0.95].tolist()
    for hc in high_corr:
        if hc in PROTECTED and col not in PROTECTED:
            to_drop.add(col)
        elif col in PROTECTED and hc not in PROTECTED:
            to_drop.add(hc)
        else:
            c1 = abs(train_fe[col].corr(train['event']))
            c2 = abs(train_fe[hc].corr(train['event']))
            to_drop.add(hc if c1 >= c2 else col)
to_drop -= PROTECTED
candidate_features = [f for f in candidate_features if f not in to_drop]

# Limit to top 25 by correlation with event
if len(candidate_features) > 25:
    corrs = train_fe[candidate_features].corrwith(train['event']).abs()
    candidate_features = corrs.sort_values(ascending=False).head(25).index.tolist()

ALL_FEATURES = candidate_features
print(f"Final features: {len(ALL_FEATURES)}")

# Winsorize
for col in ALL_FEATURES:
    p1, p99 = train_fe[col].quantile(0.01), train_fe[col].quantile(0.99)
    train_fe[col] = train_fe[col].clip(p1, p99)
    test_fe[col] = test_fe[col].clip(p1, p99)

X_train_full = np.nan_to_num(train_fe[ALL_FEATURES].values.astype(np.float64))
X_test = np.nan_to_num(test_fe[ALL_FEATURES].values.astype(np.float64))

# ============================================================
# PHASE 2 — NATIVE SURVIVAL MODELS (the paradigm shift!)
# ============================================================
print("\n" + "=" * 70)
print("PHASE 2: Native Survival Models (scikit-survival)")
print("=" * 70)

# Create structured array for sksurv
# event=True means the fire DID hit (uncensored), event=False means censored
y_surv = np.array(
    [(bool(row['event'] == 1), row['time_to_hit_hours'])
     for _, row in train.iterrows()],
    dtype=[('event', bool), ('time', float)]
)

print(f"  Events: {y_surv['event'].sum()}, Censored: {(~y_surv['event']).sum()}")
print(f"  Time range: {y_surv['time'].min():.1f} - {y_surv['time'].max():.1f} hours")

# Train survival models with multi-seed CV
surv_oof = {h: {} for h in TIME_HORIZONS}
surv_test = {h: {} for h in TIME_HORIZONS}

SURV_MODELS = {
    'gbsa': lambda seed: GradientBoostingSurvivalAnalysis(
        n_estimators=200,
        max_depth=2,
        min_samples_leaf=15,
        min_samples_split=20,
        learning_rate=0.03,
        subsample=0.7,
        max_features=0.6,
        dropout_rate=0.1,
        random_state=seed,
    ),
    'rsf': lambda seed: RandomSurvivalForest(
        n_estimators=300,
        max_depth=3,
        min_samples_leaf=12,
        min_samples_split=18,
        max_features=0.5,
        random_state=seed,
        n_jobs=-1,
    ),
}

for model_name, model_fn in SURV_MODELS.items():
    print(f"\n  Training {model_name} (10 seeds × 5 folds)...")
    
    for h in TIME_HORIZONS:
        surv_oof[h][model_name] = np.zeros(len(train))
        surv_test[h][model_name] = np.zeros(len(X_test))
    
    for seed in SEEDS:
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        
        for tr_idx, val_idx in kf.split(X_train_full):
            mdl = model_fn(seed)
            mdl.fit(X_train_full[tr_idx], y_surv[tr_idx])
            
            # Get survival functions for validation and test
            val_surv_fns = mdl.predict_survival_function(X_train_full[val_idx])
            test_surv_fns = mdl.predict_survival_function(X_test)
            
            for h in TIME_HORIZONS:
                # P(T <= h) = 1 - S(h)
                for i, fn in enumerate(val_surv_fns):
                    try:
                        surv_oof[h][model_name][val_idx[i]] += (1.0 - fn(h)) / len(SEEDS)
                    except:
                        # If h is outside the function's domain, use boundary
                        if h <= fn.x.min():
                            surv_oof[h][model_name][val_idx[i]] += 0.0 / len(SEEDS)
                        else:
                            surv_oof[h][model_name][val_idx[i]] += 1.0 / len(SEEDS)
                
                for i, fn in enumerate(test_surv_fns):
                    try:
                        surv_test[h][model_name][i] += (1.0 - fn(h)) / (len(SEEDS) * N_SPLITS)
                    except:
                        if h <= fn.x.min():
                            surv_test[h][model_name][i] += 0.0 / (len(SEEDS) * N_SPLITS)
                        else:
                            surv_test[h][model_name][i] += 1.0 / (len(SEEDS) * N_SPLITS)
    
    # Print metrics
    for h in TIME_HORIZONS:
        y_binary = ((train['event'] == 1) & (train['time_to_hit_hours'] <= h)).astype(float).values
        p = np.clip(surv_oof[h][model_name], 0.001, 0.999)
        auc = roc_auc_score(y_binary, p)
        brier = brier_score_loss(y_binary, p)
        print(f"    {h}h: AUC={auc:.4f} Brier={brier:.5f}")

# ============================================================
# PHASE 3 — BINARY CLASSIFIERS (complementary to survival)
# ============================================================
print("\n" + "=" * 70)
print("PHASE 3: Binary Classifiers (CatBoost + LightGBM + ExtraTrees)")
print("=" * 70)

# IPCW weights
def compute_ipcw_weights(train_df, horizon, max_weight=3.0):
    n = len(train_df)
    labels = np.zeros(n, dtype=np.float64)
    weights = np.ones(n, dtype=np.float64)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        ev = train_df.iloc[i]['event']
        tth = train_df.iloc[i]['time_to_hit_hours']
        if ev == 1 and tth <= horizon:
            labels[i] = 1
        elif ev == 1 and tth > horizon:
            labels[i] = 0
        elif ev == 0:
            if tth >= horizon:
                labels[i] = 0
            elif tth >= horizon * 0.7:
                labels[i] = 0
                weights[i] = tth / horizon
            else:
                mask[i] = False
    if labels[mask].sum() > 0:
        neg_count = (labels[mask] == 0).sum()
        pos_count = int(labels[mask].sum())
        if neg_count < max(10, pos_count * 0.3):
            for relax in [0.5, 0.3]:
                for i in range(n):
                    if not mask[i] and train_df.iloc[i]['event'] == 0:
                        if train_df.iloc[i]['time_to_hit_hours'] >= horizon * relax:
                            mask[i] = True
                            labels[i] = 0
                            weights[i] = train_df.iloc[i]['time_to_hit_hours'] / horizon * 0.5
                if (labels[mask] == 0).sum() >= max(10, pos_count * 0.3):
                    break
    weights = np.clip(weights, 0.1, max_weight)
    if mask.sum() > 0 and weights[mask].mean() > 0:
        weights[mask] /= weights[mask].mean()
    return labels, weights, mask

ipcw_data = {}
for h in TIME_HORIZONS:
    labels, weights, mask = compute_ipcw_weights(train, h)
    ipcw_data[h] = (labels, weights, mask)

# Optuna tuning (50 trials — faster but still good)
N_TRIALS = 50

def run_optuna(X, y, w, model_type, horizon, n_trials=N_TRIALS):
    def objective(trial):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        metrics = []
        for tr_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            w_tr = w[tr_idx]

            if model_type == 'lgbm':
                params = {
                    'max_depth': trial.suggest_int('max_depth', 2, 4),
                    'num_leaves': trial.suggest_int('num_leaves', 4, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.06),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1.0, 30.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 2.0, 60.0, log=True),
                    'min_child_samples': trial.suggest_int('min_child_samples', 12, 40),
                    'subsample': trial.suggest_float('subsample', 0.5, 0.85),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.35, 0.75),
                    'min_split_gain': trial.suggest_float('min_split_gain', 0.01, 1.5),
                }
                mdl = lgb.LGBMClassifier(**params, n_estimators=500, objective='binary',
                                          verbosity=-1, random_state=42)
                mdl.fit(X_tr, y_tr, sample_weight=w_tr,
                       eval_set=[(X_val, y_val)],
                       callbacks=[lgb.early_stopping(50, verbose=False)])

            elif model_type == 'catboost':
                params = {
                    'depth': trial.suggest_int('depth', 2, 4),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.06),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 5.0, 60.0, log=True),
                    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 8, 30),
                    'subsample': trial.suggest_float('subsample', 0.5, 0.85),
                }
                mdl = CatBoostClassifier(**params, iterations=500, verbose=0,
                                          random_seed=42, early_stopping_rounds=50)
                mdl.fit(X_tr, y_tr, sample_weight=w_tr,
                       eval_set=(X_val, y_val), verbose=0)

            p = np.clip(mdl.predict_proba(X_val)[:, 1], 0.001, 0.999)

            if horizon == 12:
                if len(np.unique(y_val)) > 1:
                    metrics.append(-roc_auc_score(y_val, p))
                else:
                    metrics.append(0)
            else:
                metrics.append(brier_score_loss(y_val, p))

        return np.mean(metrics)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value

best_params = {}
for h in TIME_HORIZONS:
    y_h, w_h, mask = ipcw_data[h]
    X_f, y_f, w_f = X_train_full[mask], y_h[mask], w_h[mask]
    opt_type = "AUC" if h == 12 else "BRIER"
    print(f"\n  --- {h}h (optimizing {opt_type}) ---")

    for mt in ['lgbm', 'catboost']:
        bp, bv = run_optuna(X_f, y_f, w_f, mt, h)
        best_params[(mt, h)] = bp
        print(f"    {mt:10s} best {opt_type}: {abs(bv):.5f}")

# Train binary classifiers
BINARY_MODELS = ['lgbm', 'catboost', 'extratrees']
bin_oof = {h: {} for h in TIME_HORIZONS}
bin_test = {h: {} for h in TIME_HORIZONS}

print("\n  Training binary classifiers (10 seeds × 5 folds)...")

for h in TIME_HORIZONS:
    y_h, w_h, mask = ipcw_data[h]
    X_f, y_f, w_f = X_train_full[mask], y_h[mask], w_h[mask]
    
    for m in BINARY_MODELS:
        bin_oof[h][m] = np.zeros(len(y_f))
        bin_test[h][m] = np.zeros(len(X_test))
        
        for seed in SEEDS:
            skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
            for tr_idx, val_idx in skf.split(X_f, y_f):
                if m == 'lgbm':
                    bp = best_params[('lgbm', h)].copy()
                    mdl = lgb.LGBMClassifier(**bp, n_estimators=500, objective='binary',
                                              verbosity=-1, random_state=seed)
                    mdl.fit(X_f[tr_idx], y_f[tr_idx], sample_weight=w_f[tr_idx],
                           eval_set=[(X_f[val_idx], y_f[val_idx])],
                           callbacks=[lgb.early_stopping(50, verbose=False)])
                           
                elif m == 'catboost':
                    bp = best_params[('catboost', h)].copy()
                    mdl = CatBoostClassifier(**bp, iterations=500, verbose=0,
                                              random_seed=seed, early_stopping_rounds=50)
                    mdl.fit(X_f[tr_idx], y_f[tr_idx], sample_weight=w_f[tr_idx],
                           eval_set=(X_f[val_idx], y_f[val_idx]), verbose=0)
                           
                elif m == 'extratrees':
                    mdl = ExtraTreesClassifier(
                        n_estimators=500, max_depth=4, min_samples_leaf=12,
                        min_samples_split=15, max_features=0.45,
                        class_weight='balanced', random_state=seed, n_jobs=-1)
                    mdl.fit(X_f[tr_idx], y_f[tr_idx], sample_weight=w_f[tr_idx])
                
                bin_oof[h][m][val_idx] += mdl.predict_proba(X_f[val_idx])[:, 1] / len(SEEDS)
                bin_test[h][m] += mdl.predict_proba(X_test)[:, 1] / (len(SEEDS) * N_SPLITS)
    
    for m in BINARY_MODELS:
        p = np.clip(bin_oof[h][m], 0.001, 0.999)
        if len(np.unique(y_f)) > 1:
            auc = roc_auc_score(y_f, p)
            brier = brier_score_loss(y_f, p)
            print(f"    {h}h {m:12s}: AUC={auc:.4f} Brier={brier:.5f}")

# ============================================================
# PHASE 4 — CALIBRATION
# ============================================================
print("\n" + "=" * 70)
print("PHASE 4: Calibration")
print("=" * 70)

# Calibrate binary classifiers
cal_bin_oof = {h: {} for h in TIME_HORIZONS}
cal_bin_test = {h: {} for h in TIME_HORIZONS}

for h in TIME_HORIZONS:
    y_h, _, mask = ipcw_data[h]
    y_f = y_h[mask]
    
    for m in BINARY_MODELS:
        r_oof = bin_oof[h][m]
        r_test = bin_test[h][m]
        brier_raw = brier_score_loss(y_f, np.clip(r_oof, 0.001, 0.999))
        
        # CV-Isotonic
        p_iso = np.zeros(len(y_f))
        iso_tests = []
        for tr_i, val_i in StratifiedKFold(5, shuffle=True, random_state=42).split(
                r_oof.reshape(-1,1), y_f):
            iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
            iso.fit(r_oof[tr_i], y_f[tr_i])
            p_iso[val_i] = iso.predict(r_oof[val_i])
            iso_tests.append(iso.predict(r_test))
        iso_test = np.mean(iso_tests, axis=0)
        brier_iso = brier_score_loss(y_f, np.clip(p_iso, 0.001, 0.999))
        
        if brier_iso < brier_raw:
            cal_bin_oof[h][m] = p_iso
            cal_bin_test[h][m] = iso_test
            print(f"  {h}h {m:12s}: raw={brier_raw:.5f} iso={brier_iso:.5f} → cv-iso")
        else:
            cal_bin_oof[h][m] = r_oof
            cal_bin_test[h][m] = r_test
            print(f"  {h}h {m:12s}: raw={brier_raw:.5f} iso={brier_iso:.5f} → raw")

# Calibrate survival models (they predict on FULL train, not masked)
cal_surv_oof = {h: {} for h in TIME_HORIZONS}
cal_surv_test = {h: {} for h in TIME_HORIZONS}

for h in TIME_HORIZONS:
    y_binary_full = ((train['event'] == 1) & (train['time_to_hit_hours'] <= h)).astype(float).values
    
    for model_name in SURV_MODELS:
        r_oof = surv_oof[h][model_name]
        r_test = surv_test[h][model_name]
        brier_raw = brier_score_loss(y_binary_full, np.clip(r_oof, 0.001, 0.999))
        
        # CV-Isotonic on full data
        p_iso = np.zeros(len(y_binary_full))
        iso_tests = []
        for tr_i, val_i in StratifiedKFold(5, shuffle=True, random_state=42).split(
                r_oof.reshape(-1,1), y_binary_full):
            iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
            iso.fit(r_oof[tr_i], y_binary_full[tr_i])
            p_iso[val_i] = iso.predict(r_oof[val_i])
            iso_tests.append(iso.predict(r_test))
        iso_test = np.mean(iso_tests, axis=0)
        brier_iso = brier_score_loss(y_binary_full, np.clip(p_iso, 0.001, 0.999))
        
        if brier_iso < brier_raw:
            cal_surv_oof[h][model_name] = p_iso
            cal_surv_test[h][model_name] = iso_test
            print(f"  {h}h {model_name:12s}: raw={brier_raw:.5f} iso={brier_iso:.5f} → cv-iso")
        else:
            cal_surv_oof[h][model_name] = r_oof
            cal_surv_test[h][model_name] = r_test
            print(f"  {h}h {model_name:12s}: raw={brier_raw:.5f} iso={brier_iso:.5f} → raw")

# ============================================================
# PHASE 5 — OPTIMAL BLEND: Survival + Binary
# ============================================================
print("\n" + "=" * 70)
print("PHASE 5: Optimal Blend (Survival + Binary)")
print("=" * 70)

# For blending, we need common ground truth for all models
# Survival models use FULL train, binary models use MASKED train
# Solution: evaluate all on the FULL train binary labels

final_oof = {}
final_test = {}

for h in TIME_HORIZONS:
    y_binary_full = ((train['event'] == 1) & (train['time_to_hit_hours'] <= h)).astype(float).values
    y_h, _, mask = ipcw_data[h]
    
    # Collect all model predictions on test
    all_test_preds = {}
    all_test_names = []
    
    # Survival models (already on full train)
    for model_name in SURV_MODELS:
        all_test_preds[f'surv_{model_name}'] = cal_surv_test[h][model_name]
        all_test_names.append(f'surv_{model_name}')
    
    # Binary models
    for m in BINARY_MODELS:
        all_test_preds[f'bin_{m}'] = cal_bin_test[h][m]
        all_test_names.append(f'bin_{m}')
    
    # For Brier optimization, use survival model OOF on full data 
    # and binary model OOF on masked data (their respective ground truths)
    # But for final blend weight optimization, just optimize test-side using 
    # OOF Brier as proxy
    
    # Simple approach: compute each model's OOF Brier on their ground truth,
    # then weight inversely proportional to Brier
    model_scores = {}
    
    for model_name in SURV_MODELS:
        p = np.clip(cal_surv_oof[h][model_name], 0.001, 0.999)
        brier = brier_score_loss(y_binary_full, p)
        auc = roc_auc_score(y_binary_full, p)
        hybrid = 0.3 * auc + 0.7 * (1 - brier)
        model_scores[f'surv_{model_name}'] = hybrid
    
    for m in BINARY_MODELS:
        y_f = y_h[mask]
        p = np.clip(cal_bin_oof[h][m], 0.001, 0.999)
        brier = brier_score_loss(y_f, p)
        auc = roc_auc_score(y_f, p) if len(np.unique(y_f)) > 1 else 0.5
        hybrid = 0.3 * auc + 0.7 * (1 - brier)
        model_scores[f'bin_{m}'] = hybrid
    
    # Softmax weights
    scores = np.array([model_scores[n] for n in all_test_names])
    w = np.exp(8 * (scores - scores.max()))
    w = np.maximum(w / w.sum(), 0.03)  # Min 3% weight
    w /= w.sum()
    
    # Weighted average on test
    blend = np.zeros(len(X_test))
    for i, name in enumerate(all_test_names):
        blend += w[i] * np.clip(all_test_preds[name], 0.001, 0.999)
    
    final_test[h] = blend
    
    print(f"\n  {h}h weights:")
    for i, name in enumerate(all_test_names):
        print(f"    {name:20s}: w={w[i]:.3f} hybrid={model_scores[name]:.5f}")

# ============================================================
# PHASE 6 — POST-PROCESSING + BASE RATE ALIGNMENT
# ============================================================
print("\n" + "=" * 70)
print("PHASE 6: Post-Processing")
print("=" * 70)

# Base rate alignment
for h in TIME_HORIZONS:
    y_binary_full = ((train['event'] == 1) & (train['time_to_hit_hours'] <= h)).astype(float).values
    train_base = y_binary_full.mean()
    tp = np.clip(final_test[h], 1e-5, 1-1e-5)
    test_base = tp.mean()
    
    if abs(test_base - train_base) > 0.005:
        shift = logit(train_base) - logit(test_base)
        final_test[h] = expit(logit(tp) + 0.25 * shift)  # Gentle alignment
        print(f"  {h}h: train_base={train_base:.4f} test_base={test_base:.4f} → aligned to {final_test[h].mean():.4f}")
    else:
        print(f"  {h}h: train_base={train_base:.4f} test_base={test_base:.4f} → OK")

# Gentle rank blend (5%)
for h in TIME_HORIZONS:
    cal = final_test[h]
    rank = rankdata(cal) / len(cal)
    final_test[h] = 0.95 * cal + 0.05 * rank

# Enforce monotonicity
def enforce_monotonicity(preds_dict, min_inc=MONOTONICITY_INCREMENT):
    n = len(preds_dict[TIME_HORIZONS[0]])
    res = {h: preds_dict[h].copy() for h in TIME_HORIZONS}
    v = 0
    for i in range(n):
        for j in range(1, len(TIME_HORIZONS)):
            prev = TIME_HORIZONS[j-1]
            curr = TIME_HORIZONS[j]
            if res[curr][i] < res[prev][i] + min_inc:
                res[curr][i] = res[prev][i] + min_inc
                v += 1
    return res, v

# ============================================================
# PHASE 7 — GENERATE SUBMISSIONS
# ============================================================
print("\n" + "=" * 70)
print("PHASE 7: Generating Submissions")
print("=" * 70)

def create_submission(preds_dict, clip_lo, clip_hi, suffix=''):
    clipped = {h: np.clip(preds_dict[h], clip_lo, clip_hi) for h in TIME_HORIZONS}
    mono, v = enforce_monotonicity(clipped)
    sub = sample.copy()
    for h in TIME_HORIZONS:
        sub[f'prob_{h}h'] = mono[h]
    fname = f'd:/WiDS/submission{suffix}.csv'
    sub.to_csv(fname, index=False)
    assert sub.shape == (95, 5)
    assert sub.isnull().sum().sum() == 0
    for h in TIME_HORIZONS:
        assert sub[f'prob_{h}h'].between(0, 1).all()
    print(f"  {os.path.basename(fname)}: mono_fixes={v}, "
          f"means=[{sub['prob_12h'].mean():.4f}, {sub['prob_24h'].mean():.4f}, "
          f"{sub['prob_48h'].mean():.4f}, {sub['prob_72h'].mean():.4f}]")
    return sub

# Pure v11 submission
sub_v11 = create_submission(final_test, 0.008, 0.992, suffix='_v11')

# Blend with LB-proven submission_07 (0.957 on LB)
old = pd.read_csv('d:/WiDS/submission_07.csv')
cols = ['prob_12h', 'prob_24h', 'prob_48h', 'prob_72h']

# Also blend with submission_08 (0.960 on LB)
sub08 = pd.read_csv('d:/WiDS/submission_08.csv')

# Strategy A: 60% v11 + 40% best_LB (submission_08)
blend_A = sample.copy()
for c in cols:
    blend_A[c] = 0.60 * sub_v11[c] + 0.40 * sub08[c]
for c in cols:
    blend_A[c] = blend_A[c].clip(0.008, 0.992)
for i in range(len(blend_A)):
    for j in range(1, len(cols)):
        if blend_A.at[blend_A.index[i], cols[j]] < blend_A.at[blend_A.index[i], cols[j-1]] + MONOTONICITY_INCREMENT:
            blend_A.at[blend_A.index[i], cols[j]] = blend_A.at[blend_A.index[i], cols[j-1]] + MONOTONICITY_INCREMENT
blend_A.to_csv('d:/WiDS/submission_v11_blend60.csv', index=False)
print(f"  submission_v11_blend60.csv: 60% v11 + 40% LB0.960")

# Strategy B: 50% v11 + 30% sub08 + 20% sub07 (triple blend)
blend_B = sample.copy()
for c in cols:
    blend_B[c] = 0.50 * sub_v11[c] + 0.30 * sub08[c] + 0.20 * old[c]
for c in cols:
    blend_B[c] = blend_B[c].clip(0.008, 0.992)
for i in range(len(blend_B)):
    for j in range(1, len(cols)):
        if blend_B.at[blend_B.index[i], cols[j]] < blend_B.at[blend_B.index[i], cols[j-1]] + MONOTONICITY_INCREMENT:
            blend_B.at[blend_B.index[i], cols[j]] = blend_B.at[blend_B.index[i], cols[j-1]] + MONOTONICITY_INCREMENT
blend_B.to_csv('d:/WiDS/submission_v11_blend50.csv', index=False)
print(f"  submission_v11_blend50.csv: 50% v11 + 30% sub08 + 20% sub07")

# Strategy C: 40% v11 + 60% sub08 (conservative, trusts LB more)
blend_C = sample.copy()
for c in cols:
    blend_C[c] = 0.40 * sub_v11[c] + 0.60 * sub08[c]
for c in cols:
    blend_C[c] = blend_C[c].clip(0.008, 0.992)
for i in range(len(blend_C)):
    for j in range(1, len(cols)):
        if blend_C.at[blend_C.index[i], cols[j]] < blend_C.at[blend_C.index[i], cols[j-1]] + MONOTONICITY_INCREMENT:
            blend_C.at[blend_C.index[i], cols[j]] = blend_C.at[blend_C.index[i], cols[j-1]] + MONOTONICITY_INCREMENT
blend_C.to_csv('d:/WiDS/submission_v11_blend40.csv', index=False)
print(f"  submission_v11_blend40.csv: 40% v11 + 60% sub08 (conservative)")

# ============================================================
# FINAL: Create THE submission_09.csv (best candidate)
# ============================================================
# Use 50/30/20 triple blend as primary — maximum diversity
sub_09 = blend_B.copy()
sub_09.to_csv('d:/WiDS/submission_09.csv', index=False)
print(f"\n  >>> submission_09.csv CREATED (50% survival + 30% binary_v10 + 20% binary_v8)")

# ============================================================
# FINAL REPORT
# ============================================================
elapsed = time.time() - start_time
print("\n" + "=" * 70)
print(f"FINAL REPORT (elapsed: {elapsed/60:.1f} min)")
print("=" * 70)

for fname in ['submission_v11.csv', 'submission_v11_blend60.csv', 
              'submission_v11_blend50.csv', 'submission_v11_blend40.csv',
              'submission_09.csv']:
    df = pd.read_csv(f'd:/WiDS/{fname}')
    print(f"\n  {fname}:")
    for c in cols:
        print(f"    {c}: mean={df[c].mean():.4f} std={df[c].std():.4f}")

# Correlation between old and new
for fname in ['submission_v11.csv', 'submission_09.csv']:
    df = pd.read_csv(f'd:/WiDS/{fname}')
    print(f"\n  {fname} correlation with sub08:")
    for c in cols:
        print(f"    {c}: r={df[c].corr(sub08[c]):.5f}")

print("\n" + "=" * 70)
print("Pipeline v11 SURVIVAL COMPLETE. 🔥🔥🔥")
print("=" * 70)
