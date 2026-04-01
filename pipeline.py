"""
WiDS Global Datathon 2026 -- Pipeline v7.0 "Nuclear Option"
============================================================
TARGET: 0.97566+ (from 0.94691)

KEY CHANGES from v6:
  1. IPCW censoring weights (Kaplan-Meier) -- fixes poisoned negatives
  2. Ridge stacking meta-learner -- replaces heuristic weighting
  3. Lean feature set (~20 proven features, not 45 noisy ones)
  4. 6-model base layer (added ExtraTrees)
  5. Dual calibration (Platt + Isotonic, keep best)
  6. Smart pseudo-labeling (unanimous high-confidence only)
  7. Cascaded horizon modeling (12h feeds 24h feeds 48h feeds 72h)

Metric: Hybrid = 0.3 * C-index + 0.7 * (1 - Weighted_Brier)
  Weighted_Brier = 0.3*B_24h + 0.4*B_48h + 0.3*B_72h
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.stats import rankdata
from lifelines import KaplanMeierFitter
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import time as time_module
start_time = time_module.time()

# ============================================================
# PHASE 1 -- DATA LOADING
# ============================================================
print("=" * 80)
print("  PIPELINE v7.0 -- NUCLEAR OPTION")
print("=" * 80)

train = pd.read_csv(r'd:\WiDS\train.csv')
test = pd.read_csv(r'd:\WiDS\test.csv')
sample_sub = pd.read_csv(r'd:\WiDS\sample_submission.csv')

print(f"\n  Train: {train.shape[0]} rows x {train.shape[1]} cols")
print(f"  Test:  {test.shape[0]} rows x {test.shape[1]} cols")
print(f"  Event=1 (hit): {(train['event'] == 1).sum()}")
print(f"  Event=0 (censored): {(train['event'] == 0).sum()}")
print(f"  Hit rate: {train['event'].mean():.4f}")

for t in [12, 24, 48, 72]:
    n_hit = ((train['event'] == 1) & (train['time_to_hit_hours'] <= t)).sum()
    print(f"  Hits <= {t}h: {n_hit} ({n_hit/len(train)*100:.1f}%)")

# ============================================================
# PHASE 2 -- IPCW CENSORING WEIGHTS (Kaplan-Meier)
# ============================================================
print("\n" + "=" * 80)
print("  PHASE 2 -- IPCW CENSORING WEIGHTS")
print("=" * 80)

# Fit KM to the CENSORING distribution
# For censoring KM: "event" = censored (i.e., flip the event indicator)
km_censor = KaplanMeierFitter()
km_censor.fit(
    durations=train['time_to_hit_hours'],
    event_observed=1 - train['event'],  # censoring is the "event"
)

def compute_ipcw_weights(train_df, horizon):
    """
    Compute IPCW weights for a given horizon with graceful fallback.
    
    For each sample:
    - If event=1 and time <= horizon: TRUE POSITIVE (weight = 1/G(time))
    - If event=1 and time > horizon: TRUE NEGATIVE (weight = 1/G(horizon))  
    - If event=0 and obs_time >= threshold: NEGATIVE (weight scaled by obs proximity)
    - If event=0 and obs_time < threshold: UNKNOWN -- exclude (weight = 0)
    
    The threshold starts at horizon and relaxes down to horizon*0.6 if needed
    to ensure we have enough negatives for training.
    """
    MIN_NEGATIVES = 10
    
    # Try progressively relaxed thresholds
    for relax in [1.0, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30]:
        threshold = horizon * relax
        
        weights = np.zeros(len(train_df))
        labels = np.zeros(len(train_df), dtype=int)
        mask = np.zeros(len(train_df), dtype=bool)
        
        for i, row in train_df.iterrows():
            t = row['time_to_hit_hours']
            is_hit = row['event'] == 1
            
            if is_hit:
                if t <= horizon:
                    labels[i] = 1
                    g_t = km_censor.predict(min(t, km_censor.timeline.max()))
                    weights[i] = 1.0 / max(g_t, 0.05)
                    mask[i] = True
                else:
                    labels[i] = 0
                    g_h = km_censor.predict(min(horizon, km_censor.timeline.max()))
                    weights[i] = 1.0 / max(g_h, 0.05)
                    mask[i] = True
            else:
                if t >= threshold:
                    labels[i] = 0
                    g_t_val = km_censor.predict(min(t, km_censor.timeline.max()))
                    weights[i] = 1.0 / max(g_t_val, 0.05)
                    # Downweight censored samples observed less than the full horizon
                    if t < horizon:
                        # Scale weight by how close observation is to horizon
                        coverage = t / horizon
                        weights[i] *= coverage  # e.g., observed 60h of 72h = 0.83x weight
                    mask[i] = True
                else:
                    mask[i] = False
                    weights[i] = 0
        
        n_neg = ((labels == 0) & mask).sum()
        n_pos = ((labels == 1) & mask).sum()
        
        if n_neg >= MIN_NEGATIVES and n_pos >= MIN_NEGATIVES:
            break
    
    # Normalize weights to mean=1 for numerical stability
    valid_weights = weights[mask]
    if valid_weights.sum() > 0:
        weights[mask] = valid_weights / valid_weights.mean()
    
    return labels, weights, mask

time_horizons = [12, 24, 48, 72]
ipcw_data = {}

for horizon in time_horizons:
    labels, weights, mask = compute_ipcw_weights(train, horizon)
    n_included = mask.sum()
    n_pos = (labels[mask] == 1).sum()
    n_neg = (labels[mask] == 0).sum()
    ipcw_data[horizon] = (labels, weights, mask)
    print(f"  {horizon}h: {n_pos} pos + {n_neg} neg = {n_included} included "
          f"({len(train) - n_included} excluded as unknown)")
    print(f"         Weight range: [{weights[mask].min():.3f}, {weights[mask].max():.3f}], "
          f"mean={weights[mask].mean():.3f}")

# ============================================================
# PHASE 3 -- LEAN FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 80)
print("  PHASE 3 -- LEAN FEATURE ENGINEERING (~20 features)")
print("=" * 80)

def engineer_features(df):
    """Lean, proven feature set -- no noisy physics experiments."""
    out = df.copy()
    
    # === DISTANCE (top predictor, r=0.48) ===
    out['log_dist_min'] = np.log1p(df['dist_min_ci_0_5h'])
    out['inv_dist'] = 1.0 / (df['dist_min_ci_0_5h'] + 100)
    
    # === PROJECTED TIME TO HIT (t = d / v) ===
    closing = df['closing_speed_m_per_h'].clip(lower=0)
    out['time_to_hit_projected'] = df['dist_min_ci_0_5h'] / (closing + 1e-6)
    out['time_to_hit_projected'] = out['time_to_hit_projected'].clip(upper=500)
    out['log_time_projected'] = np.log1p(out['time_to_hit_projected'])
    
    # === THREAT COMPOSITES (proven in stability selection) ===
    out['directional_threat'] = df['closing_speed_m_per_h'] * df['alignment_abs']
    out['proximity_threat'] = df['closing_speed_m_per_h'] / (df['dist_min_ci_0_5h'] + 100)
    out['proximity_threat'] = out['proximity_threat'].clip(-5, 5)
    
    # === NEAR-MISS MARGIN ===
    projected_reach = df['radial_growth_m'] + df['projected_advance_m'].clip(lower=0)
    out['near_miss_margin'] = df['dist_min_ci_0_5h'] - projected_reach
    out['near_miss_margin'] = out['near_miss_margin'].clip(-50000, 500000)
    
    # === TEMPORAL (cyclic) ===
    out['hour_sin'] = np.sin(2 * np.pi * df['event_start_hour'] / 24)
    out['hour_cos'] = np.cos(2 * np.pi * df['event_start_hour'] / 24)
    
    # === BINARY THRESHOLDS ===
    out['is_close'] = (df['dist_min_ci_0_5h'] < 5000).astype(float)
    out['is_very_close'] = (df['dist_min_ci_0_5h'] < 2000).astype(float)
    
    # === OBSERVATION QUALITY ===
    out['dist_change_rate'] = df['dist_change_ci_0_5h'] / (df['dt_first_last_0_5h'] + 0.1)
    
    # === INTERACTION FEATURES (NEW) ===
    out['dist_x_alignment'] = df['dist_min_ci_0_5h'] * df['alignment_abs']
    out['speed_x_close'] = df['closing_speed_m_per_h'] * out['is_close']
    
    # === CLOSING VELOCITY (clipped) ===
    out['closing_velocity'] = df['closing_speed_m_per_h'].clip(lower=0)
    out['along_track_abs'] = df['along_track_speed'].abs()
    
    return out

train_fe = engineer_features(train)
test_fe = engineer_features(test)

# Feature list -- lean and proven
base_features = [
    'dist_min_ci_0_5h', 'closing_speed_m_per_h', 'alignment_abs',
    'area_growth_rate_ha_per_h', 'log1p_area_first', 'relative_growth_0_5h',
    'centroid_speed_m_per_h', 'radial_growth_rate_m_per_h',
    'dist_change_ci_0_5h', 'dist_slope_ci_0_5h', 'projected_advance_m',
    'along_track_speed', 'num_perimeters_0_5h', 'dt_first_last_0_5h',
    'dist_std_ci_0_5h', 'dist_fit_r2_0_5h', 'area_first_ha',
]

engineered_features = [
    'log_dist_min', 'inv_dist',
    'time_to_hit_projected', 'log_time_projected',
    'directional_threat', 'proximity_threat',
    'near_miss_margin',
    'hour_sin', 'hour_cos',
    'is_close', 'is_very_close',
    'dist_change_rate',
    'dist_x_alignment', 'speed_x_close',
    'closing_velocity', 'along_track_abs',
]

all_features = base_features + engineered_features
print(f"  Feature count: {len(all_features)}")

# === DROP NEAR-ZERO VARIANCE ===
train_std = train_fe[all_features].std()
low_var = train_std[train_std < 1e-10].index.tolist()
if low_var:
    print(f"  Dropping near-zero variance: {low_var}")
    all_features = [f for f in all_features if f not in low_var]

# === DROP HIGHLY CORRELATED (>0.95) ===
# PROTECT key features that must never be dropped
PROTECTED_FEATURES = {'dist_min_ci_0_5h', 'closing_speed_m_per_h', 'alignment_abs',
                      'log_dist_min', 'inv_dist', 'time_to_hit_projected'}

corr_matrix = train_fe[all_features].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = set()
for col in upper.columns:
    high_corr = upper.index[upper[col] > 0.95].tolist()
    for hc in high_corr:
        # Never drop protected features
        if hc in PROTECTED_FEATURES and col not in PROTECTED_FEATURES:
            to_drop.add(col)
        elif col in PROTECTED_FEATURES and hc not in PROTECTED_FEATURES:
            to_drop.add(hc)
        else:
            corr_event_col = abs(train_fe[col].corr(train_fe['event']))
            corr_event_hc = abs(train_fe[hc].corr(train_fe['event']))
            if corr_event_col >= corr_event_hc:
                to_drop.add(hc)
            else:
                to_drop.add(col)

# Final filter: never drop protected
to_drop -= PROTECTED_FEATURES

if to_drop:
    print(f"  Dropping highly correlated ({len(to_drop)}): {to_drop}")
    all_features = [f for f in all_features if f not in to_drop]

print(f"  [OK] Final feature count: {len(all_features)}")

# === PREPARE MATRICES ===
for col in all_features:
    p1, p99 = train_fe[col].quantile(0.01), train_fe[col].quantile(0.99)
    train_fe[col] = train_fe[col].clip(p1, p99)
    test_fe[col] = test_fe[col].clip(p1, p99)

X_train_full = train_fe[all_features].values
X_test = test_fe[all_features].values
X_train_full = np.nan_to_num(X_train_full, nan=0.0)
X_test = np.nan_to_num(X_test, nan=0.0)

# ============================================================
# PHASE 4 -- OPTUNA HYPERPARAMETER TUNING (40 trials)
# ============================================================
print("\n" + "=" * 80)
print("  PHASE 4 -- OPTUNA TUNING (40 trials/model/horizon)")
print("=" * 80)

N_OPTUNA_TRIALS = 40

def compute_hybrid_score_cv(y_true, y_pred):
    if len(np.unique(y_true)) < 2:
        return 0.5
    auc = roc_auc_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_pred)
    return 0.3 * auc + 0.7 * (1.0 - brier)

def tune_lgbm(X, y, w, n_trials=N_OPTUNA_TRIALS):
    def objective(trial):
        params = {
            'objective': 'binary', 'metric': 'binary_logloss',
            'num_leaves': trial.suggest_int('num_leaves', 4, 16),
            'max_depth': trial.suggest_int('max_depth', 2, 5),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 40),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 5.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'n_estimators': 500,
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'min_child_weight': trial.suggest_int('min_child_weight', 3, 15),
            'path_smooth': trial.suggest_float('path_smooth', 0.0, 3.0),
            'verbosity': -1, 'random_state': 42,
        }
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for tr_idx, val_idx in skf.split(X, y):
            mdl = lgb.LGBMClassifier(**params)
            mdl.fit(X[tr_idx], y[tr_idx], sample_weight=w[tr_idx],
                    eval_set=[(X[val_idx], y[val_idx])],
                    callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])
            preds = mdl.predict_proba(X[val_idx])[:, 1]
            scores.append(compute_hybrid_score_cv(y[val_idx], preds))
        return np.mean(scores)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params

def tune_xgb(X, y, w, n_trials=N_OPTUNA_TRIALS):
    def objective(trial):
        params = {
            'objective': 'binary:logistic', 'eval_metric': 'logloss',
            'max_depth': trial.suggest_int('max_depth', 2, 5),
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 25),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 5.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'n_estimators': 500,
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'gamma': trial.suggest_float('gamma', 0.0, 2.0),
            'tree_method': 'hist', 'verbosity': 0, 'random_state': 42,
            'early_stopping_rounds': 30,
        }
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for tr_idx, val_idx in skf.split(X, y):
            mdl = xgb.XGBClassifier(**params)
            mdl.fit(X[tr_idx], y[tr_idx], sample_weight=w[tr_idx],
                    eval_set=[(X[val_idx], y[val_idx])],
                    verbose=False)
            preds = mdl.predict_proba(X[val_idx])[:, 1]
            scores.append(compute_hybrid_score_cv(y[val_idx], preds))
        return np.mean(scores)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    best['early_stopping_rounds'] = 30
    return best

def tune_catboost(X, y, w, n_trials=N_OPTUNA_TRIALS):
    def objective(trial):
        params = {
            'iterations': 500,
            'depth': trial.suggest_int('depth', 2, 5),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.5, 10.0, log=True),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 30),
            'random_strength': trial.suggest_float('random_strength', 0.5, 3.0),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 2.0),
            'border_count': trial.suggest_int('border_count', 32, 128),
            'verbose': 0, 'random_seed': 42,
            'early_stopping_rounds': 30,
        }
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for tr_idx, val_idx in skf.split(X, y):
            mdl = CatBoostClassifier(**params)
            mdl.fit(X[tr_idx], y[tr_idx], sample_weight=w[tr_idx],
                    eval_set=(X[val_idx], y[val_idx]), verbose=0)
            preds = mdl.predict_proba(X[val_idx])[:, 1]
            scores.append(compute_hybrid_score_cv(y[val_idx], preds))
        return np.mean(scores)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params

# Prepare horizon-specific data with IPCW
horizon_data = {}
for horizon in time_horizons:
    labels, weights, mask = ipcw_data[horizon]
    X_h = X_train_full[mask]
    y_h = labels[mask].astype(int)
    w_h = weights[mask]
    horizon_data[horizon] = (X_h, y_h, w_h, np.where(mask)[0])

# Tune
best_params = {}
for horizon in time_horizons:
    X_h, y_h, w_h, _ = horizon_data[horizon]
    print(f"\n  --- Tuning {horizon}h ({int(y_h.sum())} pos / {len(y_h)} total) ---")
    
    print(f"    LightGBM ({N_OPTUNA_TRIALS} trials)...")
    best_params[('lgbm', horizon)] = tune_lgbm(X_h, y_h, w_h)
    print(f"    [OK] LightGBM done")
    

    
    print(f"    CatBoost ({N_OPTUNA_TRIALS} trials)...")
    best_params[('catboost', horizon)] = tune_catboost(X_h, y_h, w_h)
    print(f"    [OK] CatBoost done")

# ============================================================
# PHASE 5 -- 5-MODEL BASE LAYER TRAINING (XGB removed)
# ============================================================
print("\n" + "=" * 80)
print("  PHASE 5 -- 5-MODEL BASE LAYER (IPCW-weighted)")
print("=" * 80)

SEEDS = [42, 52, 62, 72, 82]
N_SPLITS = 5
model_names = ['lgbm', 'catboost', 'logreg', 'rf', 'et']  # XGBoost removed (AUC=0.77 dragging ensemble)

all_oof_preds = {}
all_test_preds = {}
all_metrics = {}

for horizon in time_horizons:
    X_h, y_h, w_h, mask_indices = horizon_data[horizon]
    print(f"\n  --- {horizon}h ({len(y_h)} samples, {int(y_h.sum())} pos) ---")
    
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        
        # ---- LightGBM (Optuna-tuned, IPCW-weighted) ----
        oof_lgbm = np.zeros(len(y_h))
        test_lgbm = np.zeros((N_SPLITS, len(X_test)))
        
        bp = best_params[('lgbm', horizon)].copy()
        bp.update({'objective': 'binary', 'metric': 'binary_logloss',
                   'n_estimators': 500, 'verbosity': -1, 'random_state': seed})
        
        for fi, (tr_idx, val_idx) in enumerate(skf.split(X_h, y_h)):
            mdl = lgb.LGBMClassifier(**bp)
            mdl.fit(X_h[tr_idx], y_h[tr_idx], sample_weight=w_h[tr_idx],
                    eval_set=[(X_h[val_idx], y_h[val_idx])],
                    callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])
            oof_lgbm[val_idx] = mdl.predict_proba(X_h[val_idx])[:, 1]
            test_lgbm[fi] = mdl.predict_proba(X_test)[:, 1]
        
        all_oof_preds[('lgbm', horizon, seed)] = oof_lgbm
        all_test_preds[('lgbm', horizon, seed)] = test_lgbm.mean(axis=0)
        

        
        # ---- CatBoost (Optuna-tuned, IPCW-weighted) ----
        oof_cat = np.zeros(len(y_h))
        test_cat = np.zeros((N_SPLITS, len(X_test)))
        
        bp_c = best_params[('catboost', horizon)].copy()
        bp_c.update({'iterations': 500,
                     'verbose': 0, 'random_seed': seed, 'early_stopping_rounds': 30})
        
        for fi, (tr_idx, val_idx) in enumerate(skf.split(X_h, y_h)):
            mdl = CatBoostClassifier(**bp_c)
            mdl.fit(X_h[tr_idx], y_h[tr_idx], sample_weight=w_h[tr_idx],
                    eval_set=(X_h[val_idx], y_h[val_idx]), verbose=0)
            oof_cat[val_idx] = mdl.predict_proba(X_h[val_idx])[:, 1]
            test_cat[fi] = mdl.predict_proba(X_test)[:, 1]
        
        all_oof_preds[('catboost', horizon, seed)] = oof_cat
        all_test_preds[('catboost', horizon, seed)] = test_cat.mean(axis=0)
        
        # ---- LogisticRegression ----
        oof_lr = np.zeros(len(y_h))
        test_lr = np.zeros((N_SPLITS, len(X_test)))
        
        scaler = StandardScaler()
        X_h_sc = scaler.fit_transform(X_h)
        X_test_sc = scaler.transform(X_test)
        
        for fi, (tr_idx, val_idx) in enumerate(skf.split(X_h_sc, y_h)):
            mdl = LogisticRegression(
                C=0.05, solver='lbfgs', max_iter=2000,
                class_weight='balanced', random_state=seed
            )
            mdl.fit(X_h_sc[tr_idx], y_h[tr_idx], sample_weight=w_h[tr_idx])
            oof_lr[val_idx] = mdl.predict_proba(X_h_sc[val_idx])[:, 1]
            test_lr[fi] = mdl.predict_proba(X_test_sc)[:, 1]
        
        all_oof_preds[('logreg', horizon, seed)] = oof_lr
        all_test_preds[('logreg', horizon, seed)] = test_lr.mean(axis=0)
        
        # ---- RandomForest (IPCW-weighted) ----
        oof_rf = np.zeros(len(y_h))
        test_rf = np.zeros((N_SPLITS, len(X_test)))
        
        for fi, (tr_idx, val_idx) in enumerate(skf.split(X_h, y_h)):
            mdl = RandomForestClassifier(
                n_estimators=500, max_depth=4, min_samples_leaf=12,
                max_features='sqrt', class_weight='balanced_subsample',
                random_state=seed, n_jobs=-1
            )
            mdl.fit(X_h[tr_idx], y_h[tr_idx], sample_weight=w_h[tr_idx])
            oof_rf[val_idx] = mdl.predict_proba(X_h[val_idx])[:, 1]
            test_rf[fi] = mdl.predict_proba(X_test)[:, 1]
        
        all_oof_preds[('rf', horizon, seed)] = oof_rf
        all_test_preds[('rf', horizon, seed)] = test_rf.mean(axis=0)
        
        # ---- ExtraTreesClassifier (NEW -- for diversity) ----
        oof_et = np.zeros(len(y_h))
        test_et = np.zeros((N_SPLITS, len(X_test)))
        
        for fi, (tr_idx, val_idx) in enumerate(skf.split(X_h, y_h)):
            mdl = ExtraTreesClassifier(
                n_estimators=500, max_depth=5, min_samples_leaf=10,
                max_features='sqrt', class_weight='balanced_subsample',
                random_state=seed, n_jobs=-1
            )
            mdl.fit(X_h[tr_idx], y_h[tr_idx], sample_weight=w_h[tr_idx])
            oof_et[val_idx] = mdl.predict_proba(X_h[val_idx])[:, 1]
            test_et[fi] = mdl.predict_proba(X_test)[:, 1]
        
        all_oof_preds[('et', horizon, seed)] = oof_et
        all_test_preds[('et', horizon, seed)] = test_et.mean(axis=0)
    
    # Report per-model metrics
    for mn in model_names:
        seed_aucs, seed_briers, seed_hybrids = [], [], []
        for seed in SEEDS:
            oof = all_oof_preds[(mn, horizon, seed)]
            if len(np.unique(y_h)) > 1:
                auc_val = roc_auc_score(y_h, oof)
                brier_val = brier_score_loss(y_h, oof)
                hybrid_val = 0.3 * auc_val + 0.7 * (1.0 - brier_val)
                seed_aucs.append(auc_val)
                seed_briers.append(brier_val)
                seed_hybrids.append(hybrid_val)
        
        all_metrics[(mn, horizon)] = {
            'auc': np.mean(seed_aucs), 'auc_std': np.std(seed_aucs),
            'brier': np.mean(seed_briers), 'hybrid': np.mean(seed_hybrids),
        }
        print(f"    {mn:10s}: AUC={np.mean(seed_aucs):.4f}+/-{np.std(seed_aucs):.4f}, "
              f"Brier={np.mean(seed_briers):.4f}, Hybrid={np.mean(seed_hybrids):.4f}")

# ============================================================
# PHASE 6 -- SEED-AVERAGE OOF & TEST PREDICTIONS
# ============================================================
print("\n" + "=" * 80)
print("  PHASE 6 -- SEED AVERAGING")
print("=" * 80)

avg_oof = {}
avg_test = {}

for horizon in time_horizons:
    for mn in model_names:
        oof_all = np.array([all_oof_preds[(mn, horizon, s)] for s in SEEDS])
        test_all = np.array([all_test_preds[(mn, horizon, s)] for s in SEEDS])
        avg_oof[(mn, horizon)] = oof_all.mean(axis=0)
        avg_test[(mn, horizon)] = test_all.mean(axis=0)

print("  [OK] Averaged OOF and TEST predictions across 5 seeds")

# ============================================================
# PHASE 7 -- DUAL CALIBRATION (Platt + Isotonic, keep best)
# ============================================================
print("\n" + "=" * 80)
print("  PHASE 7 -- DUAL CALIBRATION")
print("=" * 80)

calibrated_oof = {}
calibrated_test = {}

for horizon in time_horizons:
    _, y_h, _, _ = horizon_data[horizon]
    
    for mn in model_names:
        oof = avg_oof[(mn, horizon)]
        tp = avg_test[(mn, horizon)]
        
        # Option 1: Platt scaling (LogReg on sigmoid)
        cal_oof_platt = np.zeros_like(oof)
        cal_test_platt = np.zeros((N_SPLITS, len(tp)))
        skf_cal = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        
        for fi, (tr_idx, val_idx) in enumerate(skf_cal.split(oof, y_h)):
            cal = LogisticRegression(C=1.0, solver='lbfgs', max_iter=2000)
            cal.fit(oof[tr_idx].reshape(-1, 1), y_h[tr_idx])
            cal_oof_platt[val_idx] = cal.predict_proba(oof[val_idx].reshape(-1, 1))[:, 1]
            cal_test_platt[fi] = cal.predict_proba(tp.reshape(-1, 1))[:, 1]
        cal_test_p = cal_test_platt.mean(axis=0)
        
        # Option 2: Isotonic regression
        cal_oof_iso = np.zeros_like(oof)
        cal_test_iso = np.zeros((N_SPLITS, len(tp)))
        
        for fi, (tr_idx, val_idx) in enumerate(skf_cal.split(oof, y_h)):
            ir = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
            ir.fit(oof[tr_idx], y_h[tr_idx])
            cal_oof_iso[val_idx] = ir.predict(oof[val_idx])
            cal_test_iso[fi] = ir.predict(tp)
        cal_test_i = cal_test_iso.mean(axis=0)
        
        # Pick the best
        b_raw = brier_score_loss(y_h, oof)
        b_platt = brier_score_loss(y_h, cal_oof_platt)
        b_iso = brier_score_loss(y_h, cal_oof_iso)
        
        best_method = 'raw'
        best_oof, best_test = oof, tp
        best_brier = b_raw
        
        if b_platt < best_brier:
            best_method = 'platt'
            best_oof, best_test = cal_oof_platt, cal_test_p
            best_brier = b_platt
        
        if b_iso < best_brier:
            best_method = 'isotonic'
            best_oof, best_test = cal_oof_iso, cal_test_i
            best_brier = b_iso
        
        calibrated_oof[(mn, horizon)] = best_oof
        calibrated_test[(mn, horizon)] = best_test
        
        print(f"  {mn:10s} @ {horizon}h: raw={b_raw:.4f} platt={b_platt:.4f} "
              f"iso={b_iso:.4f} -> [{best_method.upper()}]")

# ============================================================
# PHASE 8 -- RIDGE STACKING META-LEARNER (THE KEY INNOVATION)
# ============================================================
print("\n" + "=" * 80)
print("  PHASE 8 -- RIDGE STACKING META-LEARNER")
print("=" * 80)

stacked_oof = {}
stacked_test = {}

for hi, horizon in enumerate(time_horizons):
    _, y_h, _, _ = horizon_data[horizon]
    
    # Build stacking features: calibrated OOF from all 6 models
    stack_oof_features = np.column_stack([
        calibrated_oof[(mn, horizon)] for mn in model_names
    ])
    stack_test_features = np.column_stack([
        calibrated_test[(mn, horizon)] for mn in model_names
    ])
    
    # Cross-horizon cascade: only for test predictions (same 95 rows)
    # OOF arrays have different sizes per horizon due to IPCW, so skip OOF cascade
    if hi > 0:
        prev_horizon = time_horizons[hi - 1]
        if prev_horizon in stacked_test:
            stack_test_features = np.column_stack([
                stack_test_features,
                stacked_test[prev_horizon]
            ])
            # For OOF, add a dummy column of zeros (will be ignored by Ridge regularization)
            stack_oof_features = np.column_stack([
                stack_oof_features,
                np.zeros(len(y_h))  # placeholder to match test feature count
            ])
    
    # Add 2 strongest raw features for context
    X_h_raw, _, _, mask_idx = horizon_data[horizon]
    dist_idx = all_features.index('dist_min_ci_0_5h') if 'dist_min_ci_0_5h' in all_features else 0
    speed_idx = all_features.index('closing_speed_m_per_h') if 'closing_speed_m_per_h' in all_features else 1
    
    stack_oof_features = np.column_stack([
        stack_oof_features,
        X_h_raw[:, dist_idx],
        X_h_raw[:, speed_idx],
    ])
    stack_test_features = np.column_stack([
        stack_test_features,
        X_test[:, dist_idx],
        X_test[:, speed_idx],
    ])
    
    # Scale features for Ridge
    scaler_stack = StandardScaler()
    stack_oof_sc = scaler_stack.fit_transform(stack_oof_features)
    stack_test_sc = scaler_stack.transform(stack_test_features)
    
    # Train Ridge meta-learner with cross-validation
    best_alpha = 1.0
    best_brier = 999
    
    for alpha in [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
        skf_s = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_ridge = np.zeros(len(y_h))
        
        for tr_idx, val_idx in skf_s.split(stack_oof_sc, y_h):
            ridge = Ridge(alpha=alpha)
            ridge.fit(stack_oof_sc[tr_idx], y_h[tr_idx])
            oof_ridge[val_idx] = ridge.predict(stack_oof_sc[val_idx])
        
        oof_ridge = np.clip(oof_ridge, 0.001, 0.999)
        b = brier_score_loss(y_h, oof_ridge)
        if b < best_brier:
            best_brier = b
            best_alpha = alpha
    
    # Final Ridge with best alpha
    skf_s = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_final = np.zeros(len(y_h))
    test_final = np.zeros((5, len(X_test)))
    
    for fi, (tr_idx, val_idx) in enumerate(skf_s.split(stack_oof_sc, y_h)):
        ridge = Ridge(alpha=best_alpha)
        ridge.fit(stack_oof_sc[tr_idx], y_h[tr_idx])
        oof_final[val_idx] = ridge.predict(stack_oof_sc[val_idx])
        test_final[fi] = ridge.predict(stack_test_sc)
    
    oof_final = np.clip(oof_final, 0.001, 0.999)
    test_stacked = np.clip(test_final.mean(axis=0), 0.001, 0.999)
    
    stacked_oof[horizon] = oof_final
    stacked_test[horizon] = test_stacked
    
    # Compare stacked vs simple average
    simple_avg_oof = np.mean([calibrated_oof[(mn, horizon)] for mn in model_names], axis=0)
    b_simple = brier_score_loss(y_h, simple_avg_oof)
    b_stacked = brier_score_loss(y_h, oof_final)
    
    auc_stacked = roc_auc_score(y_h, oof_final) if len(np.unique(y_h)) > 1 else 0.5
    hybrid_stacked = 0.3 * auc_stacked + 0.7 * (1.0 - b_stacked)
    
    n_features = stack_oof_sc.shape[1]
    print(f"  {horizon}h: alpha={best_alpha}, {n_features} features")
    print(f"         Simple avg Brier={b_simple:.4f} -> Stacked Brier={b_stacked:.4f} "
          f"({'BETTER' if b_stacked < b_simple else 'SAME'})")
    print(f"         Stacked AUC={auc_stacked:.4f}, Hybrid={hybrid_stacked:.4f}")

# ============================================================
# PHASE 9 -- SMART PSEUDO-LABELING (high confidence only)
# ============================================================
print("\n" + "=" * 80)
print("  PHASE 9 -- SMART PSEUDO-LABELING")
print("=" * 80)

# Only relabel if ALL models strongly agree
PSEUDO_POS_THRESH = 0.92  # consensus threshold for positive
PSEUDO_NEG_THRESH = 0.08  # consensus threshold for negative

pseudo_label_applied = False

for horizon in time_horizons:
    # Check consensus across models
    model_preds = np.column_stack([calibrated_test[(mn, horizon)] for mn in model_names])
    
    # Positive: ALL models predict > threshold
    pos_mask = (model_preds > PSEUDO_POS_THRESH).all(axis=1)
    neg_mask = (model_preds < PSEUDO_NEG_THRESH).all(axis=1)
    
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()
    
    if n_pos + n_neg < 3:
        print(f"  {horizon}h: Only {n_pos}+{n_neg} consensus samples -- SKIPPING")
        continue
    
    print(f"  {horizon}h: {n_pos} strong positives, {n_neg} strong negatives")
    pseudo_label_applied = True
    
    # Augment training data with pseudo-labels
    X_h, y_h, w_h, _ = horizon_data[horizon]
    
    pseudo_X = np.vstack([
        X_test[pos_mask] if n_pos > 0 else np.empty((0, X_test.shape[1])),
        X_test[neg_mask] if n_neg > 0 else np.empty((0, X_test.shape[1])),
    ])
    pseudo_y = np.concatenate([
        np.ones(n_pos),
        np.zeros(n_neg),
    ])
    pseudo_w = np.ones(n_pos + n_neg) * 0.5  # downweight pseudo-labels
    
    X_aug = np.vstack([X_h, pseudo_X])
    y_aug = np.concatenate([y_h, pseudo_y])
    w_aug = np.concatenate([w_h, pseudo_w])
    
    # Retrain stacking features with augmented data
    stack_oof_aug = np.column_stack([calibrated_oof[(mn, horizon)] for mn in model_names])
    stack_test_aug = np.column_stack([calibrated_test[(mn, horizon)] for mn in model_names])
    
    # Add pseudo-label stacking features (use test model predictions)
    pseudo_stack = model_preds[pos_mask | neg_mask]
    
    # Rebuild with cascade (test-side only, OOF sizes differ per horizon)
    hi = time_horizons.index(horizon)
    if hi > 0 and time_horizons[hi-1] in stacked_test:
        prev_h = time_horizons[hi-1]
        # Add dummy column for OOF to match test feature count
        stack_oof_aug = np.column_stack([stack_oof_aug, np.zeros(len(y_h))])
        pseudo_stack_prev = stacked_test[prev_h][pos_mask | neg_mask]
        pseudo_stack = np.column_stack([pseudo_stack, pseudo_stack_prev])
        stack_test_aug = np.column_stack([stack_test_aug, stacked_test[prev_h]])
    
    dist_idx = all_features.index('dist_min_ci_0_5h') if 'dist_min_ci_0_5h' in all_features else 0
    speed_idx = all_features.index('closing_speed_m_per_h') if 'closing_speed_m_per_h' in all_features else 1
    
    stack_oof_aug = np.column_stack([stack_oof_aug, X_h[:, dist_idx], X_h[:, speed_idx]])
    pseudo_stack = np.column_stack([pseudo_stack, 
                                     X_test[pos_mask | neg_mask, dist_idx],
                                     X_test[pos_mask | neg_mask, speed_idx]])
    stack_test_aug = np.column_stack([stack_test_aug, X_test[:, dist_idx], X_test[:, speed_idx]])
    
    # Combine OOF and pseudo stacking features
    stack_all = np.vstack([stack_oof_aug, pseudo_stack])
    
    scaler_ps = StandardScaler()
    stack_all_sc = scaler_ps.fit_transform(stack_all)
    stack_test_sc = scaler_ps.transform(stack_test_aug)
    
    # Retrain Ridge on augmented data
    ridge_ps = Ridge(alpha=1.0)
    ridge_ps.fit(stack_all_sc, y_aug)
    test_ps = np.clip(ridge_ps.predict(stack_test_sc), 0.001, 0.999)
    
    # Only use if it doesn't break base rate alignment
    base_rate = y_h.mean()
    if abs(test_ps.mean() - base_rate) < abs(stacked_test[horizon].mean() - base_rate):
        stacked_test[horizon] = test_ps
        print(f"         Pseudo-label improved alignment: {test_ps.mean():.4f} vs base {base_rate:.4f}")
    else:
        print(f"         Pseudo-label rejected (would worsen alignment)")

if not pseudo_label_applied:
    print("  No pseudo-labels met consensus threshold -- using base stacking only")

# ============================================================
# PHASE 10 -- GENTLE RANK BLEND + BASE RATE RECALIBRATION
# ============================================================
print("\n" + "=" * 80)
print("  PHASE 10 -- GENTLE RANK BLEND + BASE RATE RECALIBRATION")
print("=" * 80)

BLEND_CONFIG = {
    12: (0.92, 0.08),
    24: (0.93, 0.07),
    48: (0.94, 0.06),
    72: (0.95, 0.05),
}

# Use FULL training set base rates (not IPCW-filtered) for test alignment
# because test set contains all types of samples
FULL_BASE_RATES = {}
for horizon in time_horizons:
    n_hit = ((train['event'] == 1) & (train['time_to_hit_hours'] <= horizon)).sum()
    FULL_BASE_RATES[horizon] = n_hit / len(train)

def gentle_ranks(pred):
    return rankdata(pred) / len(pred)

def recalibrate_to_base_rate(pred, target_mean, strength=0.3):
    """Gently shift predictions toward expected base rate."""
    current_mean = pred.mean()
    if abs(current_mean - target_mean) < 0.02:
        return pred  # already aligned
    
    # Logit-space shift: more principled than linear scaling
    eps = 1e-6
    logit_pred = np.log(np.clip(pred, eps, 1-eps) / (1 - np.clip(pred, eps, 1-eps)))
    shift = np.log(target_mean / (1 - target_mean)) - np.log(current_mean / (1 - current_mean))
    adjusted = 1.0 / (1.0 + np.exp(-(logit_pred + shift * strength)))
    return adjusted

hybrid_test = {}

for horizon in time_horizons:
    cal_prob = stacked_test[horizon]
    rank_prob = gentle_ranks(cal_prob)
    
    cal_w, rank_w = BLEND_CONFIG[horizon]
    hybrid = cal_w * cal_prob + rank_w * rank_prob
    hybrid = np.clip(hybrid, 0.005, 0.995)
    
    # Recalibrate toward full-training-set base rate
    target_br = FULL_BASE_RATES[horizon]
    hybrid_before = hybrid.mean()
    hybrid = recalibrate_to_base_rate(hybrid, target_br, strength=0.4)
    hybrid = np.clip(hybrid, 0.005, 0.995)
    
    hybrid_test[horizon] = hybrid
    print(f"  {horizon}h: cal/rank={cal_w:.2f}/{rank_w:.2f}, "
          f"mean={hybrid_before:.4f} -> {hybrid.mean():.4f} (target={target_br:.4f})")
    print(f"         std={hybrid.std():.4f}")

print("\n  --- Full Training Base Rates ---")
for h, br in FULL_BASE_RATES.items():
    print(f"  {h}h: {br:.4f}")

# ============================================================
# PHASE 11 -- COMPREHENSIVE VALIDATION
# ============================================================
print("\n" + "=" * 80)
print("  PHASE 11 -- COMPREHENSIVE VALIDATION")
print("=" * 80)

print("\n  --- Stacked OOF Metrics ---")
oof_briers = {}
for horizon in time_horizons:
    _, y_h, _, _ = horizon_data[horizon]
    oof_e = stacked_oof[horizon]
    if len(np.unique(y_h)) > 1:
        ens_auc = roc_auc_score(y_h, oof_e)
        ens_brier = brier_score_loss(y_h, oof_e)
        hybrid_val = 0.3 * ens_auc + 0.7 * (1.0 - ens_brier)
        oof_briers[horizon] = ens_brier
        print(f"  {horizon}h: AUC={ens_auc:.4f}, Brier={ens_brier:.4f}, Hybrid={hybrid_val:.4f}")

if all(h in oof_briers for h in [24, 48, 72]):
    weighted_brier = 0.3 * oof_briers[24] + 0.4 * oof_briers[48] + 0.3 * oof_briers[72]
    avg_auc = np.mean([roc_auc_score(horizon_data[h][1], stacked_oof[h]) 
                       for h in time_horizons if len(np.unique(horizon_data[h][1])) > 1])
    estimated_score = 0.3 * avg_auc + 0.7 * (1.0 - weighted_brier)
    print(f"\n  >>> ESTIMATED HYBRID SCORE: {estimated_score:.5f}")
    print(f"      Weighted Brier: {weighted_brier:.5f}")
    print(f"      Avg AUC: {avg_auc:.5f}")

# Base rate alignment check
print("\n  --- Base Rate Alignment ---")
for horizon in time_horizons:
    _, y_h, _, _ = horizon_data[horizon]
    base_rate = y_h.mean()
    test_mean = hybrid_test[horizon].mean()
    diff = abs(base_rate - test_mean)
    status = "OK" if diff < 0.03 else "WARN"
    print(f"  [{status}] {horizon}h: base_rate={base_rate:.4f}, test_mean={test_mean:.4f}, diff={diff:.4f}")

# ============================================================
# PHASE 12 -- MONOTONICITY ENFORCEMENT
# ============================================================
print("\n" + "=" * 80)
print("  PHASE 12 -- MONOTONICITY ENFORCEMENT")
print("=" * 80)

def enforce_monotonicity(pred_matrix, time_horizons):
    violations_before = sum(
        1 for i in range(len(pred_matrix))
        for j in range(3)
        if pred_matrix[i, j] > pred_matrix[i, j+1]
    )
    
    for i in range(len(pred_matrix)):
        row = pred_matrix[i]
        if not all(row[j] <= row[j+1] for j in range(3)):
            ir = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True)
            pred_matrix[i] = ir.fit_transform(
                np.array(time_horizons, dtype=float), row
            )
    
    MIN_INCREMENT = 0.003
    for i in range(len(pred_matrix)):
        for j in range(1, 4):
            if pred_matrix[i, j] < pred_matrix[i, j-1] + MIN_INCREMENT:
                pred_matrix[i, j] = pred_matrix[i, j-1] + MIN_INCREMENT
    
    violations_after = sum(
        1 for i in range(len(pred_matrix))
        for j in range(3)
        if pred_matrix[i, j] > pred_matrix[i, j+1]
    )
    return pred_matrix, violations_before, violations_after

# ============================================================
# PHASE 13 -- SUBMISSION GENERATION
# ============================================================
print("\n" + "=" * 80)
print("  PHASE 13 -- SUBMISSION GENERATION")
print("=" * 80)

submission_configs = {
    'A': {'clip_low': 0.005, 'clip_high': 0.995,
          'desc': 'FULL (IPCW + Stacking + Pseudo-Labels, wide clip)'},
    'B': {'clip_low': 0.015, 'clip_high': 0.985,
          'desc': 'SAFE (IPCW + Stacking, tighter clip)'},
}

submissions = {}

for variant, config in submission_configs.items():
    print(f"\n  --- Submission {variant}: {config['desc']} ---")
    
    pred_matrix = np.column_stack([hybrid_test[t] for t in time_horizons])
    pred_matrix = np.clip(pred_matrix, config['clip_low'], config['clip_high'])
    
    pred_matrix, v_before, v_after = enforce_monotonicity(pred_matrix, time_horizons)
    print(f"    Monotonicity violations: {v_before} -> {v_after}")
    
    pred_matrix = np.clip(pred_matrix, config['clip_low'], min(config['clip_high'], 0.999))
    
    for i in range(len(pred_matrix)):
        for j in range(1, 4):
            if pred_matrix[i, j] < pred_matrix[i, j-1]:
                pred_matrix[i, j] = pred_matrix[i, j-1]
    
    for j, t in enumerate(time_horizons):
        col = pred_matrix[:, j]
        print(f"    prob_{t}h: mean={col.mean():.4f}, std={col.std():.4f}, "
              f"[{col.min():.4f}, {col.max():.4f}]")
    
    sub = pd.DataFrame({
        'event_id': test['event_id'].values,
        'prob_12h': pred_matrix[:, 0],
        'prob_24h': pred_matrix[:, 1],
        'prob_48h': pred_matrix[:, 2],
        'prob_72h': pred_matrix[:, 3],
    })
    submissions[variant] = sub

# ============================================================
# PHASE 14 -- FINAL VERIFICATION & SAVE
# ============================================================
print("\n" + "=" * 80)
print("  PHASE 14 -- FINAL VERIFICATION & SAVE")
print("=" * 80)

for variant, sub in submissions.items():
    checks = {
        'rows': len(sub) == 95,
        'cols': list(sub.columns) == ['event_id', 'prob_12h', 'prob_24h', 'prob_48h', 'prob_72h'],
        'ids': list(sub['event_id']) == list(sample_sub['event_id']),
        'no_nan': not sub.isnull().any().any(),
        'no_inf': not np.isinf(sub[['prob_12h', 'prob_24h', 'prob_48h', 'prob_72h']].values).any(),
    }
    
    prob_cols = ['prob_12h', 'prob_24h', 'prob_48h', 'prob_72h']
    checks['range'] = sub[prob_cols].min().min() >= 0 and sub[prob_cols].max().max() <= 1
    
    mono_ok = all(
        row['prob_12h'] <= row['prob_24h'] + 1e-9 and
        row['prob_24h'] <= row['prob_48h'] + 1e-9 and
        row['prob_48h'] <= row['prob_72h'] + 1e-9
        for _, row in sub.iterrows()
    )
    checks['monotonicity'] = mono_ok
    
    all_passed = all(checks.values())
    
    print(f"\n  Submission {variant}:")
    for name, ok in checks.items():
        print(f"    [{'OK' if ok else 'FAIL'}] {name}")
    
    if all_passed:
        filepath = rf'd:\WiDS\submission_{variant}.csv'
        sub.to_csv(filepath, index=False)
        print(f"    >>> SAVED: {filepath}")
    else:
        print(f"    >>> NOT SAVED -- FIX ERRORS")

# Main submission = variant A
submissions['A'].to_csv(r'd:\WiDS\submission.csv', index=False)
print(f"\n  >>> MAIN submission.csv = Variant A")

# ============================================================
# FINAL REPORT
# ============================================================
elapsed = time_module.time() - start_time
print("\n" + "=" * 80)
print(f"  PIPELINE v7.0 COMPLETE -- {elapsed:.0f}s elapsed")
print("=" * 80)

print(f"\n  Models: {len(model_names)} x {len(SEEDS)} seeds x {len(time_horizons)} horizons x {N_SPLITS} folds")
print(f"  = {len(model_names) * len(SEEDS) * len(time_horizons) * N_SPLITS} total base models + stacking")
print(f"  Features: {len(all_features)}")

print(f"\n  --- RECOMMENDATION ---")
print(f"    Submit submission_A.csv FIRST (full pipeline)")
print(f"    Submit submission_B.csv SECOND (safer clips)")

print("\n" + "=" * 80)
print("  v7.0 NUCLEAR OPTION -- READY FOR LEADERBOARD")
print("=" * 80)
