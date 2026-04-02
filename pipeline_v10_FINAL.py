"""
pipeline_v10_FINAL.py — THE ABSOLUTE ULTIMATE PIPELINE
WiDS Global Datathon 2026: Wildfire Survival Analysis
Target: 0.975+ Leaderboard Score

NUCLEAR CHANGES from v9:
  1.  6 MODELS: LGB + CatBoost + XGB + ExtraTrees + RF + LogReg
  2.  10 SEEDS (up from 7) for maximum stability
  3.  100 OPTUNA TRIALS (up from 60) for better hyperparams
  4.  HORIZON-SPECIFIC OPTIMIZATION:
      - 12h: optimize AUC (only affects C-index, 30% of score)
      - 24h/48h/72h: optimize BRIER (affects 70% of score)
  5.  PSEUDO-LABELING: train → predict test → add high-confidence → retrain
  6.  SCIPY-OPTIMIZED ensemble weights per horizon (LOO-robust)
  7.  6-MODEL DIVERSITY (boosting + bagging + linear + random splits)
  8.  EARLY STOPPING patience=50 on all GBMs
  9.  MULTI-STAGE CALIBRATION (Raw + Platt + CV-Isotonic) per model
  10. SMART CLIPPING: based on calibration confidence
  11. GENTLE RANK BLEND (3-7% rank component)
  12. PROPER monotonicity with minimum increment
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
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import time

start_time = time.time()

# ============================================================
# CONFIGURATION
# ============================================================
TIME_HORIZONS = [12, 24, 48, 72]
SEEDS = [42, 52, 62, 72, 82, 92, 102, 112, 122, 132]  # 10 seeds
N_SPLITS = 5
N_TRIALS = 100  # Optuna trials
MAX_FEATURES = 30
CORR_THRESHOLD = 0.95
MAX_IPCW_WEIGHT = 3.0
CALIBRATION_STRENGTH = 0.30
MONOTONICITY_INCREMENT = 0.003
PSEUDO_LABEL_THRESHOLD = 0.92  # Only pseudo-label very confident predictions

CLIP_VARIANTS = {
    'A': (0.005, 0.995),
    'B': (0.010, 0.990),
    'C': (0.020, 0.980),
}

BLEND_CONFIG = {
    12: (0.93, 0.07),
    24: (0.94, 0.06),
    48: (0.95, 0.05),
    72: (0.96, 0.04),
}

# ============================================================
# DATA LOADING
# ============================================================
print("=" * 70)
print("WiDS 2026 — Pipeline v10 FINAL (NUCLEAR)")
print("=" * 70)

train = pd.read_csv('d:/WiDS/train.csv')
test = pd.read_csv('d:/WiDS/test.csv')
sample = pd.read_csv('d:/WiDS/sample_submission.csv')

print(f"Train: {train.shape}, Test: {test.shape}")
for h in TIME_HORIZONS:
    n = ((train['event'] == 1) & (train['time_to_hit_hours'] <= h)).sum()
    print(f"  Hits <= {h}h: {n} ({n/len(train)*100:.1f}%)")

# ============================================================
# PHASE 1 — FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 70)
print("PHASE 1: Feature Engineering")
print("=" * 70)

def engineer_features(df):
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        if col not in ['event_id', 'event', 'time_to_hit_hours']:
            out[col] = df[col]

    # Distance transforms (proven top predictors)
    out['log_dist'] = np.log1p(df['dist_min_ci_0_5h'].clip(lower=0))
    out['inv_dist'] = 1.0 / (df['dist_min_ci_0_5h'].clip(lower=100) + 1)
    out['sqrt_dist'] = np.sqrt(df['dist_min_ci_0_5h'].clip(lower=0))

    # Time-to-hit projection
    safe_closing = df['closing_speed_m_per_h'].clip(lower=0.001)
    out['projected_time_to_hit'] = (df['dist_min_ci_0_5h'] / safe_closing).clip(0, 500)
    out['log_time_projected'] = np.log1p(out['projected_time_to_hit'])

    # Threat interactions
    out['directional_threat'] = df['closing_speed_m_per_h'] * df['alignment_abs']
    out['proximity_threat'] = df['closing_speed_m_per_h'] / (df['dist_min_ci_0_5h'] + 100)

    # Near miss margin
    out['near_miss_margin'] = (
        df['dist_min_ci_0_5h'] - df['radial_growth_m']
        - df['projected_advance_m'].clip(lower=0)
    )
    out['log_near_miss'] = np.log1p(out['near_miss_margin'].clip(lower=0))

    # Composite risk score
    dist_max = df['dist_min_ci_0_5h'].max() + 1
    speed_max = df['closing_speed_m_per_h'].abs().max() + 1
    align_max = df['alignment_abs'].max() + 1
    out['risk_score'] = (
        (1 - df['dist_min_ci_0_5h'] / dist_max) * 0.5
        + (df['closing_speed_m_per_h'] / speed_max).clip(0, 1) * 0.3
        + (df['alignment_abs'] / align_max) * 0.2
    )

    # Temporal
    out['hour_sin'] = np.sin(2 * np.pi * df['event_start_hour'] / 24)
    out['hour_cos'] = np.cos(2 * np.pi * df['event_start_hour'] / 24)

    # Binary flags
    out['is_close'] = (df['dist_min_ci_0_5h'] < 5000).astype(float)
    out['is_very_close'] = (df['dist_min_ci_0_5h'] < 2000).astype(float)
    out['is_approaching'] = (df['closing_speed_m_per_h'] > 0).astype(float)
    out['danger_zone'] = (
        (df['dist_min_ci_0_5h'] < 5000) & (df['closing_speed_m_per_h'] > 0)
    ).astype(float)

    # Hazard proxy
    out['hazard_ratio_proxy'] = df['closing_speed_m_per_h'] / (df['dist_min_ci_0_5h'] + 10)

    # Growth features
    out['spread_acceleration'] = df['area_growth_rate_ha_per_h'] * out['directional_threat']
    out['growth_factor_prox'] = df['radial_growth_rate_m_per_h'] / (df['dist_min_ci_0_5h'] + 1)
    out['log_area'] = np.log1p(df['area_first_ha'].clip(lower=0))

    # Interactions
    out['dist_x_time'] = df['dist_min_ci_0_5h'] * df['event_start_hour']
    out['dist_x_alignment'] = df['dist_min_ci_0_5h'] * df['alignment_abs']
    out['speed_x_close'] = df['closing_speed_m_per_h'] * out['is_close']

    # Relative closing
    out['relative_closing'] = df['closing_speed_m_per_h'] / (
        df['radial_growth_rate_m_per_h'].clip(lower=0.1) + 1)

    # Night fire
    out['night_fire'] = (
        (df['event_start_hour'] >= 20) | (df['event_start_hour'] <= 6)
    ).astype(float)

    # NEW: threat index (combined)
    out['threat_index'] = (out['hazard_ratio_proxy'] * 100 +
                           out['risk_score'] * 50 +
                           out['danger_zone'] * 30)

    return out

train_fe = engineer_features(train)
test_fe = engineer_features(test)

# Feature selection
candidate_features = [c for c in train_fe.columns
                      if c not in ['event_id', 'event', 'time_to_hit_hours']]
train_std = train_fe[candidate_features].std()
candidate_features = [f for f in candidate_features if train_std[f] > 1e-10]

PROTECTED = {
    'dist_min_ci_0_5h', 'closing_speed_m_per_h', 'alignment_abs',
    'log_dist', 'inv_dist', 'projected_time_to_hit', 'risk_score',
    'is_close', 'near_miss_margin', 'danger_zone', 'hazard_ratio_proxy',
    'directional_threat', 'night_fire', 'threat_index'
}

corr_matrix = train_fe[candidate_features].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = set()
for col in upper.columns:
    high_corr = upper.index[upper[col] > CORR_THRESHOLD].tolist()
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

if len(candidate_features) > MAX_FEATURES:
    corrs = train_fe[candidate_features].corrwith(train['event']).abs()
    candidate_features = corrs.sort_values(ascending=False).head(MAX_FEATURES).index.tolist()

ALL_FEATURES = candidate_features
print(f"Final features: {len(ALL_FEATURES)}")

# Winsorize
for col in ALL_FEATURES:
    p1, p99 = train_fe[col].quantile(0.01), train_fe[col].quantile(0.99)
    train_fe[col] = train_fe[col].clip(p1, p99)
    test_fe[col] = test_fe[col].clip(p1, p99)

X_train_full = np.nan_to_num(train_fe[ALL_FEATURES].values.astype(np.float32))
X_test = np.nan_to_num(test_fe[ALL_FEATURES].values.astype(np.float32))

# ============================================================
# PHASE 2 — IPCW WEIGHTS
# ============================================================
print("\n" + "=" * 70)
print("PHASE 2: IPCW Weights")
print("=" * 70)

def compute_ipcw_weights(train_df, horizon, max_weight=MAX_IPCW_WEIGHT):
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
    print(f"  {h}h: {mask.sum()} samples, {int(labels[mask].sum())} pos, "
          f"{int((labels[mask]==0).sum())} neg")

# ============================================================
# PHASE 3 — OPTUNA TUNING (100 trials, horizon-specific objective)
# ============================================================
print("\n" + "=" * 70)
print("PHASE 3: Optuna Tuning (100 trials × 3 GBM models × 4 horizons)")
print("=" * 70)

def run_optuna(X, y, w, model_type, horizon, n_trials=N_TRIALS):
    """
    CRITICAL: 12h optimizes AUC (only C-index matters)
              24/48/72h optimizes BRIER (70% of final score)
    """
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
                    'learning_rate': trial.suggest_float('lr', 0.01, 0.06),
                    'reg_alpha': trial.suggest_float('reg_a', 1.0, 30.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_l', 2.0, 60.0, log=True),
                    'min_child_samples': trial.suggest_int('mcs', 12, 40),
                    'subsample': trial.suggest_float('sub', 0.5, 0.85),
                    'colsample_bytree': trial.suggest_float('col', 0.35, 0.75),
                    'min_split_gain': trial.suggest_float('msg', 0.01, 1.5),
                }
                mdl = lgb.LGBMClassifier(**params, n_estimators=500, objective='binary',
                                          verbosity=-1, random_state=42)
                mdl.fit(X_tr, y_tr, sample_weight=w_tr,
                       eval_set=[(X_val, y_val)],
                       callbacks=[lgb.early_stopping(50, verbose=False)])

            elif model_type == 'catboost':
                params = {
                    'depth': trial.suggest_int('depth', 2, 4),
                    'learning_rate': trial.suggest_float('lr', 0.01, 0.06),
                    'l2_leaf_reg': trial.suggest_float('l2', 5.0, 60.0, log=True),
                    'min_data_in_leaf': trial.suggest_int('mdl', 8, 30),
                    'subsample': trial.suggest_float('sub', 0.5, 0.85),
                }
                mdl = CatBoostClassifier(**params, iterations=500, verbose=0,
                                          random_seed=42, early_stopping_rounds=50)
                mdl.fit(X_tr, y_tr, sample_weight=w_tr,
                       eval_set=(X_val, y_val), verbose=0)

            elif model_type == 'xgb':
                params = {
                    'max_depth': trial.suggest_int('max_depth', 2, 4),
                    'learning_rate': trial.suggest_float('lr', 0.01, 0.06),
                    'reg_alpha': trial.suggest_float('reg_a', 1.0, 30.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_l', 2.0, 60.0, log=True),
                    'min_child_weight': trial.suggest_int('mcw', 8, 40),
                    'subsample': trial.suggest_float('sub', 0.5, 0.85),
                    'colsample_bytree': trial.suggest_float('col', 0.35, 0.75),
                    'gamma': trial.suggest_float('gamma', 0.1, 5.0),
                }
                mdl = xgb.XGBClassifier(**params, n_estimators=500,
                                         objective='binary:logistic', tree_method='hist',
                                         verbosity=0, random_state=42,
                                         early_stopping_rounds=50, eval_metric='logloss')
                mdl.fit(X_tr, y_tr, sample_weight=w_tr,
                       eval_set=[(X_val, y_val)], verbose=False)

            p = np.clip(mdl.predict_proba(X_val)[:, 1], 0.001, 0.999)

            if horizon == 12:
                # 12h: only C-index matters → optimize AUC
                if len(np.unique(y_val)) > 1:
                    metrics.append(-roc_auc_score(y_val, p))  # Negative for minimization
                else:
                    metrics.append(0)
            else:
                # 24/48/72h: Brier dominates → optimize Brier
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
    print(f"\n--- {h}h (optimizing {opt_type}) ---")

    for mt in ['lgbm', 'catboost', 'xgb']:
        bp, bv = run_optuna(X_f, y_f, w_f, mt, h)
        best_params[(mt, h)] = bp
        print(f"  {mt:10s} best {opt_type}: {abs(bv):.5f}")

# ============================================================
# PHASE 4 — MULTI-SEED 5-FOLD CV TRAINING (10 seeds × 5 folds × 6 models)
# ============================================================
print("\n" + "=" * 70)
print("PHASE 4: Training (10 seeds × 5 folds × 6 models × 4 horizons)")
print("=" * 70)

MODEL_TYPES = ['lgbm', 'catboost', 'xgb', 'extratrees', 'rf', 'logreg']
oof_preds = {h: {} for h in TIME_HORIZONS}
test_preds = {h: {} for h in TIME_HORIZONS}

# Remap Optuna short keys to full model parameter names
LGBM_REMAP = {'lr': 'learning_rate', 'reg_a': 'reg_alpha', 'reg_l': 'reg_lambda',
              'mcs': 'min_child_samples', 'sub': 'subsample', 'col': 'colsample_bytree',
              'msg': 'min_split_gain'}
CB_REMAP = {'lr': 'learning_rate', 'l2': 'l2_leaf_reg', 'mdl': 'min_data_in_leaf',
            'sub': 'subsample'}
XGB_REMAP = {'lr': 'learning_rate', 'reg_a': 'reg_alpha', 'reg_l': 'reg_lambda',
             'mcw': 'min_child_weight', 'sub': 'subsample', 'col': 'colsample_bytree'}

def remap_params(params, remap_dict):
    return {remap_dict.get(k, k): v for k, v in params.items()}

for h in TIME_HORIZONS:
    y_h, w_h, mask = ipcw_data[h]
    X_f, y_f, w_f = X_train_full[mask], y_h[mask], w_h[mask]
    print(f"\n  {h}h: {len(y_f)} samples, {int(y_f.sum())} pos, 6 models")

    for m in MODEL_TYPES:
        oof_preds[h][m] = np.zeros(len(y_f))
        test_preds[h][m] = np.zeros(len(X_test))

        for seed in SEEDS:
            skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
            for tr_idx, val_idx in skf.split(X_f, y_f):
                if m == 'lgbm':
                    bp = remap_params(best_params[('lgbm', h)], LGBM_REMAP)
                    mdl = lgb.LGBMClassifier(**bp, n_estimators=500, objective='binary',
                                              verbosity=-1, random_state=seed)
                    mdl.fit(X_f[tr_idx], y_f[tr_idx], sample_weight=w_f[tr_idx],
                           eval_set=[(X_f[val_idx], y_f[val_idx])],
                           callbacks=[lgb.early_stopping(50, verbose=False)])

                elif m == 'catboost':
                    bp = remap_params(best_params[('catboost', h)], CB_REMAP)
                    mdl = CatBoostClassifier(**bp, iterations=500, verbose=0,
                                              random_seed=seed, early_stopping_rounds=50)
                    mdl.fit(X_f[tr_idx], y_f[tr_idx], sample_weight=w_f[tr_idx],
                           eval_set=(X_f[val_idx], y_f[val_idx]), verbose=0)

                elif m == 'xgb':
                    bp = remap_params(best_params[('xgb', h)], XGB_REMAP)
                    mdl = xgb.XGBClassifier(**bp, n_estimators=500,
                                             objective='binary:logistic', tree_method='hist',
                                             verbosity=0, random_state=seed,
                                             early_stopping_rounds=50, eval_metric='logloss')
                    mdl.fit(X_f[tr_idx], y_f[tr_idx], sample_weight=w_f[tr_idx],
                           eval_set=[(X_f[val_idx], y_f[val_idx])], verbose=False)

                elif m == 'extratrees':
                    mdl = ExtraTreesClassifier(
                        n_estimators=500, max_depth=4, min_samples_leaf=12,
                        min_samples_split=15, max_features=0.45,
                        class_weight='balanced', random_state=seed, n_jobs=-1)
                    mdl.fit(X_f[tr_idx], y_f[tr_idx], sample_weight=w_f[tr_idx])

                elif m == 'rf':
                    mdl = RandomForestClassifier(
                        max_depth=4, min_samples_leaf=12, n_estimators=400,
                        max_features=0.5, random_state=seed, n_jobs=-1)
                    mdl.fit(X_f[tr_idx], y_f[tr_idx], sample_weight=w_f[tr_idx])

                elif m == 'logreg':
                    mdl = LogisticRegression(C=0.01, penalty='l2', max_iter=500,
                                              random_state=seed)
                    mdl.fit(X_f[tr_idx], y_f[tr_idx], sample_weight=w_f[tr_idx])

                oof_preds[h][m][val_idx] += mdl.predict_proba(X_f[val_idx])[:, 1] / len(SEEDS)
                test_preds[h][m] += mdl.predict_proba(X_test)[:, 1] / (len(SEEDS) * N_SPLITS)

    # Print metrics
    for m in MODEL_TYPES:
        p = np.clip(oof_preds[h][m], 0.001, 0.999)
        if len(np.unique(y_f)) > 1:
            auc = roc_auc_score(y_f, p)
            brier = brier_score_loss(y_f, p)
            print(f"    {m:12s}: AUC={auc:.4f} Brier={brier:.4f}")

# ============================================================
# PHASE 5 — PSEUDO-LABELING (semi-supervised, round 1)
# ============================================================
print("\n" + "=" * 70)
print("PHASE 5: Pseudo-Labeling (high-confidence test augmentation)")
print("=" * 70)

pseudo_oof = {h: {} for h in TIME_HORIZONS}
pseudo_test = {h: {} for h in TIME_HORIZONS}

for h in TIME_HORIZONS:
    y_h, w_h, mask = ipcw_data[h]
    X_f, y_f, w_f = X_train_full[mask], y_h[mask], w_h[mask]

    # Average test predictions across all models for pseudo-label selection
    avg_test = np.mean([test_preds[h][m] for m in MODEL_TYPES], axis=0)

    # Find high-confidence pseudo-labels
    confident_pos = avg_test > PSEUDO_LABEL_THRESHOLD
    confident_neg = avg_test < (1 - PSEUDO_LABEL_THRESHOLD)
    n_pseudo = confident_pos.sum() + confident_neg.sum()

    if n_pseudo >= 5:
        # Create augmented dataset
        pseudo_X = X_test[confident_pos | confident_neg]
        pseudo_y = np.where(avg_test[confident_pos | confident_neg] > 0.5, 1.0, 0.0)
        pseudo_w = np.ones(len(pseudo_y)) * 0.5  # Half weight for pseudo-labels

        X_aug = np.vstack([X_f, pseudo_X])
        y_aug = np.concatenate([y_f, pseudo_y])
        w_aug = np.concatenate([w_f, pseudo_w])

        print(f"  {h}h: Added {n_pseudo} pseudo-labels ({confident_pos.sum()} pos, "
              f"{confident_neg.sum()} neg) -> {len(y_aug)} total samples")

        # Retrain ALL models with augmented data
        for m in MODEL_TYPES:
            pseudo_oof[h][m] = np.zeros(len(y_f))  # OOF still on original data
            pseudo_test[h][m] = np.zeros(len(X_test))

            for seed in SEEDS[:5]:  # 5 seeds for pseudo-round (speed)
                skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
                for tr_idx, val_idx in skf.split(X_f, y_f):
                    # Train on original train fold + ALL pseudo-labels
                    X_tr = np.vstack([X_f[tr_idx], pseudo_X])
                    y_tr = np.concatenate([y_f[tr_idx], pseudo_y])
                    w_tr = np.concatenate([w_f[tr_idx], pseudo_w])

                    if m == 'lgbm':
                        bp = remap_params(best_params[('lgbm', h)], LGBM_REMAP)
                        mdl = lgb.LGBMClassifier(**bp, n_estimators=500, objective='binary',
                                                  verbosity=-1, random_state=seed)
                        mdl.fit(X_tr, y_tr, sample_weight=w_tr,
                               eval_set=[(X_f[val_idx], y_f[val_idx])],
                               callbacks=[lgb.early_stopping(50, verbose=False)])
                    elif m == 'catboost':
                        bp = remap_params(best_params[('catboost', h)], CB_REMAP)
                        mdl = CatBoostClassifier(**bp, iterations=500, verbose=0,
                                                  random_seed=seed, early_stopping_rounds=50)
                        mdl.fit(X_tr, y_tr, sample_weight=w_tr,
                               eval_set=(X_f[val_idx], y_f[val_idx]), verbose=0)
                    elif m == 'xgb':
                        bp = remap_params(best_params[('xgb', h)], XGB_REMAP)
                        mdl = xgb.XGBClassifier(**bp, n_estimators=500,
                                                 objective='binary:logistic', tree_method='hist',
                                                 verbosity=0, random_state=seed,
                                                 early_stopping_rounds=50, eval_metric='logloss')
                        mdl.fit(X_tr, y_tr, sample_weight=w_tr,
                               eval_set=[(X_f[val_idx], y_f[val_idx])], verbose=False)
                    elif m == 'extratrees':
                        mdl = ExtraTreesClassifier(n_estimators=500, max_depth=4,
                                                    min_samples_leaf=12, max_features=0.45,
                                                    class_weight='balanced', random_state=seed, n_jobs=-1)
                        mdl.fit(X_tr, y_tr, sample_weight=w_tr)
                    elif m == 'rf':
                        mdl = RandomForestClassifier(max_depth=4, min_samples_leaf=12,
                                                      n_estimators=400, max_features=0.5,
                                                      random_state=seed, n_jobs=-1)
                        mdl.fit(X_tr, y_tr, sample_weight=w_tr)
                    elif m == 'logreg':
                        mdl = LogisticRegression(C=0.01, penalty='l2', max_iter=500,
                                                  random_state=seed)
                        mdl.fit(X_tr, y_tr, sample_weight=w_tr)

                    pseudo_oof[h][m][val_idx] += mdl.predict_proba(X_f[val_idx])[:, 1] / 5
                    pseudo_test[h][m] += mdl.predict_proba(X_test)[:, 1] / (5 * N_SPLITS)

        # Compare: use pseudo-labeled if it improves Brier
        for m in MODEL_TYPES:
            orig_brier = brier_score_loss(y_f, np.clip(oof_preds[h][m], 0.001, 0.999))
            pseudo_brier = brier_score_loss(y_f, np.clip(pseudo_oof[h][m], 0.001, 0.999))
            if pseudo_brier < orig_brier:
                # Blend: 50% original + 50% pseudo-augmented
                oof_preds[h][m] = 0.5 * oof_preds[h][m] + 0.5 * pseudo_oof[h][m]
                test_preds[h][m] = 0.5 * test_preds[h][m] + 0.5 * pseudo_test[h][m]
                print(f"    {h}h {m}: pseudo IMPROVED Brier {orig_brier:.4f}->{pseudo_brier:.4f} -> blended")
            else:
                print(f"    {h}h {m}: pseudo NO improvement {orig_brier:.4f} vs {pseudo_brier:.4f} -> kept original")
    else:
        print(f"  {h}h: Only {n_pseudo} confident samples, skipping pseudo-labeling")

# ============================================================
# PHASE 6 — CALIBRATION (Raw + Platt + CV-Isotonic)
# ============================================================
print("\n" + "=" * 70)
print("PHASE 6: Multi-Stage Calibration")
print("=" * 70)

calibrated_oof = {h: {} for h in TIME_HORIZONS}
calibrated_test = {h: {} for h in TIME_HORIZONS}

for h in TIME_HORIZONS:
    y_h, _, mask = ipcw_data[h]
    y_f = y_h[mask]
    train_base = y_f.mean()

    for m in MODEL_TYPES:
        r_oof = oof_preds[h][m]
        r_test = test_preds[h][m]

        brier_raw = brier_score_loss(y_f, np.clip(r_oof, 0.001, 0.999))

        # Platt scaling
        lr = LogisticRegression(max_iter=1000)
        lr.fit(r_oof.reshape(-1, 1), y_f)
        p_platt = lr.predict_proba(r_oof.reshape(-1, 1))[:, 1]
        brier_platt = brier_score_loss(y_f, np.clip(p_platt, 0.001, 0.999))

        # CV-Isotonic
        p_iso = np.zeros(len(y_f))
        iso_tests = []
        for tr_i, val_i in StratifiedKFold(5, shuffle=True, random_state=42).split(r_oof.reshape(-1,1), y_f):
            iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
            iso.fit(r_oof[tr_i], y_f[tr_i])
            p_iso[val_i] = iso.predict(r_oof[val_i])
            iso_tests.append(iso.predict(r_test))
        iso_test = np.mean(iso_tests, axis=0)
        brier_iso = brier_score_loss(y_f, np.clip(p_iso, 0.001, 0.999))

        best = min(brier_raw, brier_platt, brier_iso)
        if best == brier_iso:
            calibrated_oof[h][m] = p_iso
            calibrated_test[h][m] = iso_test
            tag = 'cv-iso'
        elif best == brier_platt:
            calibrated_oof[h][m] = p_platt
            calibrated_test[h][m] = lr.predict_proba(r_test.reshape(-1,1))[:, 1]
            tag = 'platt'
        else:
            calibrated_oof[h][m] = r_oof
            calibrated_test[h][m] = r_test
            tag = 'raw'

        print(f"  {h}h {m:12s}: raw={brier_raw:.4f} platt={brier_platt:.4f} "
              f"iso={brier_iso:.4f} -> {tag}")

        # Logit-space base rate alignment
        tp = np.clip(calibrated_test[h][m], 1e-5, 1-1e-5)
        test_base = tp.mean()
        if abs(test_base - train_base) > 0.001:
            shift = logit(train_base) - logit(test_base)
            calibrated_test[h][m] = expit(logit(tp) + CALIBRATION_STRENGTH * shift)

# ============================================================
# PHASE 7 — SCIPY-OPTIMIZED ENSEMBLE WEIGHTS
# ============================================================
print("\n" + "=" * 70)
print("PHASE 7: Scipy-Optimized Ensemble Weights")
print("=" * 70)

ensemble_oof = {}
ensemble_test = {}

for h in TIME_HORIZONS:
    y_h, _, mask = ipcw_data[h]
    y_f = y_h[mask]
    models = MODEL_TYPES

    # Collect OOF predictions
    oof_matrix = np.column_stack([np.clip(calibrated_oof[h][m], 0.001, 0.999) for m in models])
    test_matrix = np.column_stack([np.clip(calibrated_test[h][m], 1e-5, 1-1e-5) for m in models])

    # Compute model scores for initial weights
    scores = []
    for i, m in enumerate(models):
        auc = roc_auc_score(y_f, oof_matrix[:, i]) if len(np.unique(y_f)) > 1 else 0.5
        brier = brier_score_loss(y_f, oof_matrix[:, i])
        hybrid = 0.3 * auc + 0.7 * (1 - brier)
        scores.append(hybrid)

    scores = np.array(scores)

    # Method 1: Softmax geometric mean (robust)
    w_soft = np.exp(5 * (scores - scores.max()))
    w_soft = np.maximum(w_soft / w_soft.sum(), 0.02)
    w_soft /= w_soft.sum()

    geo_oof = np.prod(oof_matrix ** w_soft, axis=1)
    geo_test = np.prod(test_matrix ** w_soft, axis=1)
    b_geo = brier_score_loss(y_f, np.clip(geo_oof, 0.001, 0.999))

    # Method 2: Scipy-optimized weights for arithmetic mean (minimize Brier)
    def brier_obj(w):
        w = np.abs(w)
        w = w / w.sum()
        blend = oof_matrix @ w
        return brier_score_loss(y_f, np.clip(blend, 0.001, 0.999))

    w0 = np.ones(len(models)) / len(models)
    # Multiple restarts for robustness
    best_result = None
    for rs in range(5):
        w_init = w0 + np.random.RandomState(rs).randn(len(models)) * 0.05
        w_init = np.abs(w_init)
        w_init /= w_init.sum()
        result = minimize(brier_obj, w_init, method='Nelder-Mead',
                         options={'maxiter': 2000, 'xatol': 1e-8})
        if best_result is None or result.fun < best_result.fun:
            best_result = result

    w_opt = np.abs(best_result.x)
    w_opt /= w_opt.sum()

    # Apply minimum weight floor (prevent collapse to single model)
    w_opt = np.maximum(w_opt, 0.02)
    w_opt /= w_opt.sum()

    arith_oof = oof_matrix @ w_opt
    arith_test = test_matrix @ w_opt
    b_arith = brier_score_loss(y_f, np.clip(arith_oof, 0.001, 0.999))

    # Method 3: Simple average
    avg_oof = oof_matrix.mean(axis=1)
    avg_test = test_matrix.mean(axis=1)
    b_avg = brier_score_loss(y_f, np.clip(avg_oof, 0.001, 0.999))

    # Pick best method
    best_b = min(b_geo, b_arith, b_avg)
    if best_b == b_geo:
        ensemble_oof[h] = geo_oof
        ensemble_test[h] = geo_test
        method = f"GEOMETRIC (w={dict(zip(models, np.round(w_soft,3)))})"
    elif best_b == b_arith:
        ensemble_oof[h] = arith_oof
        ensemble_test[h] = arith_test
        method = f"SCIPY-OPT (w={dict(zip(models, np.round(w_opt,3)))})"
    else:
        ensemble_oof[h] = avg_oof
        ensemble_test[h] = avg_test
        method = "SIMPLE AVERAGE"

    auc_e = roc_auc_score(y_f, ensemble_oof[h]) if len(np.unique(y_f)) > 1 else 0.5
    print(f"  {h}h: geo={b_geo:.5f} opt={b_arith:.5f} avg={b_avg:.5f} "
          f"-> {method}")
    print(f"       AUC={auc_e:.4f} Brier={best_b:.5f}")

# ============================================================
# PHASE 8 — POST-PROCESSING + SUBMISSION
# ============================================================
print("\n" + "=" * 70)
print("PHASE 8: Post-Processing + Submissions")
print("=" * 70)

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

# Rank blend
hybrid_test = {}
for h in TIME_HORIZONS:
    cal_prob = ensemble_test[h]
    rank_prob = rankdata(cal_prob) / len(cal_prob)
    cw, rw = BLEND_CONFIG[h]
    hybrid_test[h] = cw * cal_prob + rw * rank_prob

# Generate all variants
print("\nSubmission variants:")
sub_A = create_submission(hybrid_test, *CLIP_VARIANTS['A'], suffix='_A')
sub_B = create_submission(hybrid_test, *CLIP_VARIANTS['B'], suffix='_B')
sub_C = create_submission(hybrid_test, *CLIP_VARIANTS['C'], suffix='_C')

# ============================================================
# PHASE 9 — CREATE SUBMISSION_08 (best single + blend with old)
# ============================================================
print("\n" + "=" * 70)
print("PHASE 9: Creating submission_08.csv (THE ONE TO SUBMIT)")
print("=" * 70)

# submission_08 = pure v10 output with moderate clip [0.01, 0.99]
sub_08 = create_submission(hybrid_test, 0.008, 0.992, suffix='_08')

# Also create a blend version for safety
old = pd.read_csv('d:/WiDS/submission_07.csv')
cols = ['prob_12h', 'prob_24h', 'prob_48h', 'prob_72h']
blended = old[['event_id']].copy()
for c in cols:
    blended[c] = 0.70 * sub_A[c] + 0.30 * old[c]
for c in cols:
    blended[c] = blended[c].clip(0.008, 0.992)
for i in range(len(blended)):
    for j in range(1, len(cols)):
        if blended.at[blended.index[i], cols[j]] < blended.at[blended.index[i], cols[j-1]] + MONOTONICITY_INCREMENT:
            blended.at[blended.index[i], cols[j]] = blended.at[blended.index[i], cols[j-1]] + MONOTONICITY_INCREMENT
blended.to_csv('d:/WiDS/submission_08_safe.csv', index=False)
print(f"  submission_08_safe.csv: 70/30 blend with LB-proven old")

# ============================================================
# FINAL REPORT
# ============================================================
print("\n" + "=" * 70)
print("FINAL REPORT")
print("=" * 70)

oof_briers = {}
oof_aucs = {}
for h in TIME_HORIZONS:
    y_h, _, mask = ipcw_data[h]
    y_f = y_h[mask]
    p_f = np.clip(ensemble_oof[h], 0.001, 0.999)
    auc = roc_auc_score(y_f, p_f) if len(np.unique(y_f)) > 1 else 0.5
    brier = brier_score_loss(y_f, p_f)
    oof_aucs[h] = auc
    oof_briers[h] = brier
    print(f"  [{h}h] AUC={auc:.4f} Brier={brier:.5f} Hybrid={0.3*auc+0.7*(1-brier):.5f}")

wb = 0.3 * oof_briers[24] + 0.4 * oof_briers[48] + 0.3 * oof_briers[72]
avg_auc = np.mean([oof_aucs[h] for h in [24, 48, 72]])
est = 0.3 * avg_auc + 0.7 * (1 - wb)

print(f"\n  ESTIMATED HYBRID SCORE: {est:.5f}")
print(f"  Weighted Brier: {wb:.5f}")
print(f"  Avg AUC: {avg_auc:.5f}")

print(f"\n  Features: {len(ALL_FEATURES)}")
print(f"  Models: {MODEL_TYPES}")
print(f"  Seeds: {len(SEEDS)}")
print(f"  Runtime: {(time.time()-start_time)/60:.1f} min")

print(f"\n  FILES CREATED:")
print(f"    submission_08.csv      <- SUBMIT THIS")
print(f"    submission_08_safe.csv <- backup (70/30 blend)")
print(f"    submission_A/B/C.csv   <- variants")

print("\n" + "=" * 70)
print("Pipeline v10 FINAL COMPLETE. GO FOR 0.975! 🔥🔥🔥")
print("=" * 70)
