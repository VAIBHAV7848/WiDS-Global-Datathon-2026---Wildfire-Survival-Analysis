"""
Pipeline v25 STACK — Multi-Model Stacked + Calibrated + Blended Ensemble
=========================================================================
Target: improve public LB from 0.97284 to >= 0.98.
Metric: 0.5 * C-index + 0.5 * (1 - WeightedBrier)
  where WeightedBrier = 0.25 * (B_12h + B_24h + B_48h + B_72h)

Architecture:
  1) 80/10/10 holdout split (train_stack / calib_isotonic / calib_blend)
  2) Feature engineering: 8 domain-driven interaction + cyclical features
  3) 6 base classifiers x 4 horizons, 5-fold StratifiedKFold
  4) LogisticRegression stacking meta-model per horizon
  5) Isotonic calibration (fit on calib_isotonic)
  6) Blend with v20_MEGA prior (tune weight on calib_blend)
  7) Monotonicity enforcement and output
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from lifelines.utils import concordance_index

print("=" * 70)
print("  PIPELINE v25 STACK — Multi-Model Stacked + Calibrated Ensemble")
print("=" * 70)

# ─── Install missing packages if needed ────────────────────────────────
try:
    import lightgbm as lgb
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm"])
    import lightgbm as lgb

try:
    import xgboost as xgb
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
    import xgboost as xgb

try:
    import catboost as cb
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost"])
    import catboost as cb

# ═══════════════════════════════════════════════
# HYBRID METRIC DEFINITION
# ═══════════════════════════════════════════════
def weighted_brier(y_true_12, y_true_24, y_true_48, y_true_72,
                   pred_12, pred_24, pred_48, pred_72):
    """Weighted Brier score across 4 horizons (equal weights 0.25)."""
    b12 = np.mean((y_true_12 - pred_12) ** 2)
    b24 = np.mean((y_true_24 - pred_24) ** 2)
    b48 = np.mean((y_true_48 - pred_48) ** 2)
    b72 = np.mean((y_true_72 - pred_72) ** 2)
    return 0.25 * b12 + 0.25 * b24 + 0.25 * b48 + 0.25 * b72

def hybrid_score(y_time, y_event,
                 y_true_12, y_true_24, y_true_48, y_true_72,
                 pred_12, pred_24, pred_48, pred_72):
    """
    Competition hybrid metric.
    C-index uses -pred_72 so higher pred_72 => higher risk => lower survival.
    """
    c_idx = concordance_index(y_time, -pred_72, y_event)
    wb = weighted_brier(y_true_12, y_true_24, y_true_48, y_true_72,
                        pred_12, pred_24, pred_48, pred_72)
    return 0.5 * c_idx + 0.5 * (1 - wb)

# ═══════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
v20 = pd.read_csv("submission_v20_MEGA.csv")

print(f"\nTrain: {train.shape}, Test: {test.shape}")
print(f"Event rate: {train.event.mean():.3f}")

HORIZONS = [12, 24, 48, 72]
HORIZON_COLS = [f"prob_{h}h" for h in HORIZONS]

# ═══════════════════════════════════════════════
# STEP 1: INITIAL HOLDOUTS (80/10/10)
# ═══════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 1: Creating holdout splits (80/10/10)")
print("=" * 70)

# First split: 80% train_stack, 20% remaining
train_stack, temp = train_test_split(
    train, test_size=0.20, stratify=train['event'], random_state=42
)

# Second split: 50/50 of the 20% => 10% calib_isotonic, 10% calib_blend
calib_isotonic, calib_blend = train_test_split(
    temp, test_size=0.50, stratify=temp['event'], random_state=42
)

# Reset indices
train_stack = train_stack.reset_index(drop=True)
calib_isotonic = calib_isotonic.reset_index(drop=True)
calib_blend = calib_blend.reset_index(drop=True)

print(f"  train_stack:    {len(train_stack)} rows (event={train_stack.event.mean():.3f})")
print(f"  calib_isotonic: {len(calib_isotonic)} rows (event={calib_isotonic.event.mean():.3f})")
print(f"  calib_blend:    {len(calib_blend)} rows (event={calib_blend.event.mean():.3f})")

# ═══════════════════════════════════════════════
# STEP 2: TARGETS
# ═══════════════════════════════════════════════
def make_targets(df):
    """Create binary targets for each horizon."""
    return {
        12: (df['time_to_hit_hours'] <= 12).astype(int).values,
        24: (df['time_to_hit_hours'] <= 24).astype(int).values,
        48: (df['time_to_hit_hours'] <= 48).astype(int).values,
        72: df['event'].values,  # raw 0/1
    }

y_train_stack = make_targets(train_stack)
y_calib_iso = make_targets(calib_isotonic)
y_calib_blend = make_targets(calib_blend)

for h in HORIZONS:
    print(f"  y_{h}h: train_stack={y_train_stack[h].sum()}/{len(train_stack)} positive")

# ═══════════════════════════════════════════════
# STEP 3: FEATURE ENGINEERING
# ═══════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: Feature Engineering")
print("=" * 70)

def engineer_features(df):
    """
    Add domain-driven interaction and cyclical features.
    Applied identically to all splits + test.
    """
    df = df.copy()
    
    # Interaction features
    cs = df['closing_speed_m_per_h'].fillna(0)
    cs_abs = df['closing_speed_abs_m_per_h'].fillna(0)
    align_abs = df['alignment_abs'].fillna(0)
    rgr = df['radial_growth_rate_m_per_h'].fillna(0)
    area_first = df['area_first_ha'].fillna(0)
    cspeed = df['centroid_speed_m_per_h'].fillna(0)
    dist_min = df['dist_min_ci_0_5h'].fillna(0)
    
    # feat1: closing speed * alignment (directional threat)
    df['feat1'] = cs * align_abs
    
    # feat2: radial growth * (1 - alignment) (lateral spread risk)
    df['feat2'] = rgr * (1 - align_abs)
    
    # feat3: fire size * speed (momentum proxy)
    df['feat3'] = np.log1p(area_first) * cspeed
    
    # feat4: distance * closing direction (approach rate vs distance)
    df['feat4'] = dist_min * (cs / (cs_abs + 1e-6))
    
    # Cyclical encoding of month and hour
    df['feat5'] = np.sin(2 * np.pi * df['event_start_month'] / 12)
    df['feat6'] = np.cos(2 * np.pi * df['event_start_month'] / 12)
    df['feat7'] = np.sin(2 * np.pi * df['event_start_hour'] / 24)
    df['feat8'] = np.cos(2 * np.pi * df['event_start_hour'] / 24)
    
    # Additional physics features
    # dist_km for easier interpretation
    df['dist_km'] = dist_min / 1000.0
    df['log_dist'] = np.log1p(df['dist_km'])
    
    # Fire activity indicator
    df['is_active'] = ((rgr > 0) | (df['area_growth_rate_ha_per_h'].fillna(0) > 0)).astype(int)
    
    # Close fire indicator (< 5km)
    df['is_near'] = (dist_min < 5000).astype(int)
    
    # Interaction: near * active
    df['near_active'] = df['is_near'] * df['is_active']
    
    # Interaction: alignment * distance (directional proximity)
    df['align_dist'] = align_abs * df['dist_km']
    
    # dt observation quality
    df['dt_first_last'] = df['dt_first_last_0_5h']
    
    return df

train_stack_fe = engineer_features(train_stack)
calib_isotonic_fe = engineer_features(calib_isotonic)
calib_blend_fe = engineer_features(calib_blend)
test_fe = engineer_features(test)

# Define feature columns (exclude target and ID columns)
exclude_cols = ['event_id', 'time_to_hit_hours', 'event']
feature_cols = [c for c in train_stack_fe.columns if c not in exclude_cols]

print(f"  Total features: {len(feature_cols)}")

# Prepare feature matrices
X_train_stack = train_stack_fe[feature_cols].values.astype(np.float64)
X_calib_iso = calib_isotonic_fe[feature_cols].values.astype(np.float64)
X_calib_blend = calib_blend_fe[feature_cols].values.astype(np.float64)
X_test = test_fe[feature_cols].values.astype(np.float64)

# Replace NaN/inf
for arr in [X_train_stack, X_calib_iso, X_calib_blend, X_test]:
    arr[np.isnan(arr)] = 0
    arr[np.isinf(arr)] = 0

# ═══════════════════════════════════════════════
# STEP 4: BASE MODELS — 5-fold StratifiedKFold
# ═══════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4: Training Base Models (6 models x 4 horizons)")
print("=" * 70)

N_FOLDS = 5
RANDOM_STATE = 42

# Model definitions
def get_models():
    """Return dict of model name -> model factory."""
    return {
        'lgbm': lambda seed: None,         # handled separately
        'xgb': lambda seed: None,           # handled separately
        'catboost': lambda seed: None,      # handled separately
        'rf': lambda seed: RandomForestClassifier(
            n_estimators=300, max_depth=4, min_samples_leaf=5,
            random_state=seed, n_jobs=-1
        ),
        'et': lambda seed: ExtraTreesClassifier(
            n_estimators=300, max_depth=4, min_samples_leaf=5,
            random_state=seed, n_jobs=-1
        ),
        'lr': lambda seed: LogisticRegression(
            C=1.0, max_iter=1000, random_state=seed
        ),
    }

MODEL_NAMES = ['lgbm', 'xgb', 'catboost', 'rf', 'et', 'lr']

# Storage for all predictions
# oof_preds[horizon][model_name] = array of OOF predictions on train_stack
# test_preds[horizon][model_name] = averaged test predictions
# calib_iso_preds[horizon][model_name] = averaged calib_isotonic predictions
# calib_blend_preds[horizon][model_name] = averaged calib_blend predictions
oof_preds = {h: {} for h in HORIZONS}
test_preds_all = {h: {} for h in HORIZONS}
calib_iso_preds = {h: {} for h in HORIZONS}
calib_blend_preds = {h: {} for h in HORIZONS}

for h in HORIZONS:
    y = y_train_stack[h]
    print(f"\n{'-' * 60}")
    print(f"  Horizon {h}h: {y.sum()}/{len(y)} positive ({y.mean():.3f})")
    print(f"{'-' * 60}")
    
    for model_name in MODEL_NAMES:
        oof = np.zeros(len(train_stack))
        test_accum = np.zeros(len(test))
        iso_accum = np.zeros(len(calib_isotonic))
        blend_accum = np.zeros(len(calib_blend))
        fold_scores = []
        
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        
        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_train_stack, y)):
            X_tr, X_va = X_train_stack[tr_idx], X_train_stack[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]
            
            if model_name == 'lgbm':
                # LightGBM with early stopping
                dtrain = lgb.Dataset(X_tr, y_tr)
                dval = lgb.Dataset(X_va, y_va, reference=dtrain)
                params = {
                    "objective": "binary",
                    "metric": "binary_logloss",
                    "verbosity": -1,
                    "seed": RANDOM_STATE + fold_idx,
                    "num_leaves": 8,
                    "max_depth": 3,
                    "min_child_samples": 5,
                    "reg_alpha": 3.0,
                    "reg_lambda": 5.0,
                    "colsample_bytree": 0.6,
                    "subsample": 0.7,
                    "learning_rate": 0.02,
                    "n_jobs": -1,
                }
                model = lgb.train(
                    params, dtrain, num_boost_round=2000,
                    valid_sets=[dval],
                    callbacks=[lgb.early_stopping(50, verbose=False)]
                )
                p_va = model.predict(X_va)
                p_te = model.predict(X_test)
                p_iso = model.predict(X_calib_iso)
                p_blend = model.predict(X_calib_blend)
                
            elif model_name == 'xgb':
                # XGBoost
                dtrain = xgb.DMatrix(X_tr, label=y_tr)
                dval = xgb.DMatrix(X_va, label=y_va)
                params = {
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                    "seed": RANDOM_STATE + fold_idx,
                    "max_depth": 3,
                    "min_child_weight": 5,
                    "reg_alpha": 3.0,
                    "reg_lambda": 5.0,
                    "colsample_bytree": 0.6,
                    "subsample": 0.7,
                    "learning_rate": 0.02,
                    "tree_method": "hist",
                }
                model = xgb.train(
                    params, dtrain, num_boost_round=2000,
                    evals=[(dval, 'val')],
                    early_stopping_rounds=50,
                    verbose_eval=False
                )
                p_va = model.predict(xgb.DMatrix(X_va))
                p_te = model.predict(xgb.DMatrix(X_test))
                p_iso = model.predict(xgb.DMatrix(X_calib_iso))
                p_blend = model.predict(xgb.DMatrix(X_calib_blend))
                
            elif model_name == 'catboost':
                # CatBoost
                model = cb.CatBoostClassifier(
                    loss_function='Logloss',
                    iterations=2000,
                    depth=3,
                    learning_rate=0.02,
                    l2_leaf_reg=5.0,
                    random_seed=RANDOM_STATE + fold_idx,
                    verbose=0,
                    early_stopping_rounds=50,
                )
                model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0)
                p_va = model.predict_proba(X_va)[:, 1]
                p_te = model.predict_proba(X_test)[:, 1]
                p_iso = model.predict_proba(X_calib_iso)[:, 1]
                p_blend = model.predict_proba(X_calib_blend)[:, 1]
                
            elif model_name in ('rf', 'et'):
                # Sklearn models
                factories = get_models()
                model = factories[model_name](RANDOM_STATE + fold_idx)
                model.fit(X_tr, y_tr)
                p_va = model.predict_proba(X_va)[:, 1]
                p_te = model.predict_proba(X_test)[:, 1]
                p_iso = model.predict_proba(X_calib_iso)[:, 1]
                p_blend = model.predict_proba(X_calib_blend)[:, 1]
                
            elif model_name == 'lr':
                # Logistic Regression (scale features)
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_tr_scaled = scaler.fit_transform(X_tr)
                X_va_scaled = scaler.transform(X_va)
                X_te_scaled = scaler.transform(X_test)
                X_iso_scaled = scaler.transform(X_calib_iso)
                X_blend_scaled = scaler.transform(X_calib_blend)
                
                model = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE + fold_idx)
                model.fit(X_tr_scaled, y_tr)
                p_va = model.predict_proba(X_va_scaled)[:, 1]
                p_te = model.predict_proba(X_te_scaled)[:, 1]
                p_iso = model.predict_proba(X_iso_scaled)[:, 1]
                p_blend = model.predict_proba(X_blend_scaled)[:, 1]
            
            oof[va_idx] = p_va
            test_accum += p_te / N_FOLDS
            iso_accum += p_iso / N_FOLDS
            blend_accum += p_blend / N_FOLDS
            fold_scores.append(brier_score_loss(y_va, p_va))
        
        oof_preds[h][model_name] = oof
        test_preds_all[h][model_name] = test_accum
        calib_iso_preds[h][model_name] = iso_accum
        calib_blend_preds[h][model_name] = blend_accum
        
        bs_mean = np.mean(fold_scores)
        bs_std = np.std(fold_scores)
        print(f"    {model_name:10s}: Brier={bs_mean:.5f} ± {bs_std:.4f}")

# ═══════════════════════════════════════════════
# STEP 5: STACKING (leakage-free)
# ═══════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5: Stacking Meta-Model (LogisticRegression)")
print("=" * 70)

stacked_oof = {}     # train_stack OOF stacked predictions
stacked_test = {}    # test stacked predictions
stacked_iso = {}     # calib_isotonic stacked predictions
stacked_blend = {}   # calib_blend stacked predictions

for h in HORIZONS:
    # Build meta-features from OOF predictions
    meta_train = np.column_stack([oof_preds[h][m] for m in MODEL_NAMES])
    meta_test = np.column_stack([test_preds_all[h][m] for m in MODEL_NAMES])
    meta_iso = np.column_stack([calib_iso_preds[h][m] for m in MODEL_NAMES])
    meta_blend = np.column_stack([calib_blend_preds[h][m] for m in MODEL_NAMES])
    
    y = y_train_stack[h]
    
    # Fit meta-model on OOF predictions
    meta_model = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    meta_model.fit(meta_train, y)
    
    stacked_oof[h] = meta_model.predict_proba(meta_train)[:, 1]
    stacked_test[h] = meta_model.predict_proba(meta_test)[:, 1]
    stacked_iso[h] = meta_model.predict_proba(meta_iso)[:, 1]
    stacked_blend[h] = meta_model.predict_proba(meta_blend)[:, 1]
    
    bs = brier_score_loss(y, stacked_oof[h])
    print(f"  {h}h: Stacked OOF Brier = {bs:.5f}")

# ═══════════════════════════════════════════════
# STEP 6: CALIBRATION (on calib_isotonic ONLY)
# ═══════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 6: Isotonic Calibration")
print("=" * 70)

calibrated_test = {}
calibrated_blend = {}

for h in HORIZONS:
    y_iso = y_calib_iso[h]
    
    # Fit isotonic regression on calib_isotonic stacked probs
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(stacked_iso[h], y_iso)
    
    # Apply to test and calib_blend
    calibrated_test[h] = ir.transform(stacked_test[h])
    calibrated_blend[h] = ir.transform(stacked_blend[h])
    
    bs_before = brier_score_loss(y_iso, stacked_iso[h])
    bs_after = brier_score_loss(y_iso, ir.transform(stacked_iso[h]))
    print(f"  {h}h: Brier before={bs_before:.5f}, after={bs_after:.5f}")

# ═══════════════════════════════════════════════
# STEP 7: BLEND WITH v20 PRIOR
# ═══════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 7: Blend Weight Tuning with v20 MEGA")
print("=" * 70)

# Map v20 predictions to calib_blend rows by event_id
v20_indexed = v20.set_index('event_id')

# Get v20 probs for calib_blend events
v20_blend_probs = {}
for h in HORIZONS:
    col = f"prob_{h}h"
    v20_blend_probs[h] = calib_blend['event_id'].map(v20_indexed[col]).values

# Get v20 probs for test (already aligned)
v20_test_probs = {}
for h in HORIZONS:
    col = f"prob_{h}h"
    v20_test_probs[h] = v20[col].values

# Check if v20_blend mappings have NaN (calib_blend events might not be in v20)
has_v20_blend = True
for h in HORIZONS:
    if np.any(np.isnan(v20_blend_probs[h])):
        print(f"  WARNING: v20 doesn't have all calib_blend events for {h}h — filling with calibrated preds")
        mask = np.isnan(v20_blend_probs[h])
        v20_blend_probs[h][mask] = calibrated_blend[h][mask]

# Grid search blend weight
WEIGHTS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]

y_time_blend = calib_blend['time_to_hit_hours'].values
y_event_blend = calib_blend['event'].values

best_w = 0.50
best_score = -1

print(f"\n  {'Weight':>8s}  {'C-index':>8s}  {'WBrier':>8s}  {'Hybrid':>8s}")
print(f"  {'------':>8s}  {'-------':>8s}  {'------':>8s}  {'------':>8s}")

for w in WEIGHTS:
    blend_12 = w * calibrated_blend[12] + (1 - w) * v20_blend_probs[12]
    blend_24 = w * calibrated_blend[24] + (1 - w) * v20_blend_probs[24]
    blend_48 = w * calibrated_blend[48] + (1 - w) * v20_blend_probs[48]
    blend_72 = w * calibrated_blend[72] + (1 - w) * v20_blend_probs[72]
    
    # Enforce monotonicity on blend
    probs = np.stack([blend_12, blend_24, blend_48, blend_72], axis=1)
    probs = np.maximum.accumulate(probs, axis=1)
    blend_12, blend_24, blend_48, blend_72 = probs[:, 0], probs[:, 1], probs[:, 2], probs[:, 3]
    
    try:
        score = hybrid_score(
            y_time_blend, y_event_blend,
            y_calib_blend[12], y_calib_blend[24], y_calib_blend[48], y_calib_blend[72],
            blend_12, blend_24, blend_48, blend_72
        )
        
        c_idx = concordance_index(y_time_blend, -blend_72, y_event_blend)
        wb = weighted_brier(
            y_calib_blend[12], y_calib_blend[24], y_calib_blend[48], y_calib_blend[72],
            blend_12, blend_24, blend_48, blend_72
        )
        
        print(f"  {w:8.2f}  {c_idx:8.5f}  {wb:8.5f}  {score:8.5f}")
        
        if score > best_score:
            best_score = score
            best_w = w
    except Exception as e:
        print(f"  {w:8.2f}  ERROR: {e}")

print(f"\n  -> Best blend weight: {best_w:.2f} (hybrid score: {best_score:.5f})")

# ═══════════════════════════════════════════════
# STEP 8: FINAL TEST PREDICTIONS
# ═══════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 8: Final Predictions")
print("=" * 70)

final_12 = best_w * calibrated_test[12] + (1 - best_w) * v20_test_probs[12]
final_24 = best_w * calibrated_test[24] + (1 - best_w) * v20_test_probs[24]
final_48 = best_w * calibrated_test[48] + (1 - best_w) * v20_test_probs[48]
final_72 = best_w * calibrated_test[72] + (1 - best_w) * v20_test_probs[72]

# ═══════════════════════════════════════════════
# STEP 9: MONOTONICITY ENFORCEMENT
# ═══════════════════════════════════════════════
probs = np.stack([final_12, final_24, final_48, final_72], axis=1)
probs = np.maximum.accumulate(probs, axis=1)
final_12, final_24, final_48, final_72 = probs[:, 0], probs[:, 1], probs[:, 2], probs[:, 3]

# Clip to valid range
final_12 = np.clip(final_12, 0.001, 0.999)
final_24 = np.clip(final_24, 0.001, 0.999)
final_48 = np.clip(final_48, 0.001, 0.999)
final_72 = np.clip(final_72, 0.001, 0.999)

# ═══════════════════════════════════════════════
# STEP 10: OUTPUT
# ═══════════════════════════════════════════════
submission = pd.DataFrame({
    'event_id': test['event_id'],
    'prob_12h': np.round(final_12, 6),
    'prob_24h': np.round(final_24, 6),
    'prob_48h': np.round(final_48, 6),
    'prob_72h': np.round(final_72, 6),
})

submission.to_csv("submission_final.csv", index=False)

# Validation checks
n_rows = len(submission)
has_nulls = submission.isnull().any().any()
all_valid = (submission[HORIZON_COLS] >= 0).all().all() and (submission[HORIZON_COLS] <= 1).all().all()

# Check monotonicity
mono_violations = 0
for i in range(len(submission)):
    for j in range(3):
        if submission.iloc[i][HORIZON_COLS[j]] > submission.iloc[i][HORIZON_COLS[j+1]] + 1e-9:
            mono_violations += 1

print(f"\n  Output: submission_final.csv")
print(f"  Rows: {n_rows}")
print(f"  Nulls: {has_nulls}")
print(f"  All values in [0,1]: {all_valid}")
print(f"  Monotonicity violations: {mono_violations}")

# ═══════════════════════════════════════════════
# VALIDATION REPORT
# ═══════════════════════════════════════════════
print("\n" + "=" * 70)
print("VALIDATION REPORT")
print("=" * 70)

# Compute CV hybrid score on train_stack OOF
try:
    oof_12 = stacked_oof[12]
    oof_24 = stacked_oof[24]
    oof_48 = stacked_oof[48]
    oof_72 = stacked_oof[72]
    
    cv_score = hybrid_score(
        train_stack['time_to_hit_hours'].values, train_stack['event'].values,
        y_train_stack[12], y_train_stack[24], y_train_stack[48], y_train_stack[72],
        oof_12, oof_24, oof_48, oof_72
    )
    
    cv_cidx = concordance_index(
        train_stack['time_to_hit_hours'].values, -oof_72, train_stack['event'].values
    )
    
    cv_brier = weighted_brier(
        y_train_stack[12], y_train_stack[24], y_train_stack[48], y_train_stack[72],
        oof_12, oof_24, oof_48, oof_72
    )
    
    print(f"\n  Pipeline CV Hybrid Score: {cv_score:.5f}")
    print(f"  Pipeline CV C-index:     {cv_cidx:.5f}")
    print(f"  Pipeline CV WBrier:      {cv_brier:.5f}")
except Exception as e:
    print(f"  Could not compute CV score: {e}")

print(f"\n  Best blend weight:       {best_w:.2f}")
print(f"  Blend tuning score:      {best_score:.5f}")
print(f"  Baseline LB (v20):       0.97284")

# Statistics comparison
print(f"\n  Final prediction statistics:")
for col in HORIZON_COLS:
    vals = submission[col]
    print(f"    {col}: mean={vals.mean():.4f}, min={vals.min():.4f}, max={vals.max():.4f}")

# Compare v20 vs final
print(f"\n  v20_MEGA statistics:")
for col in HORIZON_COLS:
    vals = v20[col]
    print(f"    {col}: mean={vals.mean():.4f}, min={vals.min():.4f}, max={vals.max():.4f}")

print("\n" + "=" * 70)
print("DONE. submission_final.csv ready for Kaggle upload.")
print("=" * 70)

# ═══════════════════════════════════════════════
# ALSO: Create v20-dominant blend variants for safety
# ═══════════════════════════════════════════════
for w_name, w_val in [("w30", 0.30), ("w50", 0.50), ("w70", 0.70)]:
    f12 = w_val * calibrated_test[12] + (1 - w_val) * v20_test_probs[12]
    f24 = w_val * calibrated_test[24] + (1 - w_val) * v20_test_probs[24]
    f48 = w_val * calibrated_test[48] + (1 - w_val) * v20_test_probs[48]
    f72 = w_val * calibrated_test[72] + (1 - w_val) * v20_test_probs[72]
    
    pp = np.stack([f12, f24, f48, f72], axis=1)
    pp = np.maximum.accumulate(pp, axis=1)
    pp = np.clip(pp, 0.001, 0.999)
    
    sub_var = pd.DataFrame({
        'event_id': test['event_id'],
        'prob_12h': np.round(pp[:, 0], 6),
        'prob_24h': np.round(pp[:, 1], 6),
        'prob_48h': np.round(pp[:, 2], 6),
        'prob_72h': np.round(pp[:, 3], 6),
    })
    sub_var.to_csv(f"submission_v25_{w_name}.csv", index=False)
    print(f"  Saved submission_v25_{w_name}.csv")
