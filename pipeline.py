"""
WiDS Global Datathon 2026 — Competition Pipeline v3.0
==================================================
Right-censored survival analysis: predict wildfire hit probabilities at 12h, 24h, 48h, 72h.
Evaluation: Hybrid Score = 0.3 * C-index + 0.7 * (1 - Weighted Brier Score)
Weighted Brier = 0.3*Brier@24h + 0.4*Brier@48h + 0.3*Brier@72h

Critical: prob_12h NOT in Brier but MUST be monotonic.

v3.0 Fixes:
  1.  Progressive survival masking (no naive fallback)
  2.  Ranking features for C-index boost
  3.  Feature stability selection (drop zero-importance)
  4.  LightGBM: num_leaves=10, max_depth=3, min_child_samples=25
  5.  Ensemble OOF validation with AUC + Brier
  6.  Reduced shrinkage (0.95/0.05)
  7.  Time-aware scaling [0.9, 1.0, 1.1, 1.2]
  8.  Power transform (pred ** 0.95)
  9.  Flat prediction fix (min increase 0.01)
  10. Final monotonicity enforcement
  11. Safety clip [0.02, 0.98]
  12. Comprehensive validation report
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
import lightgbm as lgb

# ============================================================
# PHASE 1 — DATA LOADING & UNDERSTANDING
# ============================================================
print("=" * 70)
print("PHASE 1 — DATA LOADING & UNDERSTANDING")
print("=" * 70)

train = pd.read_csv(r'd:\WiDS\train.csv')
test = pd.read_csv(r'd:\WiDS\test.csv')
sample_sub = pd.read_csv(r'd:\WiDS\sample_submission.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Sample submission shape: {sample_sub.shape}")

print(f"\nTarget distribution:")
print(f"  event=1 (hit): {(train['event'] == 1).sum()}")
print(f"  event=0 (censored): {(train['event'] == 0).sum()}")
print(f"  Hit rate: {train['event'].mean():.4f}")

hits = train[train['event'] == 1]
print(f"\nTime distribution for hits (event=1):")
print(f"  Mean: {hits['time_to_hit_hours'].mean():.2f}h")
print(f"  Median: {hits['time_to_hit_hours'].median():.2f}h")
print(f"  Max: {hits['time_to_hit_hours'].max():.2f}h")

for t in [12, 24, 48, 72]:
    n = ((train['event'] == 1) & (train['time_to_hit_hours'] <= t)).sum()
    print(f"  Hits <= {t}h: {n} ({n/len(train)*100:.1f}%)")

censored = train[train['event'] == 0]
print(f"\nCensoring time distribution (event=0):")
print(f"  Mean: {censored['time_to_hit_hours'].mean():.2f}h, Max: {censored['time_to_hit_hours'].max():.2f}h")
for t in [12, 24, 48, 72]:
    n = (censored['time_to_hit_hours'] >= t).sum()
    print(f"  Observed >= {t}h: {n}/{len(censored)}")

id_col = 'event_id'
target_cols = ['time_to_hit_hours', 'event']
feature_cols = [c for c in train.columns if c not in [id_col] + target_cols]
print(f"\nRaw features: {len(feature_cols)}")

# ============================================================
# PHASE 2 — FEATURE ENGINEERING (Physics-first + Ranking)
# ============================================================
print("\n" + "=" * 70)
print("PHASE 2 — FEATURE ENGINEERING (Physics-first + Ranking)")
print("=" * 70)

def engineer_features(df):
    """Create physics-grounded + ranking features."""
    out = df.copy()
    
    # Physics features
    out['log_dist_min'] = np.log1p(df['dist_min_ci_0_5h'])
    
    out['time_to_hit_projected'] = df['dist_min_ci_0_5h'] / (df['closing_speed_m_per_h'].clip(lower=0) + 1e-9)
    out['time_to_hit_projected'] = out['time_to_hit_projected'].clip(upper=500)
    
    out['directional_threat'] = df['closing_speed_m_per_h'] * df['alignment_abs']
    out['growth_threat'] = df['area_growth_rate_ha_per_h'] * df['alignment_abs']
    out['dist_change_rate'] = df['dist_change_ci_0_5h'] / (df['dt_first_last_0_5h'] + 1e-9)
    out['fire_intensity'] = df['log1p_area_first'] * df['relative_growth_0_5h']
    out['closing_velocity'] = df['closing_speed_m_per_h'].clip(lower=0)
    
    out['proximity_threat'] = df['closing_speed_m_per_h'] / (df['dist_min_ci_0_5h'] + 1e-3)
    out['proximity_threat'] = out['proximity_threat'].clip(-1, 1)
    
    out['is_close'] = (df['dist_min_ci_0_5h'] < 5000).astype(float)
    out['along_track_abs'] = df['along_track_speed'].abs()
    
    return out

train_fe = engineer_features(train)
test_fe = engineer_features(test)

base_features = [
    'dist_min_ci_0_5h', 'closing_speed_m_per_h', 'alignment_abs',
    'area_growth_rate_ha_per_h', 'log1p_area_first', 'relative_growth_0_5h',
    'centroid_speed_m_per_h', 'radial_growth_rate_m_per_h',
    'dist_change_ci_0_5h', 'dist_slope_ci_0_5h', 'projected_advance_m',
    'along_track_speed', 'num_perimeters_0_5h', 'dt_first_last_0_5h',
    'low_temporal_resolution_0_5h', 'closing_speed_abs_m_per_h',
    'dist_std_ci_0_5h', 'dist_fit_r2_0_5h', 'area_first_ha',
]

engineered_features = [
    'log_dist_min', 'time_to_hit_projected', 'directional_threat',
    'growth_threat', 'dist_change_rate', 'fire_intensity',
    'closing_velocity', 'proximity_threat', 'is_close', 'along_track_abs',
]

all_features = base_features + engineered_features

# Drop near-zero variance
train_std = train_fe[all_features].std()
low_var = train_std[train_std < 1e-10].index.tolist()
if low_var:
    print(f"Dropping near-zero variance features: {low_var}")
    all_features = [f for f in all_features if f not in low_var]

# Drop highly correlated features (> 0.97)
corr_matrix = train_fe[all_features].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = set()
for col in upper.columns:
    high_corr = upper.index[upper[col] > 0.97].tolist()
    if high_corr:
        for hc in high_corr:
            corr_event_col = abs(train_fe[col].corr(train_fe['event']))
            corr_event_hc = abs(train_fe[hc].corr(train_fe['event']))
            if corr_event_col >= corr_event_hc:
                to_drop.add(hc)
            else:
                to_drop.add(col)

if to_drop:
    print(f"Dropping highly correlated features: {to_drop}")
    all_features = [f for f in all_features if f not in to_drop]

print(f"\nFeature count after correlation filter: {len(all_features)}")
assert len(all_features) <= 30, f"GATE FAIL: {len(all_features)} features exceeds 30 limit"

# Outlier capping at 1st/99th percentile
for col in all_features:
    p1, p99 = train_fe[col].quantile(0.01), train_fe[col].quantile(0.99)
    train_fe[col] = train_fe[col].clip(p1, p99)
    test_fe[col] = test_fe[col].clip(p1, p99)

X_train_full = train_fe[all_features].values
X_test = test_fe[all_features].values
X_train_full = np.nan_to_num(X_train_full, nan=0.0)
X_test = np.nan_to_num(X_test, nan=0.0)

# ============================================================
# PHASE 2.5 — FEATURE STABILITY SELECTION (FIX #5)
# ============================================================
print("\n" + "=" * 70)
print("PHASE 2.5 — FEATURE STABILITY SELECTION")
print("=" * 70)

# Quick LGBM scan on 12h (most samples) to find zero-importance features
y_scan = ((train['event'] == 1) & (train['time_to_hit_hours'] <= 12)).astype(int).values
skf_scan = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_importances = []

for tr_idx, val_idx in skf_scan.split(X_train_full, y_scan):
    m = lgb.LGBMClassifier(num_leaves=10, max_depth=3, n_estimators=100,
                            learning_rate=0.05, verbosity=-1, random_state=42)
    m.fit(X_train_full[tr_idx], y_scan[tr_idx])
    fold_importances.append(m.feature_importances_)

imp_matrix = np.array(fold_importances)
imp_mean = imp_matrix.mean(axis=0)
imp_std = imp_matrix.std(axis=0)

# Drop features with zero mean importance across all folds
zero_imp_mask = imp_mean == 0
zero_imp_features = [all_features[i] for i in range(len(all_features)) if zero_imp_mask[i]]

if zero_imp_features:
    print(f"Dropping zero-importance features: {zero_imp_features}")
    stable_features = [f for f in all_features if f not in zero_imp_features]
else:
    stable_features = all_features.copy()
    print("All features have non-zero importance.")

# Report stability
print(f"\nFeature stability report:")
for i, f in enumerate(all_features):
    cv = imp_std[i] / (imp_mean[i] + 1e-9)
    tag = "STABLE" if cv < 1.0 else "VARIABLE"
    kept = "KEPT" if f in stable_features else "DROPPED"
    print(f"  {f:30s}: mean_imp={imp_mean[i]:8.1f}, CV={cv:.2f} [{tag}] [{kept}]")

all_features = stable_features
X_train_full = train_fe[all_features].values
X_test = test_fe[all_features].values
X_train_full = np.nan_to_num(X_train_full, nan=0.0)
X_test = np.nan_to_num(X_test, nan=0.0)

print(f"\nFinal stable feature count: {len(all_features)}")

# ============================================================
# PHASE 3 — TARGET CONSTRUCTION (Progressive Survival Masking)
# FIX #1 + #2: No naive fallback. Progressive relaxation only.
# ============================================================
print("\n" + "=" * 70)
print("PHASE 3 — TARGET CONSTRUCTION (Progressive Survival Masking)")
print("=" * 70)

time_horizons = [12, 24, 48, 72]
horizon_data = {}       # (X_masked, y_masked, indices_masked)
MIN_NEGATIVES = 5

for horizon in time_horizons:
    # Try progressively relaxed observation thresholds
    relax_factors = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50]
    selected = None
    
    for relax in relax_factors:
        obs_threshold = horizon * relax
        y_labels = []
        indices = []
        
        for idx, row in train.iterrows():
            is_hit = row['event'] == 1
            t = row['time_to_hit_hours']
            
            if is_hit:
                y_labels.append(1 if t <= horizon else 0)
                indices.append(idx)
            else:  # Censored
                if t >= obs_threshold:
                    # Observed long enough — label as negative
                    y_labels.append(0)
                    indices.append(idx)
                # else: UNKNOWN, exclude
        
        n_neg = sum(1 for y in y_labels if y == 0)
        n_pos = sum(1 for y in y_labels if y == 1)
        
        if n_neg >= MIN_NEGATIVES and n_pos >= MIN_NEGATIVES:
            selected = (y_labels, indices, obs_threshold, relax)
            break
    
    if selected is None:
        # This should not happen with the fine-grained relaxation above
        print(f"  CRITICAL WARNING [{horizon}h]: Cannot build valid training set!")
        # Use the last attempt (most relaxed)
        selected = (y_labels, indices, obs_threshold, relax)
    
    y_labels, indices, obs_threshold, relax = selected
    y_arr = np.array(y_labels)
    X_arr = X_train_full[indices]
    horizon_data[horizon] = (X_arr, y_arr, indices)
    
    mask_type = "STRICT" if relax == 1.0 else f"RELAXED ({relax:.0%}, obs>={obs_threshold:.0f}h)"
    n_pos = y_arr.sum()
    n_neg = len(y_arr) - n_pos
    print(f"  {horizon}h: {n_pos} pos + {n_neg} neg = {len(y_arr)} total [{mask_type}]")

# ============================================================
# PHASE 4 — MODELLING (FIX #4: Updated hyperparameters)
# ============================================================
print("\n" + "=" * 70)
print("PHASE 4 — MODELLING")
print("=" * 70)

N_SPLITS = 5
RANDOM_STATE = 42
model_names = ['lgbm', 'logreg', 'rf']

oof_preds = {}
test_preds = {}
model_metrics = {}

for horizon in time_horizons:
    X_h, y, mask_indices = horizon_data[horizon]
    print(f"\n--- {horizon}h horizon ({len(y)} samples, {y.sum()} pos) ---")
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    # ---- MODEL 1: LightGBM (FIX #4) ----
    mn = 'lgbm'
    oof = np.zeros(len(y))
    test_fp = np.zeros((N_SPLITS, len(X_test)))
    fold_aucs = []
    
    for fi, (tr_idx, val_idx) in enumerate(skf.split(X_h, y)):
        X_tr, X_val = X_h[tr_idx], X_h[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        neg_c = (y_tr == 0).sum()
        pos_c = (y_tr == 1).sum()
        spw = max(neg_c / (pos_c + 1e-9), 0.1)
        
        params = {
            'objective': 'binary', 'metric': 'binary_logloss',
            'num_leaves': 10, 'max_depth': 3,           # FIX #4
            'min_child_samples': 25,                     # FIX #4
            'reg_alpha': 0.5, 'reg_lambda': 2.0,
            'learning_rate': 0.03, 'n_estimators': 300,
            'scale_pos_weight': spw,
            'verbosity': -1, 'random_state': RANDOM_STATE,
            'subsample': 0.7, 'colsample_bytree': 0.7,
        }
        
        mdl = lgb.LGBMClassifier(**params)
        mdl.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)])
        
        vp = mdl.predict_proba(X_val)[:, 1]
        oof[val_idx] = vp
        test_fp[fi] = mdl.predict_proba(X_test)[:, 1]
        fold_aucs.append(roc_auc_score(y_val, vp) if len(np.unique(y_val)) > 1 else 0.5)
    
    oof_preds[(mn, horizon)] = oof
    test_preds[(mn, horizon)] = test_fp.mean(axis=0)
    oa = roc_auc_score(y, oof) if len(np.unique(y)) > 1 else 0.5
    fs = np.std(fold_aucs)
    model_metrics[(mn, horizon)] = {'auc': oa, 'fold_std': fs}
    print(f"    LightGBM: AUC={oa:.4f}, STD={fs:.4f}")
    
    # ---- MODEL 2: Logistic Regression ----
    mn = 'logreg'
    oof = np.zeros(len(y))
    test_fp = np.zeros((N_SPLITS, len(X_test)))
    fold_aucs = []
    
    scaler = StandardScaler()
    X_h_sc = scaler.fit_transform(X_h)
    X_test_sc = scaler.transform(X_test)
    
    for fi, (tr_idx, val_idx) in enumerate(skf.split(X_h_sc, y)):
        X_tr, X_val = X_h_sc[tr_idx], X_h_sc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        mdl = LogisticRegression(C=0.05, solver='lbfgs', max_iter=1000,
                                  class_weight='balanced', random_state=RANDOM_STATE)
        mdl.fit(X_tr, y_tr)
        
        vp = mdl.predict_proba(X_val)[:, 1]
        oof[val_idx] = vp
        test_fp[fi] = mdl.predict_proba(X_test_sc)[:, 1]
        fold_aucs.append(roc_auc_score(y_val, vp) if len(np.unique(y_val)) > 1 else 0.5)
    
    oof_preds[(mn, horizon)] = oof
    test_preds[(mn, horizon)] = test_fp.mean(axis=0)
    oa = roc_auc_score(y, oof) if len(np.unique(y)) > 1 else 0.5
    fs = np.std(fold_aucs)
    model_metrics[(mn, horizon)] = {'auc': oa, 'fold_std': fs}
    print(f"    LogReg:   AUC={oa:.4f}, STD={fs:.4f}")
    
    # ---- MODEL 3: Random Forest ----
    mn = 'rf'
    oof = np.zeros(len(y))
    test_fp = np.zeros((N_SPLITS, len(X_test)))
    fold_aucs = []
    
    for fi, (tr_idx, val_idx) in enumerate(skf.split(X_h, y)):
        X_tr, X_val = X_h[tr_idx], X_h[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        mdl = RandomForestClassifier(
            n_estimators=300, max_depth=3, min_samples_leaf=15,
            max_features='sqrt', class_weight='balanced',
            random_state=RANDOM_STATE, n_jobs=-1)
        mdl.fit(X_tr, y_tr)
        
        vp = mdl.predict_proba(X_val)[:, 1]
        oof[val_idx] = vp
        test_fp[fi] = mdl.predict_proba(X_test)[:, 1]
        fold_aucs.append(roc_auc_score(y_val, vp) if len(np.unique(y_val)) > 1 else 0.5)
    
    oof_preds[(mn, horizon)] = oof
    test_preds[(mn, horizon)] = test_fp.mean(axis=0)
    oa = roc_auc_score(y, oof) if len(np.unique(y)) > 1 else 0.5
    fs = np.std(fold_aucs)
    model_metrics[(mn, horizon)] = {'auc': oa, 'fold_std': fs}
    print(f"    RF:       AUC={oa:.4f}, STD={fs:.4f}")

# ============================================================
# PHASE 5 — VALIDATION GATES
# ============================================================
print("\n" + "=" * 70)
print("PHASE 5 — VALIDATION GATES")
print("=" * 70)

valid_models = {}

for horizon in time_horizons:
    for mn in model_names:
        m = model_metrics[(mn, horizon)]
        passed = True
        reasons = []
        
        if m['auc'] > 0.985:
            reasons.append(f"AUC={m['auc']:.4f}>0.985 (overfit warning)")
        if m['fold_std'] > 0.15:
            reasons.append(f"STD={m['fold_std']:.4f}>0.15 (unstable)")
            passed = False
        
        valid_models[(mn, horizon)] = passed
        tag = "PASS" if passed else "FAIL"
        print(f"  [{tag}] {mn} @ {horizon}h: AUC={m['auc']:.4f}, STD={m['fold_std']:.4f}")
        for r in reasons:
            print(f"         WARNING: {r}")

# ============================================================
# PHASE 6 — CALIBRATION (Platt Scaling)
# ============================================================
print("\n" + "=" * 70)
print("PHASE 6 — CALIBRATION")
print("=" * 70)

calibrated_oof = {}
calibrated_test = {}

for horizon in time_horizons:
    _, y, _ = horizon_data[horizon]
    for mn in model_names:
        if not valid_models[(mn, horizon)]:
            continue
        
        oof = oof_preds[(mn, horizon)]
        tp = test_preds[(mn, horizon)]
        
        cal = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
        cal.fit(oof.reshape(-1, 1), y)
        
        cal_oof = cal.predict_proba(oof.reshape(-1, 1))[:, 1]
        cal_test = cal.predict_proba(tp.reshape(-1, 1))[:, 1]
        
        calibrated_oof[(mn, horizon)] = cal_oof
        calibrated_test[(mn, horizon)] = cal_test
        
        b_before = brier_score_loss(y, oof)
        b_after = brier_score_loss(y, cal_oof)
        print(f"  {mn} @ {horizon}h: Brier {b_before:.4f} -> {b_after:.4f}")

# ============================================================
# PHASE 7 — ENSEMBLE + OOF VALIDATION (FIX #3 + #7)
# ============================================================
print("\n" + "=" * 70)
print("PHASE 7 — ENSEMBLE + OOF VALIDATION")
print("=" * 70)

ensemble_test = {}
ensemble_oof = {}

for horizon in time_horizons:
    _, y, _ = horizon_data[horizon]
    
    valid = []
    for mn in model_names:
        if valid_models[(mn, horizon)] and (mn, horizon) in calibrated_test:
            m = model_metrics[(mn, horizon)]
            w = m['auc'] * (1.0 / (m['fold_std'] + 1e-3))  # stabilized weight
            valid.append((mn, w))
    
    if not valid:
        ensemble_test[horizon] = test_preds[('lgbm', horizon)]
        ensemble_oof[horizon] = oof_preds[('lgbm', horizon)]
        print(f"  {horizon}h: No valid models, using raw LGBM")
        continue
    
    total_w = sum(w for _, w in valid)
    t_ens = np.zeros(len(X_test))
    o_ens = np.zeros(len(y))
    
    for mn, w in valid:
        nw = w / total_w
        t_ens += nw * calibrated_test[(mn, horizon)]
        o_ens += nw * calibrated_oof[(mn, horizon)]
        print(f"  {horizon}h: {mn} weight={nw:.4f}")
    
    # FIX #7: REDUCED shrinkage (0.95/0.05 instead of 0.9/0.1)
    test_mean = t_ens.mean()
    t_ens = 0.95 * t_ens + 0.05 * test_mean
    
    ensemble_test[horizon] = t_ens
    ensemble_oof[horizon] = o_ens

# FIX #3: Ensemble OOF Validation
print("\n  --- Ensemble OOF Validation ---")
for horizon in time_horizons:
    _, y, _ = horizon_data[horizon]
    oof_e = ensemble_oof[horizon]
    if len(np.unique(y)) > 1:
        ens_auc = roc_auc_score(y, oof_e)
        ens_brier = brier_score_loss(y, oof_e)
        oof_mean = oof_e.mean()
        test_mean = ensemble_test[horizon].mean()
        diff = abs(oof_mean - test_mean)
        print(f"  {horizon}h: Ens_AUC={ens_auc:.4f}, Ens_Brier={ens_brier:.4f}, "
              f"OOF_mean={oof_mean:.4f}, TEST_mean={test_mean:.4f}, diff={diff:.4f}")
    else:
        print(f"  {horizon}h: Single class — cannot compute AUC")

# ============================================================
# PHASE 8 — MONOTONICITY ENFORCEMENT (Clean)
# ============================================================
print("\n" + "=" * 70)
print("PHASE 8 — MONOTONICITY ENFORCEMENT")
print("=" * 70)

pred_matrix = np.column_stack([ensemble_test[t] for t in time_horizons])

# Count violations before fix
violations_before = 0
for i in range(len(pred_matrix)):
    for j in range(3):
        if pred_matrix[i, j] > pred_matrix[i, j+1]:
            violations_before += 1

print(f"  Monotonicity violations before fix: {violations_before}")

# Isotonic regression — the only principled monotonicity enforcement
for i in range(len(pred_matrix)):
    row = pred_matrix[i]
    if not all(row[j] <= row[j+1] for j in range(3)):
        ir = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True)
        pred_matrix[i] = ir.fit_transform(np.array(time_horizons, dtype=float), row)

violations_after = sum(
    1 for i in range(len(pred_matrix))
    for j in range(3)
    if pred_matrix[i, j] > pred_matrix[i, j+1]
)
print(f"  Monotonicity violations after fix: {violations_after}")

# ============================================================
# PHASE 9 — SAFETY (FIX #12)
# ============================================================
print("\n" + "=" * 70)
print("PHASE 9 — SAFETY")
print("=" * 70)

# FIX #12: Clip to [0.02, 0.98]
pred_matrix = np.clip(pred_matrix, 0.02, 0.98)

# Re-enforce monotonicity after clipping (clipping can break the min-increase)
for i in range(len(pred_matrix)):
    for j in range(1, 4):
        if pred_matrix[i, j] < pred_matrix[i, j-1]:
            pred_matrix[i, j] = pred_matrix[i, j-1]

# Final verification
violations_final = 0
for i in range(len(pred_matrix)):
    for j in range(3):
        if pred_matrix[i, j] > pred_matrix[i, j+1] + 1e-9:
            violations_final += 1

has_nan = np.isnan(pred_matrix).any()
has_inf = np.isinf(pred_matrix).any()

print(f"  Final monotonicity violations: {violations_final}")
print(f"  NaN present: {has_nan}")
print(f"  Inf present: {has_inf}")

for j, t in enumerate(time_horizons):
    col = pred_matrix[:, j]
    print(f"  prob_{t}h: mean={col.mean():.4f}, std={col.std():.4f}, "
          f"min={col.min():.4f}, max={col.max():.4f}")

# ============================================================
# PHASE 10 — FINAL SUBMISSION + COMPREHENSIVE REPORT
# ============================================================
print("\n" + "=" * 70)
print("PHASE 10 — FINAL SUBMISSION")
print("=" * 70)

submission = pd.DataFrame({
    'event_id': test['event_id'].values,
    'prob_12h': pred_matrix[:, 0],
    'prob_24h': pred_matrix[:, 1],
    'prob_48h': pred_matrix[:, 2],
    'prob_72h': pred_matrix[:, 3],
})

# Verification
checks = {}
checks['rows'] = len(submission) == 95
checks['cols'] = list(submission.columns) == ['event_id', 'prob_12h', 'prob_24h', 'prob_48h', 'prob_72h']
checks['ids'] = list(submission['event_id']) == list(sample_sub['event_id'])
checks['no_nan'] = not submission.isnull().any().any()
checks['no_inf'] = not np.isinf(submission[['prob_12h', 'prob_24h', 'prob_48h', 'prob_72h']].values).any()

prob_cols = ['prob_12h', 'prob_24h', 'prob_48h', 'prob_72h']
checks['range'] = submission[prob_cols].min().min() >= 0.02 - 1e-9 and submission[prob_cols].max().max() <= 0.98 + 1e-9

mono_ok = True
for _, row in submission.iterrows():
    if not (row['prob_12h'] <= row['prob_24h'] + 1e-9 and
            row['prob_24h'] <= row['prob_48h'] + 1e-9 and
            row['prob_48h'] <= row['prob_72h'] + 1e-9):
        mono_ok = False
        break
checks['monotonicity'] = mono_ok

for name, ok in checks.items():
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

all_passed = all(checks.values())

if all_passed:
    submission.to_csv(r'd:\WiDS\submission.csv', index=False)
    print(f"\n  >>> SUBMISSION SAVED: d:\\WiDS\\submission.csv <<<")
else:
    print(f"\n  >>> SUBMISSION NOT SAVED — FIX ERRORS <<<")

# ============================================================
# COMPREHENSIVE VALIDATION REPORT
# ============================================================
print("\n" + "=" * 70)
print("COMPREHENSIVE VALIDATION REPORT")
print("=" * 70)

print("\n--- CV Metrics Per Horizon ---")
for horizon in time_horizons:
    print(f"\n  {horizon}h:")
    for mn in model_names:
        m = model_metrics[(mn, horizon)]
        v = valid_models[(mn, horizon)]
        print(f"    {mn:8s}: AUC={m['auc']:.4f}, STD={m['fold_std']:.4f}, valid={v}")

print("\n--- Ensemble OOF Summary ---")
for horizon in time_horizons:
    _, y, _ = horizon_data[horizon]
    oof_e = ensemble_oof[horizon]
    if len(np.unique(y)) > 1:
        print(f"  {horizon}h: AUC={roc_auc_score(y, oof_e):.4f}, Brier={brier_score_loss(y, oof_e):.4f}")

print("\n--- OOF vs TEST Mean ---")
for j, horizon in enumerate(time_horizons):
    _, y, _ = horizon_data[horizon]
    oof_mean = ensemble_oof[horizon].mean()
    test_mean = pred_matrix[:, j].mean()
    diff = abs(oof_mean - test_mean)
    print(f"  {horizon}h: OOF={oof_mean:.4f}, TEST={test_mean:.4f}, diff={diff:.4f}")

print("\n--- Prediction Distribution ---")
for j, t in enumerate(time_horizons):
    col = pred_matrix[:, j]
    print(f"  prob_{t}h: mean={col.mean():.4f}, std={col.std():.4f}")

print("\n--- Fixes Applied (v4.0 Clean) ---")
print("  [1] Progressive survival masking (no naive fallback)")
print("  [2] Feature stability selection (drop zero-importance)")
print("  [3] LightGBM: num_leaves=10, max_depth=3, min_child_samples=25")
print("  [4] Ensemble OOF validation (AUC + Brier)")
print("  [5] Reduced shrinkage (0.95/0.05)")
print("  [6] Isotonic monotonicity enforcement")
print("  [7] Safety clip [0.02, 0.98]")
print("  REMOVED: ranking features, time scaling, power transform, flat fix")

print("\n" + "=" * 70)
print("PIPELINE v4.0 COMPLETE")
print("=" * 70)

