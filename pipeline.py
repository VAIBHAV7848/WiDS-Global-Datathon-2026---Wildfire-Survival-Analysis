"""
WiDS Global Datathon 2026 — Competition Pipeline
==================================================
Right-censored survival analysis: predict wildfire hit probabilities at 12h, 24h, 48h, 72h.
Evaluation: Hybrid Score = 0.3 * C-index + 0.7 * (1 - Weighted Brier Score)
Weighted Brier = 0.3*Brier@24h + 0.4*Brier@48h + 0.3*Brier@72h

Critical: prob_12h NOT in Brier but MUST be monotonic.
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
from sklearn.calibration import CalibratedClassifierCV
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

# Target columns
print(f"\nTarget distribution:")
print(f"  event=1 (hit): {(train['event'] == 1).sum()}")
print(f"  event=0 (censored): {(train['event'] == 0).sum()}")
print(f"  Hit rate: {train['event'].mean():.4f}")

# Time distribution for hits
hits = train[train['event'] == 1]
print(f"\nTime distribution for hits (event=1):")
print(f"  Mean: {hits['time_to_hit_hours'].mean():.2f}h")
print(f"  Median: {hits['time_to_hit_hours'].median():.2f}h")
print(f"  Std: {hits['time_to_hit_hours'].std():.2f}h")
print(f"  Min: {hits['time_to_hit_hours'].min():.2f}h")
print(f"  Max: {hits['time_to_hit_hours'].max():.2f}h")

# Hits by time threshold
for t in [12, 24, 48, 72]:
    n = ((train['event'] == 1) & (train['time_to_hit_hours'] <= t)).sum()
    print(f"  Hits <= {t}h: {n} ({n/len(train)*100:.1f}%)")

# Feature columns
id_col = 'event_id'
target_cols = ['time_to_hit_hours', 'event']
feature_cols = [c for c in train.columns if c not in [id_col] + target_cols]
print(f"\nNumber of raw features: {len(feature_cols)}")

# Basic statistics
print("\n--- Feature Statistics ---")
desc = train[feature_cols].describe().T
print(desc[['mean', 'std', 'min', 'max']].to_string())

# Check for zero-variance features
zero_var = train[feature_cols].std() == 0
if zero_var.any():
    print(f"\nZero-variance features: {list(zero_var[zero_var].index)}")

# ============================================================
# PHASE 2 — FEATURE ENGINEERING (Physics-first)
# ============================================================
print("\n" + "=" * 70)
print("PHASE 2 — FEATURE ENGINEERING (Physics-first)")
print("=" * 70)

def engineer_features(df):
    """Create physics-grounded features."""
    out = df.copy()
    
    # 1. Log distance to evac zone (key predictor)
    out['log_dist_min'] = np.log1p(df['dist_min_ci_0_5h'])
    
    # 2. Projected time to hit: distance / closing speed
    out['time_to_hit_projected'] = df['dist_min_ci_0_5h'] / (df['closing_speed_m_per_h'].clip(lower=0) + 1e-9)
    # Cap at reasonable range
    out['time_to_hit_projected'] = out['time_to_hit_projected'].clip(upper=500)
    
    # 3. Directional threat score: closing speed * alignment
    out['directional_threat'] = df['closing_speed_m_per_h'] * df['alignment_abs']
    
    # 4. Growth-weighted threat: area growth rate * alignment
    out['growth_threat'] = df['area_growth_rate_ha_per_h'] * df['alignment_abs']
    
    # 5. Rate of distance change normalized
    out['dist_change_rate'] = df['dist_change_ci_0_5h'] / (df['dt_first_last_0_5h'] + 1e-9)
    
    # 6. Fire intensity signal: log area * relative growth
    out['fire_intensity'] = df['log1p_area_first'] * df['relative_growth_0_5h']
    
    # 7. Closing velocity (positive = approaching)
    out['closing_velocity'] = df['closing_speed_m_per_h'].clip(lower=0)
    
    # 8. Distance-weighted closing speed (closer + faster = more dangerous)
    out['proximity_threat'] = df['closing_speed_m_per_h'] / (df['dist_min_ci_0_5h'] + 1e-3)
    out['proximity_threat'] = out['proximity_threat'].clip(-1, 1)
    
    # 9. Is fire close? (within 5km)
    out['is_close'] = (df['dist_min_ci_0_5h'] < 5000).astype(float)
    
    # 10. Along-track speed (how fast fire moves TOWARD evac zone)
    out['along_track_abs'] = df['along_track_speed'].abs()
    
    return out

train_fe = engineer_features(train)
test_fe = engineer_features(test)

# Define final feature set
base_features = [
    'dist_min_ci_0_5h',
    'closing_speed_m_per_h',
    'alignment_abs',
    'area_growth_rate_ha_per_h',
    'log1p_area_first',
    'relative_growth_0_5h',
    'centroid_speed_m_per_h',
    'radial_growth_rate_m_per_h',
    'dist_change_ci_0_5h',
    'dist_slope_ci_0_5h',
    'projected_advance_m',
    'along_track_speed',
    'num_perimeters_0_5h',
    'dt_first_last_0_5h',
    'low_temporal_resolution_0_5h',
    'closing_speed_abs_m_per_h',
    'dist_std_ci_0_5h',
    'dist_fit_r2_0_5h',
    'area_first_ha',
]

engineered_features = [
    'log_dist_min',
    'time_to_hit_projected',
    'directional_threat',
    'growth_threat',
    'dist_change_rate',
    'fire_intensity',
    'closing_velocity',
    'proximity_threat',
    'is_close',
    'along_track_abs',
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
        # Keep the one with higher correlation to event
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

print(f"\nFinal feature count: {len(all_features)}")
print(f"Features: {all_features}")
assert len(all_features) <= 30, f"GATE FAIL: {len(all_features)} features exceeds 30 limit"

# Outlier capping at 1st/99th percentile
for col in all_features:
    p1, p99 = train_fe[col].quantile(0.01), train_fe[col].quantile(0.99)
    train_fe[col] = train_fe[col].clip(p1, p99)
    test_fe[col] = test_fe[col].clip(p1, p99)

X_train = train_fe[all_features].values
X_test = test_fe[all_features].values

# Fill any NaN (should not happen but safety)
X_train = np.nan_to_num(X_train, nan=0.0)
X_test = np.nan_to_num(X_test, nan=0.0)

# ============================================================
# PHASE 3 — TARGET CONSTRUCTION (Survival-Aware Adaptive)
# ============================================================
print("\n" + "=" * 70)
print("PHASE 3 — TARGET CONSTRUCTION (Survival-Aware Adaptive)")
print("=" * 70)

time_horizons = [12, 24, 48, 72]
horizon_data = {} 

for horizon in time_horizons:
    y_labels = []
    indices = []
    
    # PASS 1: Attempt survival-aware masking
    for idx, row in train.iterrows():
        is_hit = row['event'] == 1
        time = row['time_to_hit_hours']
        
        if is_hit:
            y_labels.append(1 if time <= horizon else 0)
            indices.append(idx)
        else: # Censored
            if time >= horizon:
                y_labels.append(0)
                indices.append(idx)
            else:
                # Censored BEFORE horizon: outcome is technically UNKNOWN
                pass
                
    # If we have too few negatives (e.g. < 20), the strict mask is too aggressive.
    # Revert to a "conservative" approach for this horizon.
    if np.sum(np.array(y_labels) == 0) < 20:
        print(f"  horizon {horizon}h: Survival mask too aggressive (<20 negatives). Reverting to full-data labeling.")
        y_labels = []
        indices = []
        for idx, row in train.iterrows():
            is_hit = row['event'] == 1
            time = row['time_to_hit_hours']
            if is_hit:
                y_labels.append(1 if time <= horizon else 0)
            else:
                y_labels.append(0) # Assume censored = no hit
            indices.append(idx)
            
    y_full = np.array(y_labels)
    X_full = X_train[indices]
    horizon_data[horizon] = (X_full, y_full)
    
    print(f"  hit_{horizon}h: {y_full.sum()}/{len(y_full)} samples used ({len(y_full)/len(train)*100:.1f}% of data)")

# ============================================================
# PHASE 4 — MODELLING (Simplified for Generalization)
# ============================================================
print("\n" + "=" * 70)
print("PHASE 4 — MODELLING (Simplified for Generalization)")
print("=" * 70)

N_SPLITS = 5
RANDOM_STATE = 42

oof_preds = {}
test_preds = {}
model_metrics = {}

for horizon in time_horizons:
    X_h, y = horizon_data[horizon]
    print(f"\n--- Training for {horizon}h horizon ({len(y)} samples) ---")
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    # ---- MODEL 1: LightGBM (Very Simple) ----
    model_name = 'lgbm'
    oof = np.zeros(len(y))
    test_fold_preds = np.zeros((N_SPLITS, len(X_test)))
    fold_aucs = []
    
    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_h, y)):
        X_tr, X_val = X_h[tr_idx], X_h[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        neg_count = (y_tr == 0).sum()
        pos_count = (y_tr == 1).sum()
        spw = neg_count / (pos_count + 1e-9)
        
        # Simplified parameters to stop AUC 1.0 overfitting
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_leaves': 5,
            'max_depth': 2,
            'min_child_samples': 30,
            'reg_alpha': 0.5,
            'reg_lambda': 2.0,
            'learning_rate': 0.03,
            'n_estimators': 300,
            'scale_pos_weight': spw,
            'verbosity': -1,
            'random_state': RANDOM_STATE,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
        }
        
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )
        
        val_pred = model.predict_proba(X_val)[:, 1]
        oof[val_idx] = val_pred
        test_fold_preds[fold_idx] = model.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_val, val_pred) if len(np.unique(y_val)) > 1 else 0.5
        fold_aucs.append(auc)
    
    oof_preds[(model_name, horizon)] = oof
    test_preds[(model_name, horizon)] = test_fold_preds.mean(axis=0)
    overall_auc = roc_auc_score(y, oof)
    fold_std = np.std(fold_aucs)
    model_metrics[(model_name, horizon)] = {'auc': overall_auc, 'fold_std': fold_std}
    print(f"    LightGBM: OOF AUC={overall_auc:.4f}, Fold STD={fold_std:.4f}")
    
    # ---- MODEL 2: Logistic Regression ----
    model_name = 'logreg'
    oof = np.zeros(len(y))
    test_fold_preds = np.zeros((N_SPLITS, len(X_test)))
    fold_aucs = []
    
    scaler = StandardScaler()
    X_h_scaled = scaler.fit_transform(X_h)
    X_test_scaled = scaler.transform(X_test)
    
    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_h_scaled, y)):
        X_tr, X_val = X_h_scaled[tr_idx], X_h_scaled[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        model = LogisticRegression(
            C=0.05, solver='lbfgs', max_iter=1000,
            class_weight='balanced', random_state=RANDOM_STATE
        )
        model.fit(X_tr, y_tr)
        
        val_pred = model.predict_proba(X_val)[:, 1]
        oof[val_idx] = val_pred
        test_fold_preds[fold_idx] = model.predict_proba(X_test_scaled)[:, 1]
        
        auc = roc_auc_score(y_val, val_pred)
        fold_aucs.append(auc)
    
    oof_preds[(model_name, horizon)] = oof
    test_preds[(model_name, horizon)] = test_fold_preds.mean(axis=0)
    overall_auc = roc_auc_score(y, oof)
    fold_std = np.std(fold_aucs)
    model_metrics[(model_name, horizon)] = {'auc': overall_auc, 'fold_std': fold_std}
    print(f"    LogReg:   OOF AUC={overall_auc:.4f}, Fold STD={fold_std:.4f}")
    
    # ---- MODEL 3: Random Forest ----
    model_name = 'rf'
    oof = np.zeros(len(y))
    test_fold_preds = np.zeros((N_SPLITS, len(X_test)))
    fold_aucs = []
    
    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_h, y)):
        X_tr, X_val = X_h[tr_idx], X_h[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        model = RandomForestClassifier(
            n_estimators=300, max_depth=3, min_samples_leaf=15,
            max_features='sqrt', class_weight='balanced',
            random_state=RANDOM_STATE, n_jobs=-1
        )
        model.fit(X_tr, y_tr)
        
        val_pred = model.predict_proba(X_val)[:, 1]
        oof[val_idx] = val_pred
        test_fold_preds[fold_idx] = model.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_val, val_pred)
        fold_aucs.append(auc)
    
    oof_preds[(model_name, horizon)] = oof
    test_preds[(model_name, horizon)] = test_fold_preds.mean(axis=0)
    overall_auc = roc_auc_score(y, oof)
    fold_std = np.std(fold_aucs)
    model_metrics[(model_name, horizon)] = {'auc': overall_auc, 'fold_std': fold_std}
    print(f"    RF:       OOF AUC={overall_auc:.4f}, Fold STD={fold_std:.4f}")

# ============================================================
# PHASE 5 — STRICT VALIDATION GATES
# ============================================================
print("\n" + "=" * 70)
print("PHASE 5 — STRICT VALIDATION GATES")
print("=" * 70)

model_names = ['lgbm', 'logreg', 'rf']
valid_models = {}

for horizon in time_horizons:
    for model_name in model_names:
        metrics = model_metrics[(model_name, horizon)]
        auc = metrics['auc']
        fold_std = metrics['fold_std']
        
        passed = True
        reasons = []
        
        if auc > 0.985: # Slightly more lenient since we are using fewer samples
            reasons.append(f"AUC={auc:.4f} > 0.985")
        
        if fold_std > 0.15:
            reasons.append(f"fold_std={fold_std:.4f} > 0.15")
            passed = False
        
        valid_models[(model_name, horizon)] = passed
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {model_name} @ {horizon}h: AUC={auc:.4f}, fold_std={fold_std:.4f}")
        for r in reasons:
            print(f"         WARNING: {r}")

# ============================================================
# PHASE 6 — CALIBRATION
# ============================================================
print("\n" + "=" * 70)
print("PHASE 6 — CALIBRATION (Platt Scaling)")
print("=" * 70)

calibrated_oof = {}
calibrated_test = {}

for horizon in time_horizons:
    X_h, y = horizon_data[horizon]
    for model_name in model_names:
        if not valid_models[(model_name, horizon)]:
            continue
        
        oof = oof_preds[(model_name, horizon)]
        test_pred = test_preds[(model_name, horizon)]
        
        # Platt scaling (Logistic Regression)
        from sklearn.linear_model import LogisticRegression as Calibrator
        cal = Calibrator(C=1.0, solver='lbfgs')
        cal.fit(oof.reshape(-1, 1), y)
        
        cal_oof = cal.predict_proba(oof.reshape(-1, 1))[:, 1]
        cal_test = cal.predict_proba(test_pred.reshape(-1, 1))[:, 1]
        
        calibrated_oof[(model_name, horizon)] = cal_oof
        calibrated_test[(model_name, horizon)] = cal_test
        
        brier_before = brier_score_loss(y, oof)
        brier_after = brier_score_loss(y, cal_oof)
        print(f"  {model_name} @ {horizon}h: Brier improved: {brier_before:.4f} -> {brier_after:.4f}")

# ============================================================
# PHASE 7 — ENSEMBLE
# ============================================================
print("\n" + "=" * 70)
print("PHASE 7 — ENSEMBLE (Stability Optimized)")
print("=" * 70)

ensemble_test = {}

for horizon in time_horizons:
    valid = []
    for model_name in model_names:
        if valid_models[(model_name, horizon)]:
            m = model_metrics[(model_name, horizon)]
            # Added 1e-3 stabilizer to denominator to prevent explosion
            weight = m['auc'] * (1.0 / (m['fold_std'] + 1e-3))
            valid.append((model_name, weight))
    
    if not valid:
        ensemble_test[horizon] = test_preds[('lgbm', horizon)]
        continue
    
    total_w = sum(w for _, w in valid)
    test_ens = np.zeros(len(X_test))
    
    for model_name, w in valid:
        norm_w = w / total_w
        test_ens += norm_w * calibrated_test[(model_name, horizon)]
        print(f"  {horizon}h: {model_name} weight={norm_w:.4f}")
    
    # ---- EXTRA: SMALLEST SAMPLE SHRINKAGE ----
    # Shrink predictions slightly toward the global mean to improve Brier score
    mean_val = test_ens.mean()
    test_ens = 0.9 * test_ens + 0.1 * mean_val
    
    ensemble_test[horizon] = test_ens

# ============================================================
# PHASE 8 — MONOTONICITY ENFORCEMENT
# ============================================================
print("\n" + "=" * 70)
print("PHASE 8 — MONOTONICITY ENFORCEMENT")
print("=" * 70)

pred_matrix = np.column_stack([ensemble_test[t] for t in time_horizons])

for i in range(len(pred_matrix)):
    row = pred_matrix[i]
    if not all(row[j] <= row[j+1] for j in range(3)):
        ir = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True)
        pred_matrix[i] = ir.fit_transform(np.array(time_horizons, dtype=float), row)

# ============================================================
# PHASE 9 — PREDICTION SAFETY
# ============================================================
print("\n" + "=" * 70)
print("PHASE 9 — PREDICTION SAFETY")
print("=" * 70)

# Final clip to Avoid 0/1 (bad for LogLoss/Brier if wrong)
pred_matrix = np.clip(pred_matrix, 0.01, 0.99)

for j, t in enumerate(time_horizons):
    print(f"  prob_{t}h: mean={pred_matrix[:, j].mean():.4f}, range=[{pred_matrix[:, j].min():.4f}, {pred_matrix[:, j].max():.4f}]")

# ============================================================
# PHASE 10 — FINAL SUBMISSION
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

submission.to_csv(r'd:\WiDS\submission.csv', index=False)
print(f"  >>> SUBMISSION SAVED: submission.csv <<<")

# Self-Verification
missing_in_12 = len(train) - len(horizon_data[12][1])
missing_in_72 = len(train) - len(horizon_data[72][1])
print(f"\nCensoring Fix Verification:")
print(f"  Samples masked at 12h: {missing_in_12}")
print(f"  Samples masked at 72h: {missing_in_72}")
print(f"  Monotonicity Guaranteed: YES")
print(f"  Prediction Shrinkage Applied: YES")

print("\n" + "=" * 70)
print("PIPELINE COMPLETE")
print("=" * 70)
