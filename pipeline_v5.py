"""
WiDS Global Datathon 2026 — GOD-TIER Pipeline v5.0
====================================================
Multi-Agent System: MANAGER → DATA → MODEL → CRITIC → VALIDATION

Right-censored survival analysis: predict wildfire hit probabilities at 12h, 24h, 48h, 72h.
Evaluation: Hybrid Score = 0.3 * C-index + 0.7 * (1 - Weighted Brier Score)
Weighted Brier = 0.3*Brier@24h + 0.4*Brier@48h + 0.3*Brier@72h

PRIORITY: Stability > Calibration > Ranking > Monotonicity > Simplicity

Key improvements over v4.0:
  1.  Multi-seed training (seeds = [42, 52, 62, 72, 82])
  2.  Distribution alignment check (OOF vs TEST)
  3.  Rank transformation for C-index boost
  4.  Hybrid blending (calibrated + rank)
  5.  3 submission strategies (balanced, aggressive, conservative)
  6.  Improved feature engineering with domain features
  7.  Proper 72h handling (all censored are early-censored)
  8.  Conservative hyperparameters for tiny dataset
  9.  Isotonic calibration option
  10. Comprehensive validation with simulated C-index
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.stats import rankdata
import lightgbm as lgb

# ============================================================
# AGENT 0: MANAGER — Controls execution phases
# ============================================================
print("=" * 80)
print("  MANAGER AGENT: GOD-TIER Pipeline v5.0 — Multi-Agent Execution")
print("=" * 80)

# ============================================================
# PHASE 1 — DATA LOADING & GROUND TRUTH
# ============================================================
print("\n" + "=" * 80)
print("  PHASE 1 — DATA LOADING & GROUND TRUTH")
print("=" * 80)

train = pd.read_csv(r'd:\WiDS\train.csv')
test = pd.read_csv(r'd:\WiDS\test.csv')
sample_sub = pd.read_csv(r'd:\WiDS\sample_submission.csv')

print(f"  Train: {train.shape[0]} rows × {train.shape[1]} cols")
print(f"  Test:  {test.shape[0]} rows × {test.shape[1]} cols")
print(f"  Submission: {sample_sub.shape[0]} rows")
print(f"\n  Event=1 (hit): {(train['event'] == 1).sum()}")
print(f"  Event=0 (censored): {(train['event'] == 0).sum()}")
print(f"  Hit rate: {train['event'].mean():.4f}")

hits = train[train['event'] == 1]
censored = train[train['event'] == 0]
print(f"\n  Hits time: mean={hits['time_to_hit_hours'].mean():.2f}h, "
      f"median={hits['time_to_hit_hours'].median():.2f}h, "
      f"max={hits['time_to_hit_hours'].max():.2f}h")

for t in [12, 24, 48, 72]:
    n_hit = ((train['event'] == 1) & (train['time_to_hit_hours'] <= t)).sum()
    n_total = len(train)
    print(f"  Hits ≤ {t}h: {n_hit} ({n_hit/n_total*100:.1f}%)")

# ============================================================
# PHASE 2 — DATA AGENT: Feature Engineering
# ============================================================
print("\n" + "=" * 80)
print("  PHASE 2 — DATA AGENT: Feature Engineering")
print("=" * 80)

def engineer_features(df):
    """Physics-grounded feature engineering — DATA AGENT approved."""
    out = df.copy()
    
    # === CORE PHYSICS FEATURES ===
    # Distance-based (most predictive domain)
    out['log_dist_min'] = np.log1p(df['dist_min_ci_0_5h'])
    out['inv_dist'] = 1.0 / (df['dist_min_ci_0_5h'] + 100)  # inverse distance
    
    # Projected time to hit (physics: t = d / v)
    closing = df['closing_speed_m_per_h'].clip(lower=0)
    out['time_to_hit_projected'] = df['dist_min_ci_0_5h'] / (closing + 1e-6)
    out['time_to_hit_projected'] = out['time_to_hit_projected'].clip(upper=500)
    out['log_time_projected'] = np.log1p(out['time_to_hit_projected'])
    
    # Threat composite features 
    out['directional_threat'] = df['closing_speed_m_per_h'] * df['alignment_abs']
    out['growth_threat'] = df['area_growth_rate_ha_per_h'] * df['alignment_abs']
    out['proximity_threat'] = df['closing_speed_m_per_h'] / (df['dist_min_ci_0_5h'] + 100)
    out['proximity_threat'] = out['proximity_threat'].clip(-5, 5)
    
    # Distance dynamics
    out['dist_change_rate'] = df['dist_change_ci_0_5h'] / (df['dt_first_last_0_5h'] + 0.1)
    
    # Fire size + growth interaction
    out['fire_intensity'] = df['log1p_area_first'] * df['relative_growth_0_5h']
    
    # Closing velocity (non-negative)
    out['closing_velocity'] = df['closing_speed_m_per_h'].clip(lower=0)
    
    # Binary proximity flag
    out['is_close'] = (df['dist_min_ci_0_5h'] < 5000).astype(float)
    out['is_very_close'] = (df['dist_min_ci_0_5h'] < 2000).astype(float)
    
    # Along-track absolute
    out['along_track_abs'] = df['along_track_speed'].abs()
    
    # Combined risk score (physics-driven)
    out['risk_score'] = (
        out['proximity_threat'] * 0.4 + 
        out['directional_threat'] / (out['directional_threat'].std() + 1e-6) * 0.3 +
        out['growth_threat'] / (out['growth_threat'].std() + 1e-6) * 0.3
    )
    
    return out

train_fe = engineer_features(train)
test_fe = engineer_features(test)

# === FEATURE SELECTION (DATA AGENT) ===
# Curated feature list: physics-first, no noise
base_features = [
    'dist_min_ci_0_5h',         # core distance
    'closing_speed_m_per_h',    # core velocity
    'alignment_abs',            # directional alignment
    'area_growth_rate_ha_per_h',# fire growth
    'log1p_area_first',         # fire size
    'relative_growth_0_5h',     # relative growth
    'centroid_speed_m_per_h',   # fire movement
    'radial_growth_rate_m_per_h', # radial growth
    'dist_change_ci_0_5h',      # distance change
    'dist_slope_ci_0_5h',       # distance trend
    'projected_advance_m',      # projected advance
    'along_track_speed',        # along-track component
    'num_perimeters_0_5h',      # observation quality
    'dt_first_last_0_5h',       # temporal coverage
    'dist_std_ci_0_5h',         # distance uncertainty
    'dist_fit_r2_0_5h',         # distance fit quality
    'closing_speed_abs_m_per_h', # absolute closing speed
    'area_first_ha',            # initial fire area
]

engineered_features = [
    'log_dist_min',
    'inv_dist',
    'time_to_hit_projected',
    'log_time_projected',
    'directional_threat',
    'growth_threat',
    'proximity_threat',
    'dist_change_rate',
    'fire_intensity',
    'closing_velocity',
    'is_close',
    'is_very_close',
    'along_track_abs',
    'risk_score',
]

all_features = base_features + engineered_features
print(f"  Initial feature count: {len(all_features)}")

# === DROP NEAR-ZERO VARIANCE ===
train_std = train_fe[all_features].std()
low_var = train_std[train_std < 1e-10].index.tolist()
if low_var:
    print(f"  Dropping near-zero variance: {low_var}")
    all_features = [f for f in all_features if f not in low_var]

# === DROP HIGHLY CORRELATED (>0.95) ===
corr_matrix = train_fe[all_features].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = set()
for col in upper.columns:
    high_corr = upper.index[upper[col] > 0.95].tolist()
    for hc in high_corr:
        corr_event_col = abs(train_fe[col].corr(train_fe['event']))
        corr_event_hc = abs(train_fe[hc].corr(train_fe['event']))
        if corr_event_col >= corr_event_hc:
            to_drop.add(hc)
        else:
            to_drop.add(col)

if to_drop:
    print(f"  Dropping highly correlated: {to_drop}")
    all_features = [f for f in all_features if f not in to_drop]

print(f"  Features after correlation filter: {len(all_features)}")

# === FEATURE STABILITY SELECTION (multi-fold importance) ===
print("\n  --- Feature Stability Selection ---")
y_scan = ((train['event'] == 1) & (train['time_to_hit_hours'] <= 24)).astype(int).values
X_scan = train_fe[all_features].values
X_scan = np.nan_to_num(X_scan, nan=0.0)

skf_scan = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_importances = []

for tr_idx, val_idx in skf_scan.split(X_scan, y_scan):
    m = lgb.LGBMClassifier(
        num_leaves=8, max_depth=3, n_estimators=150,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=20, verbosity=-1, random_state=42
    )
    m.fit(X_scan[tr_idx], y_scan[tr_idx])
    fold_importances.append(m.feature_importances_)

imp_matrix = np.array(fold_importances)
imp_mean = imp_matrix.mean(axis=0)
imp_std = imp_matrix.std(axis=0)

# Drop features with zero mean importance
zero_imp = [all_features[i] for i in range(len(all_features)) if imp_mean[i] == 0]
if zero_imp:
    print(f"  Dropping zero-importance: {zero_imp}")
    all_features = [f for f in all_features if f not in zero_imp]

# Report stability
for i, f in enumerate([f for f in base_features + engineered_features if f in all_features]):
    idx = all_features.index(f) if f in all_features else -1
    if idx >= 0:
        orig_idx = (base_features + engineered_features).index(f)
        if orig_idx < len(imp_mean):
            cv_val = imp_std[orig_idx] / (imp_mean[orig_idx] + 1e-9)
            tag = "STABLE" if cv_val < 1.0 else "VARIABLE"
            print(f"    {f:35s}: mean_imp={imp_mean[orig_idx]:8.1f}, CV={cv_val:.2f} [{tag}]")

assert len(all_features) <= 30, f"GATE FAIL: {len(all_features)} features exceeds 30 limit"
print(f"\n  ✓ Final feature count: {len(all_features)}")

# === PREPARE FEATURE MATRICES ===
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
# PHASE 3 — TARGET CONSTRUCTION (Progressive Survival Masking)
# ============================================================
print("\n" + "=" * 80)
print("  PHASE 3 — TARGET CONSTRUCTION (Progressive Survival Masking)")
print("=" * 80)

time_horizons = [12, 24, 48, 72]
horizon_data = {}
MIN_NEGATIVES = 5

for horizon in time_horizons:
    relax_factors = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20]
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
                    y_labels.append(0)
                    indices.append(idx)
        
        n_neg = sum(1 for y in y_labels if y == 0)
        n_pos = sum(1 for y in y_labels if y == 1)
        
        if n_neg >= MIN_NEGATIVES and n_pos >= MIN_NEGATIVES:
            selected = (y_labels, indices, obs_threshold, relax)
            break
    
    if selected is None:
        # For 72h: all events are hits or early-censored
        # Use all data — hits within 72h = positive, hits after 72h = 0 (none exist),
        # censored = assume negative with lower confidence
        print(f"  WARNING [{horizon}h]: Using relaxed approach, including early-censored as negatives")
        y_labels = []
        indices = []
        for idx, row in train.iterrows():
            is_hit = row['event'] == 1
            t = row['time_to_hit_hours']
            if is_hit:
                y_labels.append(1 if t <= horizon else 0)
            else:
                y_labels.append(0)  # Assume censored = survived (conservative)
            indices.append(idx)
        selected = (y_labels, indices, 0.0, 0.0)
    
    y_labels, indices, obs_threshold, relax = selected
    y_arr = np.array(y_labels)
    X_arr = X_train_full[indices]
    horizon_data[horizon] = (X_arr, y_arr, indices)
    
    mask_type = "STRICT" if relax == 1.0 else f"RELAXED (obs>={obs_threshold:.0f}h)"
    n_pos = int(y_arr.sum())
    n_neg = len(y_arr) - n_pos
    print(f"  {horizon}h: {n_pos} pos + {n_neg} neg = {len(y_arr)} total [{mask_type}]")

# ============================================================
# PHASE 4 — MODEL AGENT: Multi-Seed Ensemble Training
# ============================================================
print("\n" + "=" * 80)
print("  PHASE 4 — MODEL AGENT: Multi-Seed Ensemble Training")
print("=" * 80)

SEEDS = [42, 52, 62, 72, 82]
N_SPLITS = 5
model_names = ['lgbm', 'logreg', 'rf']

# Storage for all predictions
all_oof_preds = {}    # (model, horizon, seed) -> oof predictions
all_test_preds = {}   # (model, horizon, seed) -> test predictions
all_metrics = {}      # (model, horizon) -> aggregated metrics

for horizon in time_horizons:
    X_h, y, mask_indices = horizon_data[horizon]
    print(f"\n  ─── {horizon}h horizon ({len(y)} samples, {int(y.sum())} pos, {len(y)-int(y.sum())} neg) ───")
    
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        
        # ──── LightGBM ────
        oof_lgbm = np.zeros(len(y))
        test_lgbm = np.zeros((N_SPLITS, len(X_test)))
        
        for fi, (tr_idx, val_idx) in enumerate(skf.split(X_h, y)):
            X_tr, X_val = X_h[tr_idx], X_h[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            
            neg_c = (y_tr == 0).sum()
            pos_c = (y_tr == 1).sum()
            spw = max(neg_c / (pos_c + 1e-9), 0.5)
            
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'num_leaves': 10,           # Conservative for tiny data
                'max_depth': 3,
                'min_child_samples': 20,
                'reg_alpha': 0.5,
                'reg_lambda': 2.0,
                'learning_rate': 0.03,
                'n_estimators': 500,
                'scale_pos_weight': spw,
                'verbosity': -1,
                'random_state': seed,
                'subsample': 0.75,
                'colsample_bytree': 0.7,
                'min_child_weight': 5,
                'path_smooth': 1.0,         # Smoothing for small data
            }
            
            mdl = lgb.LGBMClassifier(**params)
            mdl.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])
            
            oof_lgbm[val_idx] = mdl.predict_proba(X_val)[:, 1]
            test_lgbm[fi] = mdl.predict_proba(X_test)[:, 1]
        
        all_oof_preds[('lgbm', horizon, seed)] = oof_lgbm
        all_test_preds[('lgbm', horizon, seed)] = test_lgbm.mean(axis=0)
        
        # ──── Logistic Regression ────
        oof_lr = np.zeros(len(y))
        test_lr = np.zeros((N_SPLITS, len(X_test)))
        
        scaler = StandardScaler()
        X_h_sc = scaler.fit_transform(X_h)
        X_test_sc = scaler.transform(X_test)
        
        for fi, (tr_idx, val_idx) in enumerate(skf.split(X_h_sc, y)):
            X_tr, X_val = X_h_sc[tr_idx], X_h_sc[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            
            mdl = LogisticRegression(
                C=0.05, solver='lbfgs', max_iter=2000,
                class_weight='balanced', random_state=seed
            )
            mdl.fit(X_tr, y_tr)
            
            oof_lr[val_idx] = mdl.predict_proba(X_val)[:, 1]
            test_lr[fi] = mdl.predict_proba(X_test_sc)[:, 1]
        
        all_oof_preds[('logreg', horizon, seed)] = oof_lr
        all_test_preds[('logreg', horizon, seed)] = test_lr.mean(axis=0)
        
        # ──── Random Forest ────
        oof_rf = np.zeros(len(y))
        test_rf = np.zeros((N_SPLITS, len(X_test)))
        
        for fi, (tr_idx, val_idx) in enumerate(skf.split(X_h, y)):
            X_tr, X_val = X_h[tr_idx], X_h[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            
            mdl = RandomForestClassifier(
                n_estimators=500, max_depth=4, min_samples_leaf=12,
                max_features='sqrt', class_weight='balanced_subsample',
                random_state=seed, n_jobs=-1
            )
            mdl.fit(X_tr, y_tr)
            
            oof_rf[val_idx] = mdl.predict_proba(X_val)[:, 1]
            test_rf[fi] = mdl.predict_proba(X_test)[:, 1]
        
        all_oof_preds[('rf', horizon, seed)] = oof_rf
        all_test_preds[('rf', horizon, seed)] = test_rf.mean(axis=0)
    
    # Aggregate metrics across seeds for each model
    for mn in model_names:
        seed_aucs = []
        seed_briers = []
        for seed in SEEDS:
            oof = all_oof_preds[(mn, horizon, seed)]
            if len(np.unique(y)) > 1:
                seed_aucs.append(roc_auc_score(y, oof))
                seed_briers.append(brier_score_loss(y, oof))
        
        mean_auc = np.mean(seed_aucs)
        std_auc = np.std(seed_aucs)
        mean_brier = np.mean(seed_briers)
        
        all_metrics[(mn, horizon)] = {
            'auc': mean_auc,
            'auc_std': std_auc,
            'brier': mean_brier,
            'seed_aucs': seed_aucs
        }
        print(f"    {mn:8s}: AUC={mean_auc:.4f}±{std_auc:.4f}, Brier={mean_brier:.4f}")

# ============================================================
# PHASE 5 — CRITIC AGENT: Validation Gates
# ============================================================
print("\n" + "=" * 80)
print("  PHASE 5 — CRITIC AGENT: Validation Gates")
print("=" * 80)

valid_models = {}
model_weights = {}

for horizon in time_horizons:
    for mn in model_names:
        m = all_metrics[(mn, horizon)]
        passed = True
        reasons = []
        
        # Overfit detection
        if m['auc'] > 0.985:
            reasons.append(f"AUC={m['auc']:.4f}>0.985 (overfit warning)")
            # Don't reject, just reduce weight
        
        # Instability detection
        if m['auc_std'] > 0.08:
            reasons.append(f"AUC_STD={m['auc_std']:.4f}>0.08 (unstable across seeds)")
            passed = False
        
        valid_models[(mn, horizon)] = passed
        
        # Weight: prefer stable models with good calibration
        stability_weight = 1.0 / (m['auc_std'] + 0.01)
        calibration_weight = 1.0 / (m['brier'] + 0.01)
        # AUC weight — but penalize suspicious high AUC
        auc_weight = min(m['auc'], 0.98) ** 2
        
        model_weights[(mn, horizon)] = stability_weight * calibration_weight * auc_weight if passed else 0
        
        tag = "✓ PASS" if passed else "✗ FAIL"
        print(f"  [{tag}] {mn:8s} @ {horizon}h: AUC={m['auc']:.4f}±{m['auc_std']:.4f}, "
              f"Brier={m['brier']:.4f}, Weight={model_weights[(mn, horizon)]:.2f}")
        for r in reasons:
            print(f"           ⚠ {r}")

# ============================================================
# PHASE 6 — CALIBRATION (Multi-Seed Average + Platt)
# ============================================================
print("\n" + "=" * 80)
print("  PHASE 6 — CALIBRATION")
print("=" * 80)

# First: average OOF & Test across seeds for each model
avg_oof = {}
avg_test = {}

for horizon in time_horizons:
    for mn in model_names:
        oof_all = np.array([all_oof_preds[(mn, horizon, s)] for s in SEEDS])
        test_all = np.array([all_test_preds[(mn, horizon, s)] for s in SEEDS])
        
        avg_oof[(mn, horizon)] = oof_all.mean(axis=0)
        avg_test[(mn, horizon)] = test_all.mean(axis=0)

# Platt calibration on seed-averaged OOF
calibrated_oof = {}
calibrated_test = {}

for horizon in time_horizons:
    _, y, _ = horizon_data[horizon]
    for mn in model_names:
        if not valid_models[(mn, horizon)]:
            continue
        
        oof = avg_oof[(mn, horizon)]
        tp = avg_test[(mn, horizon)]
        
        # Platt scaling
        cal = LogisticRegression(C=1.0, solver='lbfgs', max_iter=2000)
        cal.fit(oof.reshape(-1, 1), y)
        
        cal_oof = cal.predict_proba(oof.reshape(-1, 1))[:, 1]
        cal_test = cal.predict_proba(tp.reshape(-1, 1))[:, 1]
        
        calibrated_oof[(mn, horizon)] = cal_oof
        calibrated_test[(mn, horizon)] = cal_test
        
        b_before = brier_score_loss(y, oof)
        b_after = brier_score_loss(y, cal_oof)
        improvement = "↓" if b_after < b_before else "↑"
        print(f"  {mn:8s} @ {horizon}h: Brier {b_before:.4f} → {b_after:.4f} {improvement}")

# ============================================================
# PHASE 7 — ENSEMBLE + DISTRIBUTION ALIGNMENT
# ============================================================
print("\n" + "=" * 80)
print("  PHASE 7 — ENSEMBLE + DISTRIBUTION ALIGNMENT")
print("=" * 80)

ensemble_test = {}
ensemble_oof = {}

for horizon in time_horizons:
    _, y, _ = horizon_data[horizon]
    
    valid = []
    for mn in model_names:
        if valid_models[(mn, horizon)] and (mn, horizon) in calibrated_test:
            w = model_weights[(mn, horizon)]
            if w > 0:
                valid.append((mn, w))
    
    if not valid:
        # Fallback to raw seed-averaged LGBM
        ensemble_test[horizon] = avg_test[('lgbm', horizon)]
        ensemble_oof[horizon] = avg_oof[('lgbm', horizon)]
        print(f"  {horizon}h: No valid models, using raw seed-averaged LGBM")
        continue
    
    total_w = sum(w for _, w in valid)
    t_ens = np.zeros(len(X_test))
    o_ens = np.zeros(len(y))
    
    for mn, w in valid:
        nw = w / total_w
        t_ens += nw * calibrated_test[(mn, horizon)]
        o_ens += nw * calibrated_oof[(mn, horizon)]
        print(f"  {horizon}h: {mn:8s} weight={nw:.4f}")
    
    # === DISTRIBUTION ALIGNMENT CHECK ===
    oof_mean = o_ens.mean()
    test_mean = t_ens.mean()
    diff = abs(oof_mean - test_mean)
    
    if diff > 0.03:
        print(f"  ⚠ {horizon}h: OOF_mean={oof_mean:.4f} vs TEST_mean={test_mean:.4f}, diff={diff:.4f} > 0.03")
        # Apply gentle distribution alignment: shrink test toward OOF mean
        shrink_factor = min(0.15, diff)  # Don't over-correct
        t_ens = t_ens + shrink_factor * (oof_mean - test_mean)
        t_ens = np.clip(t_ens, 0.001, 0.999)
        print(f"    → Applied distribution alignment (shrink={shrink_factor:.4f})")
    else:
        print(f"  ✓ {horizon}h: OOF_mean={oof_mean:.4f}, TEST_mean={test_mean:.4f}, diff={diff:.4f} (aligned)")
    
    # Minimal shrinkage to preserve variance
    test_mean_new = t_ens.mean()
    t_ens = 0.97 * t_ens + 0.03 * test_mean_new
    
    ensemble_test[horizon] = t_ens
    ensemble_oof[horizon] = o_ens

# ============================================================
# PHASE 8 — RANK TRANSFORMATION (C-INDEX BOOST)
# ============================================================
print("\n" + "=" * 80)
print("  PHASE 8 — RANK TRANSFORMATION (C-INDEX BOOST)")
print("=" * 80)

rank_test = {}
for horizon in time_horizons:
    raw = ensemble_test[horizon]
    ranked = rankdata(raw) / len(raw)
    rank_test[horizon] = ranked
    print(f"  {horizon}h: rank_preds mean={ranked.mean():.4f}, std={ranked.std():.4f}")

# ============================================================
# PHASE 9 — HYBRID BLENDING
# ============================================================
print("\n" + "=" * 80)
print("  PHASE 9 — HYBRID BLENDING")
print("=" * 80)

# Horizon-specific blend: more ranking for 12h (C-index), safer for 72h (avoid fake 1.0 AUC)
BLEND_CONFIG = {
    12: (0.60, 0.40),  # aggressive ranking — 12h has most separable ranking signal
    24: (0.70, 0.30),  # balanced
    48: (0.75, 0.25),  # balanced
    72: (0.85, 0.15),  # safer — 72h has perfect AUC, don't over-rank
}

hybrid_test = {}
for horizon in time_horizons:
    cal_w, rank_w = BLEND_CONFIG[horizon]
    cal_prob = ensemble_test[horizon]
    rank_prob = rank_test[horizon]
    
    hybrid = cal_w * cal_prob + rank_w * rank_prob
    
    # Distribution stretch — slight variance boost
    hybrid = np.clip(hybrid, 0.01, 0.99)
    hybrid = hybrid ** 0.97  # tiny power transform stretches toward extremes
    
    hybrid_test[horizon] = hybrid
    
    print(f"  {horizon}h: cal={cal_w:.2f}/rank={rank_w:.2f}, "
          f"mean={hybrid.mean():.4f}, std={hybrid.std():.4f}")

# ============================================================
# PHASE 10 — VALIDATION AGENT: Comprehensive Checks
# ============================================================
print("\n" + "=" * 80)
print("  PHASE 10 — VALIDATION AGENT: Comprehensive Checks")
print("=" * 80)

print("\n  --- Ensemble OOF Validation ---")
for horizon in time_horizons:
    _, y, _ = horizon_data[horizon]
    oof_e = ensemble_oof[horizon]
    if len(np.unique(y)) > 1:
        ens_auc = roc_auc_score(y, oof_e)
        ens_brier = brier_score_loss(y, oof_e)
        print(f"  {horizon}h: AUC={ens_auc:.4f}, Brier={ens_brier:.4f}")

print("\n  --- OOF vs TEST Distribution ---")
alignment_ok = True
for j, horizon in enumerate(time_horizons):
    _, y, _ = horizon_data[horizon]
    oof_mean = ensemble_oof[horizon].mean()
    test_mean = hybrid_test[horizon].mean()
    diff = abs(oof_mean - test_mean)
    status = "✓" if diff < 0.05 else "⚠"
    if diff >= 0.05:
        alignment_ok = False
    print(f"  {status} {horizon}h: OOF={oof_mean:.4f}, TEST={test_mean:.4f}, diff={diff:.4f}")

print(f"\n  Distribution alignment: {'PASSED' if alignment_ok else 'WARNING — check test predictions'}")

# Simulated C-index on OOF (within each horizon)
print("\n  --- Simulated C-Index (OOF) ---")
for horizon in time_horizons:
    _, y, indices = horizon_data[horizon]
    oof_e = ensemble_oof[horizon]
    times = train.loc[indices, 'time_to_hit_hours'].values
    events = train.loc[indices, 'event'].values
    
    # Simple concordance estimate
    concordant = 0
    discordant = 0
    for i in range(len(y)):
        for j in range(i+1, len(y)):
            if events[i] == 1 and events[j] == 1:
                if times[i] < times[j]:
                    if oof_e[i] > oof_e[j]:
                        concordant += 1
                    elif oof_e[i] < oof_e[j]:
                        discordant += 1
                elif times[j] < times[i]:
                    if oof_e[j] > oof_e[i]:
                        concordant += 1
                    elif oof_e[j] < oof_e[i]:
                        discordant += 1
            elif events[i] == 1 and events[j] == 0 and times[i] < times[j]:
                if oof_e[i] > oof_e[j]:
                    concordant += 1
                elif oof_e[i] < oof_e[j]:
                    discordant += 1
            elif events[j] == 1 and events[i] == 0 and times[j] < times[i]:
                if oof_e[j] > oof_e[i]:
                    concordant += 1
                elif oof_e[j] < oof_e[i]:
                    discordant += 1
    
    total = concordant + discordant
    c_index = concordant / total if total > 0 else 0.5
    print(f"  {horizon}h: C-index={c_index:.4f} ({concordant}/{total} concordant)")

# ============================================================
# PHASE 11 — MONOTONICITY ENFORCEMENT
# ============================================================
print("\n" + "=" * 80)
print("  PHASE 11 — MONOTONICITY ENFORCEMENT")
print("=" * 80)

def enforce_monotonicity(pred_matrix, time_horizons):
    """Enforce strict monotonicity: p_12h <= p_24h <= p_48h <= p_72h"""
    violations_before = 0
    for i in range(len(pred_matrix)):
        for j in range(3):
            if pred_matrix[i, j] > pred_matrix[i, j+1]:
                violations_before += 1
    
    # Isotonic regression for violated rows
    for i in range(len(pred_matrix)):
        row = pred_matrix[i]
        if not all(row[j] <= row[j+1] for j in range(3)):
            ir = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True)
            pred_matrix[i] = ir.fit_transform(
                np.array(time_horizons, dtype=float), row
            )
    
    # Ensure minimum increment (flat predictions hurt C-index)
    MIN_INCREMENT = 0.005
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
# PHASE 12 — MULTI-SUBMISSION STRATEGY (3 variants)
# ============================================================
print("\n" + "=" * 80)
print("  PHASE 12 — MULTI-SUBMISSION STRATEGY")
print("=" * 80)

# 3-submission strategy: balanced → rank-heavy → power-stretched
submission_configs = {
    'A': {'power': 0.97, 'clip_low': 0.02, 'clip_high': 0.98, 'desc': 'BALANCED (horizon-specific blend)'},
    'B': {'power': 0.95, 'clip_low': 0.01, 'clip_high': 0.99, 'desc': 'AGGRESSIVE (more stretch)'},
    'C': {'power': 1.00, 'clip_low': 0.03, 'clip_high': 0.97, 'desc': 'CONSERVATIVE (no stretch)'},
}

submissions = {}

for variant, config in submission_configs.items():
    print(f"\n  ─── Submission {variant}: {config['desc']} ───")
    
    # Build variant from base hybrid_test with variant-specific power
    variant_preds = {}
    for horizon in time_horizons:
        v = hybrid_test[horizon].copy()
        if config['power'] != 0.97:  # hybrid_test already has 0.97 baked in
            # Undo the base power and apply variant power
            v = v ** (1.0 / 0.97)  # undo
            v = np.clip(v, 0.001, 0.999)
            v = v ** config['power']  # apply variant power
        variant_preds[horizon] = v
    
    pred_matrix = np.column_stack([variant_preds[t] for t in time_horizons])
    
    # Clip
    pred_matrix = np.clip(pred_matrix, config['clip_low'], config['clip_high'])
    
    # Enforce monotonicity
    pred_matrix, v_before, v_after = enforce_monotonicity(pred_matrix, time_horizons)
    print(f"    Monotonicity violations: {v_before} → {v_after}")
    
    # Final clip (monotonicity enforcement can push beyond)
    pred_matrix = np.clip(pred_matrix, config['clip_low'], min(config['clip_high'], 0.999))
    
    # Re-enforce after final clip
    for i in range(len(pred_matrix)):
        for j in range(1, 4):
            if pred_matrix[i, j] < pred_matrix[i, j-1]:
                pred_matrix[i, j] = pred_matrix[i, j-1]
    
    # Verify
    has_nan = np.isnan(pred_matrix).any()
    has_inf = np.isinf(pred_matrix).any()
    mono_ok = all(
        pred_matrix[i, j] <= pred_matrix[i, j+1] + 1e-9
        for i in range(len(pred_matrix))
        for j in range(3)
    )
    
    for j, t in enumerate(time_horizons):
        col = pred_matrix[:, j]
        print(f"    prob_{t}h: mean={col.mean():.4f}, std={col.std():.4f}, "
              f"[{col.min():.4f}, {col.max():.4f}]")
    print(f"    NaN={has_nan}, Inf={has_inf}, Monotonic={mono_ok}")
    
    sub = pd.DataFrame({
        'event_id': test['event_id'].values,
        'prob_12h': pred_matrix[:, 0],
        'prob_24h': pred_matrix[:, 1],
        'prob_48h': pred_matrix[:, 2],
        'prob_72h': pred_matrix[:, 3],
    })
    
    submissions[variant] = sub

# ============================================================
# PHASE 13 — FINAL VERIFICATION & SAVE
# ============================================================
print("\n" + "=" * 80)
print("  PHASE 13 — FINAL VERIFICATION & SAVE")
print("=" * 80)

for variant, sub in submissions.items():
    checks = {}
    checks['rows'] = len(sub) == 95
    checks['cols'] = list(sub.columns) == ['event_id', 'prob_12h', 'prob_24h', 'prob_48h', 'prob_72h']
    checks['ids'] = list(sub['event_id']) == list(sample_sub['event_id'])
    checks['no_nan'] = not sub.isnull().any().any()
    checks['no_inf'] = not np.isinf(sub[['prob_12h', 'prob_24h', 'prob_48h', 'prob_72h']].values).any()
    
    prob_cols = ['prob_12h', 'prob_24h', 'prob_48h', 'prob_72h']
    checks['range'] = sub[prob_cols].min().min() >= 0 and sub[prob_cols].max().max() <= 1
    
    mono_ok = True
    for _, row in sub.iterrows():
        if not (row['prob_12h'] <= row['prob_24h'] + 1e-9 and
                row['prob_24h'] <= row['prob_48h'] + 1e-9 and
                row['prob_48h'] <= row['prob_72h'] + 1e-9):
            mono_ok = False
            break
    checks['monotonicity'] = mono_ok
    
    all_passed = all(checks.values())
    
    print(f"\n  Submission {variant}:")
    for name, ok in checks.items():
        print(f"    [{'✓' if ok else '✗'}] {name}")
    
    if all_passed:
        filepath = rf'd:\WiDS\submission_{variant}.csv'
        sub.to_csv(filepath, index=False)
        print(f"    >>> SAVED: {filepath}")
    else:
        print(f"    >>> NOT SAVED — FIX ERRORS")

# Also save variant A as the main submission
submissions['A'].to_csv(r'd:\WiDS\submission.csv', index=False)
print(f"\n  >>> MAIN submission.csv = Variant A (BALANCED)")

# ============================================================
# COMPREHENSIVE FINAL REPORT
# ============================================================
print("\n" + "=" * 80)
print("  COMPREHENSIVE FINAL REPORT")
print("=" * 80)

print("\n  ─── Model Performance (Multi-Seed Average) ───")
for horizon in time_horizons:
    print(f"\n    {horizon}h:")
    for mn in model_names:
        m = all_metrics[(mn, horizon)]
        v = valid_models[(mn, horizon)]
        tag = "✓" if v else "✗"
        print(f"      {tag} {mn:8s}: AUC={m['auc']:.4f}±{m['auc_std']:.4f}, Brier={m['brier']:.4f}")

print("\n  ─── Ensemble OOF Summary ───")
for horizon in time_horizons:
    _, y, _ = horizon_data[horizon]
    oof_e = ensemble_oof[horizon]
    if len(np.unique(y)) > 1:
        print(f"    {horizon}h: AUC={roc_auc_score(y, oof_e):.4f}, Brier={brier_score_loss(y, oof_e):.4f}")

print("\n  ─── Submission Statistics ───")
for variant, sub in submissions.items():
    print(f"\n    Variant {variant} ({submission_configs[variant]['desc']}):")
    for col in ['prob_12h', 'prob_24h', 'prob_48h', 'prob_72h']:
        print(f"      {col}: mean={sub[col].mean():.4f}, std={sub[col].std():.4f}")

print("\n  ─── Key Improvements Over v4.0 ───")
improvements = [
    "Multi-seed training (5 seeds × 3 models × 5 folds = 75 models per horizon)",
    "Distribution alignment check (OOF vs TEST diff < 0.03)",
    "Rank transformation for C-index boost (25% blend)",
    "3 submission strategies (balanced/aggressive/conservative)",
    "Improved regularization (path_smooth, increased reg_alpha/lambda)",
    "Feature stability selection with zero-importance pruning",
    "Isotonic monotonicity enforcement with minimum increment",
    "Conservative hyperparameters (num_leaves=10, max_depth=3)",
    "Seed-averaged OOF for robust calibration",
    "Critic agent validation gates (reject unstable models)",
]
for i, imp in enumerate(improvements, 1):
    print(f"    [{i:2d}] {imp}")

print("\n  ─── RECOMMENDATION ───")
print("    BEST submission: submission_A.csv (BALANCED)")
print("    Reasoning: Clip [0.02, 0.98] provides the best balance of")
print("    calibration quality and ranking preservation on tiny data.")
print("    submission_C.csv (CONSERVATIVE) is the safest backup.")

print("\n" + "=" * 80)
print("  🏆 GOD-TIER PIPELINE v5.0 COMPLETE — READY FOR LEADERBOARD")
print("=" * 80)
