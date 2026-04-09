"""
Pipeline v23 ULTRA — The 0.98+ Push
======================================================
1. Physics gates: FAR=0.005, ACTIVE=0.995 (aggressive but safe bounds)
2. IPCW-Weighted Models for STATIC: LGB, CatBoost, RF, LogReg, GBSA
3. Multi-model Ensemble + Logit-Space Base Rate Shifting
4. Blended with h_blend & Monotonicity Enforced
"""
import time
start = time.time()

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from scipy.special import expit, logit
import warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("  PIPELINE v23 ULTRA")
print("  Physics Gradients + Multi-Model IPCW + Base Rate Calibration")
print("=" * 70)

# ============================================================
# DATA & HORIZONS
# ============================================================
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
hblend = pd.read_csv("submission.csv")

EVAL_TIMES = [12, 24, 48, 72]
HORIZON_COLS = [f"prob_{t}h" for t in EVAL_TIMES]
n_train, n_test = len(train), len(test)

# ============================================================
# FEATURE ENGINEERING (Only proven physics-grounded features)
# ============================================================
def make_features(df):
    f = pd.DataFrame(index=df.index)
    f["dist_km"] = df.dist_min_ci_0_5h / 1000
    f["log_dist"] = np.log1p(f.dist_km)
    
    # Growth & Speed
    f["closing_speed"] = df.closing_speed_m_per_h
    f["closing_speed_abs"] = df.closing_speed_abs_m_per_h
    f["radial_growth"] = df.radial_growth_rate_m_per_h
    f["area_growth"] = df.area_growth_rate_ha_per_h
    f["log1p_area"] = df.log1p_area_first
    f["log_area_ratio"] = df.log_area_ratio_0_5h
    
    # Dynamics
    f["alignment"] = df.alignment_abs
    f["along_track"] = df.along_track_speed
    f["dt_first_last"] = df.dt_first_last_0_5h
    f["num_perimeters"] = df.num_perimeters_0_5h
    
    # Derived Physical Risk Metrics
    safe_speed = np.maximum(df.closing_speed_abs_m_per_h, 0.01)
    f["eta_hours"] = df.dist_min_ci_0_5h / safe_speed
    f["log_eta"] = np.log1p(f.eta_hours.clip(0, 1000))
    f["projected_advance"] = df.projected_advance_m
    f["dist_accel"] = df.dist_accel_m_per_h2
    
    # Interactions
    f["dist_x_speed"] = f.dist_km * f.closing_speed
    f["dist_x_align"] = f.dist_km * f.alignment
    f["eta_x_align"] = f.log_eta * f.alignment
    
    return f

X_train_df = make_features(train)
X_test_df = make_features(test)

# Zone Classification
def get_zones(df):
    far = df.dist_min_ci_0_5h >= 5000
    near = ~far
    active = near & ((df.radial_growth_rate_m_per_h > 0) | (df.area_growth_rate_ha_per_h > 0))
    static = near & ~active
    return far, active, static

train_far, train_active, train_static = get_zones(train)
test_far, test_active, test_static = get_zones(test)

X_static_train = X_train_df[train_static].values
X_static_test = X_test_df[test_static].values
y_static_train_time = train.loc[train_static, "time_to_hit_hours"].values
y_static_train_event = train.loc[train_static, "event"].astype(int).values

n_static_train = len(X_static_train)
n_static_test = len(X_static_test)

print(f"Zones - Train: {train_far.sum()} far, {train_active.sum()} active, {n_static_train} static")
print(f"Zones - Test : {test_far.sum()} far, {test_active.sum()} active, {n_static_test} static")

# ============================================================
# IPCW ENGINE
# ============================================================
def compute_ipcw_weights(times, events, horizon):
    unique_t = np.sort(np.unique(times))
    surv = np.ones(len(unique_t))
    for i, t_val in enumerate(unique_t):
        at_risk = (times >= t_val).sum()
        censored = ((times == t_val) & (events == 0)).sum()
        if at_risk > 0: surv[i] = 1 - censored / at_risk
        if i > 0: surv[i] *= surv[i - 1]
    def G(t_q):
        idx = np.searchsorted(unique_t, t_q, side="right") - 1
        return max(surv[idx], 0.05) if idx >= 0 else 1.0
    
    weights = np.ones(len(times))
    for i in range(len(times)):
        if events[i] == 1 and times[i] <= horizon:
            weights[i] = 1.0 / G(times[i])
        elif times[i] >= horizon:
            weights[i] = 1.0 / G(horizon)
    return weights

# ============================================================
# MODEL TRAINING ON STATIC ZONE ONLY
# ============================================================
N_FOLDS = 5
SEEDS = [42, 123, 777, 999, 2026, 888, 314, 555]
# Use first 5 seeds for models
N_SEEDS = 5 

# We will collect predictions per horizon
oof_preds = {h: [] for h in EVAL_TIMES}
test_preds = {h: [] for h in EVAL_TIMES}

for h in EVAL_TIMES:
    print(f"\\n--- Training STATIC models for {h}h ---")
    y_binary = ((y_static_train_event == 1) & (y_static_train_time <= h)).astype(int)
    pos_rate = y_binary.mean()
    
    if pos_rate >= 0.99:
        print(f"All STATIC events bit at {h}h in train -> predicting 0.995")
        oof_preds[h] = np.ones(n_static_train) * 0.995
        test_preds[h] = np.ones(n_static_test) * 0.995
        continue
        
    weights = compute_ipcw_weights(y_static_train_time, y_static_train_event, h)
    
    model_oof = np.zeros(n_static_train)
    model_test = np.zeros(n_static_test)
    counts = np.zeros(n_static_train)
    
    scaler_X = StandardScaler()
    X_static_train_scaled = scaler_X.fit_transform(X_static_train)
    X_static_test_scaled = scaler_X.transform(X_static_test)
    
    for seed in SEEDS[:N_SEEDS]:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for tr_idx, va_idx in skf.split(X_static_train, y_binary):
            X_tr, X_va = X_static_train[tr_idx], X_static_train[va_idx]
            X_tr_sc, X_va_sc = X_static_train_scaled[tr_idx], X_static_train_scaled[va_idx]
            y_tr, y_va = y_binary[tr_idx], y_binary[va_idx]
            w_tr, w_va = weights[tr_idx], weights[va_idx]
            
            # 1. LightGBM
            lgb_dtrain = lgb.Dataset(X_tr, y_tr, weight=w_tr)
            lgb_dval = lgb.Dataset(X_va, y_va, weight=w_va, reference=lgb_dtrain)
            lgb_params = {"objective":"binary", "learning_rate":0.01, "max_depth":3, "num_leaves":6, "verbosity":-1, "seed":seed}
            gbm = lgb.train(lgb_params, lgb_dtrain, 1000, valid_sets=[lgb_dval], callbacks=[lgb.early_stopping(50, verbose=False)])
            p_lgb_va = gbm.predict(X_va)
            p_lgb_te = gbm.predict(X_static_test)
            
            # 2. CatBoost
            cb_train = Pool(X_tr, y_tr, weight=w_tr)
            cb_val = Pool(X_va, y_va, weight=w_va)
            cb = CatBoostClassifier(iterations=800, learning_rate=0.015, depth=3, logging_level='Silent', random_seed=seed)
            cb.fit(cb_train, eval_set=cb_val, early_stopping_rounds=50)
            p_cb_va = cb.predict_proba(X_va)[:, 1]
            p_cb_te = cb.predict_proba(X_static_test)[:, 1]
            
            # 3. Logistic Regression
            lr = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000)
            lr.fit(X_tr_sc, y_tr, sample_weight=w_tr)
            p_lr_va = lr.predict_proba(X_va_sc)[:, 1]
            p_lr_te = lr.predict_proba(X_static_test_scaled)[:, 1]
            
            # 4. RandomForest
            rf = RandomForestClassifier(n_estimators=300, max_depth=3, min_samples_leaf=5, random_state=seed, class_weight='balanced')
            rf.fit(X_tr, y_tr, sample_weight=w_tr)
            p_rf_va = rf.predict_proba(X_va)[:, 1]
            p_rf_te = rf.predict_proba(X_static_test)[:, 1]
            
            # Average blend
            blend_va = 0.4*p_lgb_va + 0.3*p_cb_va + 0.15*p_lr_va + 0.15*p_rf_va
            blend_te = 0.4*p_lgb_te + 0.3*p_cb_te + 0.15*p_lr_te + 0.15*p_rf_te
            
            model_oof[va_idx] += blend_va
            counts[va_idx] += 1
            model_test += blend_te
            
    model_oof /= counts
    model_test /= (N_SEEDS * N_FOLDS)
    
    # Store
    oof_preds[h] = model_oof
    test_preds[h] = model_test
    print(f"  {h}h OOF Brier Score: {brier_score_loss(y_binary, model_oof):.5f}")

# ============================================================
# ASSEMBLE FULL-PREDICTIONS & APPLY PHYSICS GATES
# ============================================================
FAR_P = 0.001
ACTIVE_P = 0.999

def build_full_probs(static_arr_dict, df_far, df_active, df_static, df_id):
    df = pd.DataFrame({"event_id": df_id})
    for h in EVAL_TIMES:
        col = f"prob_{h}h"
        df[col] = 0.0
        df.loc[df_far, col] = FAR_P
        df.loc[df_active, col] = ACTIVE_P
        df.loc[df_static, col] = static_arr_dict[h]
    return df

train_full = build_full_probs(oof_preds, train_far, train_active, train_static, train.event_id)
test_full_pure = build_full_probs(test_preds, test_far, test_active, test_static, test.event_id)

# ============================================================
# MONOTONICITY & CLIPPING
# ============================================================
def enforce_constraints(df):
    d = df.copy()
    for col in HORIZON_COLS: d[col] = d[col].clip(FAR_P, ACTIVE_P)
    for i in range(len(d)):
        for j in range(1, 4):
            if d.loc[i, HORIZON_COLS[j]] < d.loc[i, HORIZON_COLS[j-1]]:
                d.loc[i, HORIZON_COLS[j]] = d.loc[i, HORIZON_COLS[j-1]]
    return d

train_full = enforce_constraints(train_full)
test_full_pure = enforce_constraints(test_full_pure)

# ============================================================
# BLEND WITH H_BLEND & COMPILE VERSIONS
# ============================================================
def create_blend(weight_ml, weight_hb, name):
    sub = pd.DataFrame({"event_id": test.event_id})
    for col in HORIZON_COLS:
        # Physics gates hold solid
        sub[col] = test_full_pure[col] 
        # Only blend the static zone
        sub.loc[test_static, col] = weight_ml * test_full_pure.loc[test_static, col] + weight_hb * hblend.loc[test_static, col]
    
    sub = enforce_constraints(sub)
    sub.to_csv(f"{name}.csv", index=False)
    print(f"Created {name}.csv (ML {weight_ml:.2f} : HB {weight_hb:.2f})")
    
print("\\n--- GENERATING SUBMISSIONS ---")
create_blend(1.0, 0.0, "submission_v23_PURE")
create_blend(0.6, 0.4, "submission_v23_B60")
create_blend(0.4, 0.6, "submission_v23_B40")
create_blend(0.5, 0.5, "submission_v23_SAFE")

print(f"\\nPipeline execution took: {(time.time() - start)/60:.2f} minutes")
print("=" * 70)
print("  ALL SUCCESSFUL")
print("=" * 70)
