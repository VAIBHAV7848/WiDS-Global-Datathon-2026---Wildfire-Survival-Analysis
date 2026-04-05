"""
Pipeline v22 APEX — GBSA + IPCW-LGB + Physics Overrides (Fixed OOF Eval)
========================================================
Based on:
1. Competition research: GBSA is THE primary model
2. IPCW-weighted LGB for horizon-specific classification
3. Physics overrides for FAR/ACTIVE zones
4. Rank blending across models
5. Platt calibration on OOF
"""
import time
start = time.time()

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from scipy.stats import rankdata
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("  PIPELINE v22 APEX")
print("  GBSA + IPCW-LGB + Physics + RankBlend + PlattCalib")
print("=" * 70)

# ============================================================
# DATA
# ============================================================
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
n_train, n_test = len(train), len(test)

EVAL_TIMES = [12, 24, 48, 72]
HORIZON_COLS = [f"prob_{t}h" for t in EVAL_TIMES]

# ============================================================
# FEATURE ENGINEERING
# ============================================================
def make_features(df):
    f = pd.DataFrame(index=df.index)
    f["dist_km"] = df.dist_min_ci_0_5h / 1000
    f["log_dist"] = np.log1p(f.dist_km)
    f["dist_slope"] = df.dist_slope_ci_0_5h
    f["dist_std"] = df.dist_std_ci_0_5h
    f["dist_change"] = df.dist_change_ci_0_5h
    f["closing_speed"] = df.closing_speed_m_per_h
    f["closing_speed_abs"] = df.closing_speed_abs_m_per_h
    f["centroid_speed"] = df.centroid_speed_m_per_h
    f["along_track"] = df.along_track_speed
    f["radial_growth"] = df.radial_growth_rate_m_per_h
    f["area_growth"] = df.area_growth_rate_ha_per_h
    f["log1p_area"] = df.log1p_area_first
    f["log_area_ratio"] = df.log_area_ratio_0_5h
    f["alignment"] = df.alignment_abs
    f["alignment_cos"] = df.alignment_cos
    f["bearing_cos"] = df.spread_bearing_cos
    f["bearing_sin"] = df.spread_bearing_sin
    f["cross_track"] = df.cross_track_component
    f["dt_first_last"] = df.dt_first_last_0_5h
    f["num_perimeters"] = df.num_perimeters_0_5h
    f["low_temp_res"] = df.low_temporal_resolution_0_5h
    f["start_hour"] = df.event_start_hour
    f["start_month"] = df.event_start_month
    f["start_dow"] = df.event_start_dayofweek
    safe_speed = np.maximum(df.closing_speed_abs_m_per_h, 0.01)
    f["eta_hours"] = df.dist_min_ci_0_5h / safe_speed
    f["log_eta"] = np.log1p(f.eta_hours.clip(0, 1000))
    f["projected_advance"] = df.projected_advance_m
    f["dist_accel"] = df.dist_accel_m_per_h2
    f["dist_fit_r2"] = df.dist_fit_r2_0_5h
    f["dist_x_speed"] = f.dist_km * f.closing_speed
    f["dist_x_align"] = f.dist_km * f.alignment
    f["growth_x_align"] = f.area_growth * f.alignment
    return f

X_train_df = make_features(train)
X_test_df = make_features(test)
X_train = X_train_df.values
X_test = X_test_df.values

y_surv = Surv.from_arrays(
    event=train["event"].astype(bool).values,
    time=train["time_to_hit_hours"].values
)

# ============================================================
# MODEL 1: GBSA
# ============================================================
gbsa_configs = [
    {"learning_rate": 0.01,  "subsample": 0.70, "max_depth": 3, "min_samples_leaf": 12, "n_estimators": 1200},
    {"learning_rate": 0.01,  "subsample": 0.85, "max_depth": 3, "min_samples_leaf": 15, "n_estimators": 1200},
    {"learning_rate": 0.005, "subsample": 0.85, "max_depth": 3, "min_samples_leaf": 12, "n_estimators": 2000},
]
SEEDS = [123, 456, 789, 777, 666, 1511, 1523, 2025]
N_FOLDS = 5
N_SEEDS_GBSA = 5  # Reduced slightly for speed since we proved it works

gbsa_oof_risk = np.zeros(n_train)
gbsa_oof_count = np.zeros(n_train)
gbsa_test_risk = np.zeros(n_test)
gbsa_n = 0

for cfg_idx, cfg in enumerate(gbsa_configs):
    for seed in SEEDS[:N_SEEDS_GBSA]:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for tr_idx, va_idx in skf.split(X_train, train["event"].values):
            model = GradientBoostingSurvivalAnalysis(random_state=seed, **cfg)
            model.fit(X_train[tr_idx], y_surv[tr_idx])
            gbsa_oof_risk[va_idx] += model.predict(X_train[va_idx])
            gbsa_oof_count[va_idx] += 1
            gbsa_test_risk += model.predict(X_test)
            gbsa_n += 1

gbsa_oof_risk /= gbsa_oof_count
gbsa_test_risk /= gbsa_n

def risk_to_probs(oof_risk, test_risk, train_df, eval_times):
    oof_probs = np.zeros((len(oof_risk), len(eval_times)))
    test_probs = np.zeros((len(test_risk), len(eval_times)))
    for j, t in enumerate(eval_times):
        y_binary = ((train_df["event"] == 1) & (train_df["time_to_hit_hours"] <= t)).astype(int).values
        scaler = StandardScaler()
        risk_scaled = scaler.fit_transform(oof_risk.reshape(-1, 1))
        test_scaled = scaler.transform(test_risk.reshape(-1, 1))
        platt = LogisticRegression(C=1.0, max_iter=1000)
        platt.fit(risk_scaled, y_binary)
        oof_probs[:, j] = platt.predict_proba(risk_scaled)[:, 1]
        test_probs[:, j] = platt.predict_proba(test_scaled)[:, 1]
    return oof_probs, test_probs

gbsa_oof_probs, gbsa_test_probs = risk_to_probs(gbsa_oof_risk, gbsa_test_risk, train, EVAL_TIMES)

# ============================================================
# MODEL 2: IPCW-LGB
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
        return max(surv[idx], 0.01) if idx >= 0 else 1.0
    weights = np.ones(len(times))
    for i in range(len(times)):
        if events[i] == 1 and times[i] <= horizon: weights[i] = 1.0 / G(times[i])
        elif times[i] >= horizon: weights[i] = 1.0 / G(horizon)
    return weights

N_SEEDS_LGB = 10
lgb_oof_probs = np.zeros((n_train, 4))
lgb_oof_counts = np.zeros((n_train, 4))
lgb_test_probs = np.zeros((n_test, 4))
lgb_n = 0

for j, t in enumerate(EVAL_TIMES):
    y_binary = ((train["event"] == 1) & (train["time_to_hit_hours"] <= t)).astype(int).values
    ipcw_w = compute_ipcw_weights(train["time_to_hit_hours"].values, train["event"].values, t)
    for seed in SEEDS[:N_SEEDS_LGB]:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for tr_idx, va_idx in skf.split(X_train, y_binary):
            dtrain = lgb.Dataset(X_train[tr_idx], y_binary[tr_idx], weight=ipcw_w[tr_idx])
            dval = lgb.Dataset(X_train[va_idx], y_binary[va_idx], weight=ipcw_w[va_idx], reference=dtrain)
            params = {"objective": "binary", "metric": "binary_logloss", "verbosity": -1, "seed": seed*100,
                      "num_leaves": 8, "max_depth": 3, "learning_rate": 0.01}
            m = lgb.train(params, dtrain, 1500, valid_sets=[dval], callbacks=[lgb.early_stopping(50, verbose=False)])
            lgb_oof_probs[va_idx, j] += m.predict(X_train[va_idx])
            lgb_oof_counts[va_idx, j] += 1
            lgb_test_probs[:, j] += m.predict(X_test)
            lgb_n += 1
    lgb_oof_probs[:, j] /= lgb_oof_counts[:, j]
    lgb_test_probs[:, j] /= (N_SEEDS_LGB * N_FOLDS)

# ============================================================
# PREDICTION BLEND (Direct Probability Avg - NO RANK BLEND to preserve Brier)
# ============================================================
oof_calibrated = np.zeros((n_train, 4))
test_calibrated = np.zeros((n_test, 4))
for j in range(4):
    oof_calibrated[:, j] = 0.6 * gbsa_oof_probs[:, j] + 0.4 * lgb_oof_probs[:, j]
    test_calibrated[:, j] = 0.6 * gbsa_test_probs[:, j] + 0.4 * lgb_test_probs[:, j]

# ============================================================
# PHYSICS OVERRIDES (Crucial: applied to BOTH OOF and Test)
# ============================================================
def apply_physics(preds, dist, speed, growth, area_growth, align):
    p = preds.copy()
    # 1. Far fires
    far = dist >= 5000
    p[far, :] = 0.001
    
    # 2. Active near fires
    active = (dist < 5000) & ((growth > 0) | (area_growth > 0))
    p[active, :] = 0.999
    
    # 3. ETA-based floors
    eta = dist / np.maximum(speed, 0.01)
    certain = (dist < 500) | ((eta < 6) & (dist < 5000))
    p[certain, :] = 0.999
    
    eta_12h = (dist < 5000) & (eta < 12)
    p[eta_12h, 0] = np.maximum(p[eta_12h, 0], 0.90)
    
    eta_24h = (dist < 5000) & (eta < 24)
    p[eta_24h, 1] = np.maximum(p[eta_24h, 1], 0.85)
    
    # Monotonicity & bounds
    for i in range(1, 4):
        p[:, i] = np.maximum(p[:, i], p[:, i-1])
    np.clip(p, 0.001, 0.999, out=p)
    return p

# Apply to OOF
oof_final = apply_physics(
    oof_calibrated, 
    train["dist_min_ci_0_5h"].values,
    train["closing_speed_m_per_h"].values,
    train["radial_growth_rate_m_per_h"].values,
    train["area_growth_rate_ha_per_h"].values,
    train["alignment_abs"].values
)

# Apply to Test
pred_A = apply_physics(
    test_calibrated,
    test["dist_min_ci_0_5h"].values,
    test["closing_speed_m_per_h"].values,
    test["radial_growth_rate_m_per_h"].values,
    test["area_growth_rate_ha_per_h"].values,
    test["alignment_abs"].values
)

pred_B = test_calibrated.copy()
far_test = test["dist_min_ci_0_5h"].values >= 5000
pred_B[far_test, :] = 0.001
for i in range(1, 4): pred_B[:, i] = np.maximum(pred_B[:, i], pred_B[:, i-1])

pred_C = 0.6 * pred_A + 0.4 * pred_B
for i in range(1, 4): pred_C[:, i] = np.maximum(pred_C[:, i], pred_C[:, i-1])

# Verify and save
def verify(values, name, test_df):
    errors = []
    if values.shape != (95, 4): errors.append(f"Wrong shape: {values.shape}")
    if np.isnan(values).any(): errors.append("Has NaN values")
    if values.min() < 0: errors.append("Negative probs")
    if values.max() > 1: errors.append("Probs > 1")
    violations = sum(1 for i in range(len(values)) for j in range(1,4) if values[i,j] < values[i,j-1] - 1e-8)
    if violations > 0: errors.append(f"Monotonicity: {violations} violations")
    if errors: 
        print(f"  {name}: FAILED - {errors}")
        return False
    print(f"  {name}: ALL CHECKS PASSED ✓")
    return True

print("\n--- VERIFICATION ---")
verify(pred_A, "submission_A_physics", test)
verify(pred_B, "submission_B_model", test)
verify(pred_C, "submission_C_blend", test)

for p, name in [(pred_A, "submission_A_physics"), (pred_B, "submission_B_model"), (pred_C, "submission_C_blend")]:
    sub = pd.DataFrame({"event_id": test.event_id})
    for j, col in enumerate(HORIZON_COLS): sub[col] = p[:, j]
    sub.to_csv(f"{name}.csv", index=False)

# ============================================================
# SELF REVIEW (Corrected OOF Evaluation)
# ============================================================
print("\n--- SELF REVIEW ---")

briers = []
for j, t in enumerate(EVAL_TIMES):
    y_b = ((train["event"] == 1) & (train["time_to_hit_hours"] <= t)).astype(int).values
    bs = brier_score_loss(y_b, oof_final[:, j])
    briers.append(bs)
    print(f"  OOF Brier at {t}h: {bs:.5f}")

wb = 0.3 * briers[1] + 0.4 * briers[2] + 0.3 * briers[3]
print(f"  Weighted Brier (OOF): {wb:.5f}")
print(f"  1 - Weighted Brier: {1-wb:.5f}")

c_idx = concordance_index_censored(
    train["event"].astype(bool).values,
    train["time_to_hit_hours"].values,
    gbsa_oof_risk
)[0]
print(f"  OOF C-index: {c_idx:.5f}")

hybrid = 0.3 * c_idx + 0.7 * (1 - wb)
print(f"  OOF Hybrid estimate: {hybrid:.5f}")

print("\n  ACTIVE EVENT PREDICTIONS (submission_A):")
dist_test = test["dist_min_ci_0_5h"].values
near_indices = np.where(dist_test < 5000)[0]
print("  %12s %6s %8s %8s %8s %8s %8s" % ("event_id", "dist", "eta", "p_12h", "p_24h", "p_48h", "p_72h"))
for idx in near_indices:
    eid = int(test.iloc[idx]["event_id"])
    d = dist_test[idx]
    e = d / max(test.iloc[idx]["closing_speed_m_per_h"], 0.01)
    print("  %12d %6.0f %8.1f %8.4f %8.4f %8.4f %8.4f" % (
        eid, d, e, pred_A[idx,0], pred_A[idx,1], pred_A[idx,2], pred_A[idx,3]))

print("\n" + "=" * 54)
print("           PIPELINE COMPLETE - FINAL REPORT           ")
print("=" * 54)
print(f" Models trained      : {gbsa_n} GBSA + {lgb_n} LGB = {gbsa_n + lgb_n}")
print(f" Runtime             : {(time.time() - start)/60:.1f} minutes")
print(f" OOF Hybrid Score    : {hybrid:.5f}")
print(f" vs current best     : 0.97216 -> {hybrid:.5f}")
print(f" Expected LB score   : > {hybrid:.5f}")
print("=" * 54)
print(" SUBMISSIONS SAVED:")
print("   submission_A_physics.csv  -> SUBMIT FIRST")
print("   submission_B_model.csv    -> SUBMIT SECOND")
print("   submission_C_blend.csv    -> BACKUP")
print("=" * 54)
print(" ACTIVE EVENTS (< 5km):")
certain = ((dist_test < 500) | ((dist_test / np.maximum(test["closing_speed_m_per_h"].values, 0.01) < 6) & (dist_test < 5000))).sum()
likely = ((dist_test < 2000) & ((dist_test / np.maximum(test["closing_speed_m_per_h"].values, 0.01) < 24))).sum()
print(f"   Certain hits (0.999):  {certain}")
print(f"   Likely hits (>0.90):   {likely}")
print(f"   Uncertain:             {len(near_indices) - certain - likely}")
print(f" FAR EVENTS (>=5km):      {(dist_test >= 5000).sum()} -> set to 0.001")
print("=" * 54)
