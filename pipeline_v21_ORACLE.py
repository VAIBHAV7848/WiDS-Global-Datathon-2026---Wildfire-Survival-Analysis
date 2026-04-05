"""
Pipeline v21 ORACLE — The Definitive Submission
================================================
EVERYTHING discovered, validated, and combined:

INSIGHT 1: Far zone (>=5km) NEVER hits → 0/152 → P=0.001
INSIGHT 2: Active near fires (<5km, growing) ALWAYS hit ALL horizons → 18/18 → P=0.999
INSIGHT 3: Static near fires (<5km, not growing) → ML required
INSIGHT 4: prob_12h NOT in Brier (only affects C-index weight=0.3)
INSIGHT 5: prob_48h has HIGHEST Brier weight (0.4)
INSIGHT 6: IPCW likely down-weights censored events → near zone matters most
INSIGHT 7: Diverse ensemble > single model on 52 training rows
INSIGHT 8: h_blend (LB=0.97175) has validated ranking signal worth preserving
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from scipy.stats import rankdata
import warnings
warnings.filterwarnings("ignore")
import lightgbm as lgb

try:
    import catboost as cb
    HAS_CB = True
except ImportError:
    HAS_CB = False

print("=" * 70)
print("  PIPELINE v21 ORACLE — Zero-Flaw Definitive Model")
print("=" * 70)

# ============================================================
# LOAD DATA
# ============================================================
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Load PROVEN submissions for blending
hblend = pd.read_csv("submission.csv")         # LB = 0.97175
v18b = pd.read_csv("submission_v18_BLEND.csv") # LB = 0.97216

HORIZONS = [12, 24, 48, 72]
HORIZON_COLS = ["prob_12h", "prob_24h", "prob_48h", "prob_72h"]

# ============================================================
# ZONE CLASSIFICATION (verified: perfect 5km split)
# ============================================================
def get_zones(df):
    far = (df.dist_min_ci_0_5h >= 5000).values
    active = ((df.dist_min_ci_0_5h < 5000) &
              ((df.radial_growth_rate_m_per_h > 0) |
               (df.area_growth_rate_ha_per_h > 0))).values
    static = ((df.dist_min_ci_0_5h < 5000) & ~(
              (df.radial_growth_rate_m_per_h > 0) |
              (df.area_growth_rate_ha_per_h > 0))).values
    return far, active, static

tr_far, tr_active, tr_static = get_zones(train)
te_far, te_active, te_static = get_zones(test)

print(f"Train: {tr_far.sum()} far | {tr_active.sum()} active | {tr_static.sum()} static")
print(f"Test:  {te_far.sum()} far | {te_active.sum()} active | {te_static.sum()} static")

# ============================================================
# STATIC ZONE FEATURES (the only zone needing ML)
# ============================================================
static_train = train[tr_static].copy().reset_index(drop=True)
static_test = test[te_static].copy().reset_index(drop=True)

def make_features(df):
    """Features validated by correlation analysis on static zone."""
    f = pd.DataFrame(index=df.index)
    # Primary predictors (highest correlation with time_to_hit in static zone)
    f["dt_first_last"] = df.dt_first_last_0_5h.values           # r=-0.44
    f["num_perimeters"] = df.num_perimeters_0_5h.values          # r=-0.36
    f["alignment"] = df.alignment_abs.values                      # r=-0.37
    f["log1p_area"] = df.log1p_area_first.values                 # r=-0.24
    f["bearing_cos"] = df.spread_bearing_cos.values              # r=+0.30
    # Secondary predictors
    f["centroid_speed"] = df.centroid_speed_m_per_h.values        # r=-0.18
    f["bearing_sin"] = df.spread_bearing_sin.values              # r=-0.18
    f["dist_km"] = (df.dist_min_ci_0_5h / 1000).values
    f["log_dist"] = np.log1p(f["dist_km"].values)
    f["dist_slope"] = df.dist_slope_ci_0_5h.values
    f["closing_abs"] = df.closing_speed_abs_m_per_h.values
    # Temporal
    f["start_hour"] = df.event_start_hour.values
    f["start_month"] = df.event_start_month.values
    # Key interactions
    f["align_x_dt"] = f["alignment"].values * f["dt_first_last"].values
    f["dist_x_align"] = f["dist_km"].values * f["alignment"].values
    return f.values, list(f.columns)

X_train, feat_names = make_features(static_train)
X_test, _ = make_features(static_test)
print(f"\nStatic features: {len(feat_names)}")

# ============================================================
# MULTI-MODEL ENSEMBLE FOR STATIC ZONE
# ============================================================
SEEDS = [42, 123, 2024, 7, 999, 314, 1337, 555, 888, 2025]
N_FOLDS = 5

static_oof = {}
static_preds = {}

for h in HORIZONS:
    y = (static_train.time_to_hit_hours <= h).astype(int).values
    pos_rate = y.mean()
    print(f"\n--- Horizon {h}h: {y.sum()}/{len(y)} positive ({pos_rate:.3f}) ---")

    if pos_rate >= 0.99:
        static_oof[h] = np.ones(len(static_train)) * 0.999
        static_preds[h] = np.ones(len(static_test)) * 0.999
        print("  All positive -> assign 0.999")
        continue

    # Accumulate predictions from all models and seeds
    oof_accum = np.zeros(len(static_train))
    oof_counts = np.zeros(len(static_train))
    test_accum = np.zeros(len(static_test))
    n_models_total = 0
    all_fold_briers = []

    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y)):
            X_tr, X_va = X_train[tr_idx], X_train[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            fold_preds_va = []
            fold_preds_te = []
            rs = seed * 100 + fold

            # ---- Model 1: LightGBM ----
            dtrain = lgb.Dataset(X_tr, y_tr)
            dval = lgb.Dataset(X_va, y_va, reference=dtrain)
            lgb_params = {
                "objective": "binary", "metric": "binary_logloss",
                "verbosity": -1, "seed": rs,
                "num_leaves": 6, "max_depth": 2,
                "min_child_samples": 5,
                "reg_alpha": 5.0, "reg_lambda": 10.0,
                "colsample_bytree": 0.5, "subsample": 0.6,
                "learning_rate": 0.01,
            }
            m_lgb = lgb.train(lgb_params, dtrain, 1000,
                              valid_sets=[dval],
                              callbacks=[lgb.early_stopping(50, verbose=False)])
            fold_preds_va.append(m_lgb.predict(X_va))
            fold_preds_te.append(m_lgb.predict(X_test))

            # ---- Model 2: Gradient Boosting ----
            m_gb = GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.02, max_depth=2,
                min_samples_leaf=5, subsample=0.6, random_state=rs
            )
            m_gb.fit(X_tr, y_tr)
            fold_preds_va.append(m_gb.predict_proba(X_va)[:, 1])
            fold_preds_te.append(m_gb.predict_proba(X_test)[:, 1])

            # ---- Model 3: Logistic Regression ----
            m_lr = LogisticRegression(C=0.1, max_iter=1000, random_state=rs)
            m_lr.fit(X_tr, y_tr)
            fold_preds_va.append(m_lr.predict_proba(X_va)[:, 1])
            fold_preds_te.append(m_lr.predict_proba(X_test)[:, 1])

            # ---- Model 4: CatBoost ----
            if HAS_CB:
                m_cb = cb.CatBoostClassifier(
                    iterations=300, learning_rate=0.02, depth=2,
                    l2_leaf_reg=10.0, random_seed=rs,
                    verbose=0, subsample=0.6
                )
                m_cb.fit(X_tr, y_tr, eval_set=(X_va, y_va),
                         early_stopping_rounds=50, verbose=0)
                fold_preds_va.append(m_cb.predict_proba(X_va)[:, 1])
                fold_preds_te.append(m_cb.predict_proba(X_test)[:, 1])

            # ---- Model 5: Random Forest ----
            m_rf = RandomForestClassifier(
                n_estimators=200, max_depth=3, min_samples_leaf=5,
                random_state=rs, n_jobs=-1
            )
            m_rf.fit(X_tr, y_tr)
            fold_preds_va.append(m_rf.predict_proba(X_va)[:, 1])
            fold_preds_te.append(m_rf.predict_proba(X_test)[:, 1])

            # Average across all models for this fold
            p_va = np.mean(fold_preds_va, axis=0)
            p_te = np.mean(fold_preds_te, axis=0)

            oof_accum[va_idx] += p_va
            oof_counts[va_idx] += 1
            test_accum += p_te
            n_models_total += 1

            bs = brier_score_loss(y_va, p_va)
            all_fold_briers.append(bs)

    # Compute final OOF and test predictions
    oof = oof_accum / oof_counts
    test_pred = test_accum / n_models_total

    static_oof[h] = oof
    static_preds[h] = test_pred

    mean_bs = np.mean(all_fold_briers)
    oof_bs = brier_score_loss(y, oof)
    n_models = 5 if HAS_CB else 4
    print(f"  {n_models} models x {len(SEEDS)} seeds x {N_FOLDS} folds")
    print(f"  Fold Brier: {mean_bs:.5f} +/- {np.std(all_fold_briers):.4f}")
    print(f"  OOF  Brier: {oof_bs:.5f} (vs base rate: {min(pos_rate, 1-pos_rate):.5f})")

# ============================================================
# ASSEMBLE SUBMISSIONS
# ============================================================
print("\n" + "=" * 70)
print("ASSEMBLING SUBMISSIONS")
print("=" * 70)

def assemble(far_p, active_p, static_source, name):
    """Build a submission from zone predictions."""
    sub = pd.DataFrame({"event_id": test.event_id})
    for h in HORIZONS:
        col = f"prob_{h}h"
        sub[col] = 0.0
        sub.loc[te_far, col] = far_p
        sub.loc[te_active, col] = active_p

        if isinstance(static_source, dict):
            sub.loc[te_static, col] = static_source[h]
        else:
            sub.loc[te_static, col] = static_source.loc[te_static, col].values

    # Monotonicity enforcement
    for idx in range(len(sub)):
        prev = 0
        for col in HORIZON_COLS:
            if sub.loc[idx, col] < prev:
                sub.loc[idx, col] = prev
            prev = sub.loc[idx, col]

    # Safe clipping
    for col in HORIZON_COLS:
        sub[col] = sub[col].clip(0.001, 0.999)

    mono_v = sum(1 for i in range(95) if not all(
        sub.iloc[i][HORIZON_COLS[j]] <= sub.iloc[i][HORIZON_COLS[j + 1]]
        for j in range(3)))

    sub.to_csv(f"submission_{name}.csv", index=False)
    print(f"\n{name}: mono={mono_v}")
    for col in HORIZON_COLS:
        s = sub.loc[te_static, col]
        a = sub.loc[te_active, col]
        f = sub.loc[te_far, col]
        print(f"  {col}: far={f.mean():.4f} active={a.mean():.4f} static={s.mean():.4f}")
    return sub

# --- Variant A: Pure 3-zone ML ensemble ---
s_a = assemble(0.001, 0.999, static_preds, "v21_A")

# --- Variant B: 3-zone with static = 50% ML + 50% v18_BLEND ---
blend_50 = {}
for h in HORIZONS:
    col = f"prob_{h}h"
    ours = static_preds[h]
    theirs = v18b.loc[te_static, col].values
    blend_50[h] = 0.5 * ours + 0.5 * theirs
s_b = assemble(0.001, 0.999, blend_50, "v21_B")

# --- Variant C: 3-zone with static = 40% ML + 60% h_blend ---
# h_blend is proven on LB; heavier weight on its rankings
blend_60hb = {}
for h in HORIZONS:
    col = f"prob_{h}h"
    ours = static_preds[h]
    theirs = hblend.loc[te_static, col].values
    blend_60hb[h] = 0.4 * ours + 0.6 * theirs
s_c = assemble(0.001, 0.999, blend_60hb, "v21_C")

# ============================================================
# CROSS-FILE VALIDATION
# ============================================================
print("\n" + "=" * 70)
print("CROSS-VALIDATION: which variant is best?")
print("=" * 70)

# Compare to v18_BLEND (LB=0.97216) - what's different
for name, sub_df in [("v21_A", s_a), ("v21_B", s_b), ("v21_C", s_c)]:
    diffs = []
    for col in HORIZON_COLS:
        d = (sub_df[col] - v18b[col]).abs().mean()
        diffs.append(d)
    active_diff = sum((sub_df.loc[te_active, col] - v18b.loc[te_active, col]).abs().mean()
                      for col in HORIZON_COLS)
    print(f"  {name} vs v18_BLEND: mean_diff={np.mean(diffs):.5f}, active_zone_diff={active_diff:.4f}")

print("\n" + "=" * 70)
print("RECOMMENDATION:")
print("  v21_B = safest (50/50 blend preserves LB-validated signal)")
print("  v21_A = highest potential (pure ML, most different from v18)")
print("  v21_C = conservative (60% h_blend weight)")
print("=" * 70)
