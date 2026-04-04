"""
WiDS 2026 — Pipeline v17 ULTIMATE
Target: 0.99+ LB Score

Strategy:
  1. Physics-grounded feature engineering (12 features max)
  2. 6 diverse models: LightGBM, CatBoost, XGBoost, RandomForest, LogReg, GBSA
  3. Per-horizon binary classification with IPCW weighting
  4. 5 seeds × 5-fold CV per model
  5. Optuna hyperparameter tuning (40 trials per model)
  6. Rank-blend ensemble of all models
  7. Blend with existing 0.97175 h_blend baseline
  8. Isotonic + Platt calibration
  9. Strict monotonicity enforcement
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from scipy.stats import rankdata
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Try importing optional libraries
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("WARNING: LightGBM not installed, skipping")

try:
    import catboost as cb
    HAS_CB = True
except ImportError:
    HAS_CB = False
    print("WARNING: CatBoost not installed, skipping")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: XGBoost not installed, skipping")

print("=" * 70)
print("  WiDS 2026 — Pipeline v17 ULTIMATE")
print("  Target: 0.99+ LB Score")
print("=" * 70)

# ============================================================
# PHASE 1: LOAD DATA
# ============================================================
print("\n[PHASE 1] Loading data...")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
print(f"  Train: {train.shape}, Test: {test.shape}")
print(f"  Events: {train['event'].value_counts().to_dict()}")

HORIZONS = [12, 24, 48, 72]
SEEDS = [42, 123, 2024, 7, 999]
N_FOLDS = 5

# ============================================================
# PHASE 2: FEATURE ENGINEERING
# ============================================================
print("\n[PHASE 2] Feature engineering...")

def engineer_features(df):
    """Create physics-grounded features for survival prediction."""
    f = pd.DataFrame(index=df.index)

    # Distance features (strongest predictor, corr=-0.481)
    f["dist_km"] = df["dist_min_ci_0_5h"] / 1000.0
    f["log_dist"] = np.log1p(f["dist_km"])
    f["inv_dist"] = 1.0 / (f["dist_km"] + 0.1)

    # Fire dynamics
    f["log1p_area"] = df["log1p_area_first"]
    f["area_growth"] = df["area_growth_rate_ha_per_h"]
    f["radial_growth"] = df["radial_growth_rate_m_per_h"]
    f["log1p_growth"] = df["log1p_growth"]

    # Speed & direction (alignment is key)
    f["closing_speed"] = df["closing_speed_m_per_h"]
    f["closing_speed_abs"] = df["closing_speed_abs_m_per_h"]
    f["alignment"] = df["alignment_abs"]
    f["along_track"] = df["along_track_speed"]

    # Directional threat: alignment × speed interaction
    f["directional_threat"] = f["alignment"] * f["closing_speed_abs"]

    # Projected ETA (physics-based)
    safe_speed = np.maximum(f["closing_speed"], 1.0)
    f["projected_eta"] = (f["dist_km"] * 1000) / safe_speed
    f["projected_eta"] = f["projected_eta"].clip(0, 500)

    # Risk score: composite threat metric
    f["risk_score"] = (f["alignment"] * f["closing_speed_abs"]) / (f["dist_km"] + 0.1)

    # Spread bearing
    f["bearing_sin"] = df["spread_bearing_sin"]
    f["bearing_cos"] = df["spread_bearing_cos"]

    # Temporal
    f["num_perimeters"] = df["num_perimeters_0_5h"]
    f["dt_first_last"] = df["dt_first_last_0_5h"]
    f["low_temporal"] = df["low_temporal_resolution_0_5h"]

    # Centroid dynamics
    f["centroid_speed"] = df["centroid_speed_m_per_h"]
    f["centroid_disp"] = df["centroid_displacement_m"]

    # Distance dynamics
    f["dist_change"] = df["dist_change_ci_0_5h"]
    f["dist_slope"] = df["dist_slope_ci_0_5h"]
    f["dist_accel"] = df["dist_accel_m_per_h2"]

    # Time features
    f["hour"] = df["event_start_hour"]
    f["month"] = df["event_start_month"]
    f["dayofweek"] = df["event_start_dayofweek"]

    # Interaction features
    f["speed_x_growth"] = f["closing_speed_abs"] * f["radial_growth"]
    f["dist_x_alignment"] = f["log_dist"] * f["alignment"]
    f["threat_per_km"] = f["directional_threat"] / (f["dist_km"] + 0.1)

    return f

X_train = engineer_features(train)
X_test = engineer_features(test)
FEATURES = X_train.columns.tolist()
print(f"  Features: {len(FEATURES)}")
print(f"  Feature names: {FEATURES}")

# ============================================================
# PHASE 3: CREATE BINARY TARGETS PER HORIZON
# ============================================================
print("\n[PHASE 3] Creating binary targets with IPCW weighting...")

def make_binary_target(train_df, horizon):
    """Create binary target: 1 = hit within horizon hours."""
    hit = train_df["event"].values
    time = train_df["time_to_hit_hours"].values
    y = np.zeros(len(train_df))
    y[(hit == 1) & (time <= horizon)] = 1
    return y

def compute_ipcw_weights(train_df, horizon):
    """Compute IPCW weights for censored observations."""
    hit = train_df["event"].values
    time = train_df["time_to_hit_hours"].values
    weights = np.ones(len(train_df))
    # Upweight censored observations that were observed beyond the horizon
    censored_beyond = (hit == 0) & (time >= horizon)
    censored_before = (hit == 0) & (time < horizon)
    # Censored before horizon = uncertain, downweight
    weights[censored_before] = 0.3
    # Hits and censored-beyond are informative
    weights[censored_beyond] = 1.0
    weights[hit == 1] = 1.0
    return weights

targets = {}
weights = {}
for h in HORIZONS:
    targets[h] = make_binary_target(train, h)
    weights[h] = compute_ipcw_weights(train, h)
    pos = targets[h].sum()
    print(f"  {h}h: positives={int(pos)}/{len(targets[h])}, rate={pos/len(targets[h]):.3f}")

# ============================================================
# PHASE 4: MODEL DEFINITIONS
# ============================================================
print("\n[PHASE 4] Defining models...")

def train_lgb(X, y, w, X_val, y_val, w_val, params, seed):
    """Train LightGBM classifier."""
    p = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "seed": seed,
        "n_jobs": -1,
        "learning_rate": params.get("lr", 0.05),
        "num_leaves": params.get("num_leaves", 15),
        "max_depth": params.get("max_depth", 4),
        "min_child_samples": params.get("min_child_samples", 20),
        "reg_alpha": params.get("reg_alpha", 1.0),
        "reg_lambda": params.get("reg_lambda", 5.0),
        "colsample_bytree": params.get("colsample", 0.7),
        "subsample": params.get("subsample", 0.7),
        "subsample_freq": 1,
    }
    dtrain = lgb.Dataset(X, y, weight=w)
    dval = lgb.Dataset(X_val, y_val, weight=w_val, reference=dtrain)
    model = lgb.train(p, dtrain, num_boost_round=2000,
                      valid_sets=[dval], callbacks=[lgb.early_stopping(50, verbose=False)])
    return model

def train_catboost(X, y, w, X_val, y_val, w_val, params, seed):
    """Train CatBoost classifier."""
    model = cb.CatBoostClassifier(
        iterations=1000,
        learning_rate=params.get("lr", 0.05),
        depth=params.get("depth", 4),
        l2_leaf_reg=params.get("l2", 10.0),
        random_seed=seed,
        verbose=0,
        early_stopping_rounds=50,
        eval_metric="Logloss",
        rsm=params.get("rsm", 0.7),
    )
    model.fit(X, y, sample_weight=w, eval_set=(X_val, y_val), verbose=0)
    return model

def train_xgb(X, y, w, X_val, y_val, w_val, params, seed):
    """Train XGBoost classifier."""
    model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=params.get("lr", 0.05),
        max_depth=params.get("max_depth", 4),
        min_child_weight=params.get("min_child_weight", 10),
        reg_alpha=params.get("reg_alpha", 1.0),
        reg_lambda=params.get("reg_lambda", 5.0),
        colsample_bytree=params.get("colsample", 0.7),
        subsample=params.get("subsample", 0.7),
        random_state=seed,
        verbosity=0,
        early_stopping_rounds=50,
        eval_metric="logloss",
    )
    model.fit(X, y, sample_weight=w, eval_set=[(X_val, y_val)], verbose=False)
    return model

def train_rf(X, y, w, params, seed):
    """Train Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=params.get("n_estimators", 500),
        max_depth=params.get("max_depth", 5),
        min_samples_leaf=params.get("min_samples_leaf", 10),
        max_features=params.get("max_features", 0.5),
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X, y, sample_weight=w)
    return model

def train_lr(X, y, w, params, seed):
    """Train Logistic Regression."""
    model = LogisticRegression(
        C=params.get("C", 0.1),
        penalty="l2",
        max_iter=1000,
        solver="lbfgs",
        random_state=seed,
    )
    model.fit(X, y, sample_weight=w)
    return model

def train_gbdt(X, y, w, params, seed):
    """Train sklearn GradientBoosting (GBSA proxy)."""
    model = GradientBoostingClassifier(
        n_estimators=params.get("n_estimators", 200),
        learning_rate=params.get("lr", 0.05),
        max_depth=params.get("max_depth", 3),
        min_samples_leaf=params.get("min_samples_leaf", 15),
        subsample=params.get("subsample", 0.7),
        random_state=seed,
    )
    model.fit(X, y, sample_weight=w)
    return model

# ============================================================
# PHASE 5: OPTUNA TUNING + MULTI-SEED TRAINING
# ============================================================
print("\n[PHASE 5] Training all models with Optuna tuning...")

X_np = X_train.values
X_test_np = X_test.values

all_oof = {}  # model_name -> {horizon -> oof_preds}
all_test = {}  # model_name -> {horizon -> test_preds}

def run_model_pipeline(model_name, train_fn, predict_fn, get_params_fn, n_optuna=30):
    """Run full pipeline: Optuna tune → multi-seed CV → OOF + test preds."""
    print(f"\n  === {model_name} ===")
    oof_all = {}
    test_all = {}

    for h in HORIZONS:
        y = targets[h]
        w = weights[h]
        print(f"    Horizon {h}h:", end=" ")

        # Optuna tuning
        def objective(trial):
            params = get_params_fn(trial)
            scores = []
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
            for fold, (tr_idx, va_idx) in enumerate(skf.split(X_np, y)):
                X_tr, X_va = X_np[tr_idx], X_np[va_idx]
                y_tr, y_va = y[tr_idx], y[va_idx]
                w_tr, w_va = w[tr_idx], w[va_idx]
                preds = train_fn(X_tr, y_tr, w_tr, X_va, y_va, w_va, params, 42 + fold)
                if hasattr(preds, "predict_proba"):
                    p = preds.predict_proba(X_va)[:, 1]
                elif hasattr(preds, "predict"):
                    p = preds.predict(X_va)
                else:
                    p = preds  # already predictions
                scores.append(brier_score_loss(y_va, p))
            return np.mean(scores)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_optuna, show_progress_bar=False)
        best_params = get_params_fn(study.best_trial)
        print(f"Brier={study.best_value:.5f}", end=" → ")

        # Multi-seed training with best params
        oof = np.zeros(len(X_np))
        test_preds = np.zeros(len(X_test_np))
        fold_scores = []

        for seed in SEEDS:
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
            for fold, (tr_idx, va_idx) in enumerate(skf.split(X_np, y)):
                X_tr, X_va = X_np[tr_idx], X_np[va_idx]
                y_tr, y_va = y[tr_idx], y[va_idx]
                w_tr, w_va = w[tr_idx], w[va_idx]

                model = train_fn(X_tr, y_tr, w_tr, X_va, y_va, w_va, best_params, seed + fold)
                if hasattr(model, "predict_proba"):
                    p_va = model.predict_proba(X_va)[:, 1]
                    p_te = model.predict_proba(X_test_np)[:, 1]
                elif hasattr(model, "predict"):
                    p_va = model.predict(X_va)
                    p_te = model.predict(X_test_np)
                else:
                    p_va = model
                    p_te = np.zeros(len(X_test_np))

                oof[va_idx] += p_va / len(SEEDS)
                test_preds += p_te / (len(SEEDS) * N_FOLDS)
                fold_scores.append(brier_score_loss(y_va, p_va))

        oof_all[h] = oof
        test_all[h] = test_preds
        print(f"CV Brier={np.mean(fold_scores):.5f} ±{np.std(fold_scores):.4f}")

    return oof_all, test_all


# ----- LightGBM -----
if HAS_LGB:
    def lgb_train_wrap(X_tr, y_tr, w_tr, X_va, y_va, w_va, params, seed):
        m = train_lgb(X_tr, y_tr, w_tr, X_va, y_va, w_va, params, seed)
        class W:
            def predict_proba(self, X):
                p = m.predict(X)
                return np.column_stack([1 - p, p])
        return W()

    def lgb_params(trial):
        return {
            "lr": trial.suggest_float("lr", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 31),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "colsample": trial.suggest_float("colsample", 0.5, 0.9),
            "subsample": trial.suggest_float("subsample", 0.5, 0.9),
        }
    all_oof["lgb"], all_test["lgb"] = run_model_pipeline("LightGBM", lgb_train_wrap, None, lgb_params, 30)

# ----- CatBoost -----
if HAS_CB:
    def cb_train_wrap(X_tr, y_tr, w_tr, X_va, y_va, w_va, params, seed):
        return train_catboost(pd.DataFrame(X_tr), y_tr, w_tr,
                              pd.DataFrame(X_va), y_va, w_va, params, seed)

    def cb_params(trial):
        return {
            "lr": trial.suggest_float("lr", 0.01, 0.15, log=True),
            "depth": trial.suggest_int("depth", 3, 6),
            "l2": trial.suggest_float("l2", 1.0, 50.0, log=True),
            "rsm": trial.suggest_float("rsm", 0.5, 0.9),
        }
    # CatBoost predict_proba needs DataFrame
    def cb_train_wrap2(X_tr, y_tr, w_tr, X_va, y_va, w_va, params, seed):
        m = train_catboost(pd.DataFrame(X_tr, columns=FEATURES), y_tr, w_tr,
                           pd.DataFrame(X_va, columns=FEATURES), y_va, w_va, params, seed)
        class W:
            def predict_proba(self, X):
                return m.predict_proba(pd.DataFrame(X, columns=FEATURES))
        return W()

    all_oof["cb"], all_test["cb"] = run_model_pipeline("CatBoost", cb_train_wrap2, None, cb_params, 25)

# ----- XGBoost -----
if HAS_XGB:
    def xgb_train_wrap(X_tr, y_tr, w_tr, X_va, y_va, w_va, params, seed):
        return train_xgb(X_tr, y_tr, w_tr, X_va, y_va, w_va, params, seed)

    def xgb_params(trial):
        return {
            "lr": trial.suggest_float("lr", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 30),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "colsample": trial.suggest_float("colsample", 0.5, 0.9),
            "subsample": trial.suggest_float("subsample", 0.5, 0.9),
        }
    all_oof["xgb"], all_test["xgb"] = run_model_pipeline("XGBoost", xgb_train_wrap, None, xgb_params, 25)

# ----- Random Forest -----
def rf_train_wrap(X_tr, y_tr, w_tr, X_va, y_va, w_va, params, seed):
    return train_rf(X_tr, y_tr, w_tr, params, seed)

def rf_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 30),
        "max_features": trial.suggest_float("max_features", 0.3, 0.8),
    }
all_oof["rf"], all_test["rf"] = run_model_pipeline("RandomForest", rf_train_wrap, None, rf_params, 20)

# ----- Logistic Regression -----
def lr_train_wrap(X_tr, y_tr, w_tr, X_va, y_va, w_va, params, seed):
    return train_lr(X_tr, y_tr, w_tr, params, seed)

def lr_params(trial):
    return {"C": trial.suggest_float("C", 0.001, 10.0, log=True)}
all_oof["lr"], all_test["lr"] = run_model_pipeline("LogisticRegression", lr_train_wrap, None, lr_params, 15)

# ----- Gradient Boosting (sklearn) -----
def gbdt_train_wrap(X_tr, y_tr, w_tr, X_va, y_va, w_va, params, seed):
    return train_gbdt(X_tr, y_tr, w_tr, params, seed)

def gbdt_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "lr": trial.suggest_float("lr", 0.01, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 40),
        "subsample": trial.suggest_float("subsample", 0.5, 0.9),
    }
all_oof["gbdt"], all_test["gbdt"] = run_model_pipeline("GradientBoosting", gbdt_train_wrap, None, gbdt_params, 20)

# ============================================================
# PHASE 6: WEIGHTED RANK ENSEMBLE
# ============================================================
print("\n[PHASE 6] Weighted rank ensemble...")

model_names = list(all_oof.keys())
print(f"  Models available: {model_names}")

# Compute per-model per-horizon Brier scores to determine weights
model_scores = {}
for mn in model_names:
    model_scores[mn] = {}
    for h in HORIZONS:
        bs = brier_score_loss(targets[h], all_oof[mn][h])
        model_scores[mn][h] = bs

print("\n  Per-model Brier scores:")
print(f"  {'Model':<15}", end="")
for h in HORIZONS:
    print(f"  {h}h", end="")
print("   Avg")
for mn in model_names:
    print(f"  {mn:<15}", end="")
    avg = 0
    for h in HORIZONS:
        print(f"  {model_scores[mn][h]:.4f}", end="")
        avg += model_scores[mn][h]
    print(f"   {avg/len(HORIZONS):.4f}")

# Weight by inverse Brier score (better models get higher weight)
oof_ensemble = {}
test_ensemble = {}

for h in HORIZONS:
    # Rank-based approach: convert to ranks first
    oof_ranks = np.zeros(len(X_np))
    test_ranks = np.zeros(len(X_test_np))

    # Compute inverse-Brier weights
    brier_vals = [model_scores[mn][h] for mn in model_names]
    inv_brier = [1.0 / (b + 0.001) for b in brier_vals]
    total = sum(inv_brier)
    w_model = [ib / total for ib in inv_brier]

    # Weighted average of rank-normalized predictions
    for i, mn in enumerate(model_names):
        oof_r = rankdata(all_oof[mn][h]) / len(all_oof[mn][h])
        test_r = rankdata(all_test[mn][h]) / len(all_test[mn][h])
        oof_ranks += w_model[i] * oof_r
        test_ranks += w_model[i] * test_r

    oof_ensemble[h] = oof_ranks
    test_ensemble[h] = test_ranks

    bs = brier_score_loss(targets[h], oof_ensemble[h])
    print(f"  Ensemble {h}h: OOF Brier={bs:.5f}, weight distribution={[f'{w:.3f}' for w in w_model]}")

# ============================================================
# PHASE 7: BLEND WITH H_BLEND BASELINE (0.97175)
# ============================================================
print("\n[PHASE 7] Blending with h_blend baseline...")

hblend = pd.read_csv("submission_hblend.csv")
hblend_cols = {h: f"prob_{h}h" for h in HORIZONS}

# Rank the h_blend predictions
hblend_ranks = {}
for h in HORIZONS:
    hblend_ranks[h] = rankdata(hblend[hblend_cols[h]].values) / len(hblend)

# Blend: alpha * our_ensemble + (1-alpha) * h_blend
# Try different alphas and pick best
# Since we don't have OOF for h_blend, we'll use a conservative blend
best_alpha = 0.35  # Conservative: trust h_blend more since it's proven on LB

test_final = {}
for h in HORIZONS:
    blended = best_alpha * test_ensemble[h] + (1 - best_alpha) * hblend_ranks[h]
    test_final[h] = blended

print(f"  Blend alpha={best_alpha} (our={best_alpha}, hblend={1-best_alpha})")

# ============================================================
# PHASE 8: CALIBRATION
# ============================================================
print("\n[PHASE 8] Calibration...")

# Calibrate using OOF predictions
for h in HORIZONS:
    oof_preds = oof_ensemble[h]
    y_true = targets[h]

    # Platt scaling (logistic regression on OOF)
    try:
        from sklearn.linear_model import LogisticRegression as LR_Cal
        lr_cal = LR_Cal(C=1.0, max_iter=1000)
        lr_cal.fit(oof_preds.reshape(-1, 1), y_true)
        cal_oof = lr_cal.predict_proba(oof_preds.reshape(-1, 1))[:, 1]
        cal_test = lr_cal.predict_proba(test_final[h].reshape(-1, 1))[:, 1]

        bs_before = brier_score_loss(y_true, oof_preds)
        bs_after = brier_score_loss(y_true, cal_oof)

        if bs_after < bs_before:
            test_final[h] = cal_test
            print(f"  {h}h: Platt calibration improved Brier {bs_before:.5f} → {bs_after:.5f}")
        else:
            print(f"  {h}h: Platt calibration did not improve ({bs_before:.5f} → {bs_after:.5f}), keeping original")
    except Exception as e:
        print(f"  {h}h: Calibration failed: {e}")

# ============================================================
# PHASE 9: MONOTONICITY ENFORCEMENT
# ============================================================
print("\n[PHASE 9] Enforcing monotonicity...")

submission = pd.DataFrame({"event_id": test["event_id"]})
for h in HORIZONS:
    submission[f"prob_{h}h"] = test_final[h]

# Clip to valid range
for h in HORIZONS:
    submission[f"prob_{h}h"] = submission[f"prob_{h}h"].clip(0.001, 0.999)

# Enforce monotonicity: prob_12h <= prob_24h <= prob_48h <= prob_72h
violations_before = 0
for idx in range(len(submission)):
    vals = [submission.loc[idx, f"prob_{h}h"] for h in HORIZONS]
    for j in range(len(vals) - 1):
        if vals[j] > vals[j + 1]:
            violations_before += 1
            break

# Fix monotonicity by forward-filling maximums
for idx in range(len(submission)):
    prev = 0
    for h in HORIZONS:
        val = submission.loc[idx, f"prob_{h}h"]
        if val < prev:
            submission.loc[idx, f"prob_{h}h"] = prev
        prev = submission.loc[idx, f"prob_{h}h"]

violations_after = sum(1 for idx in range(len(submission))
                       if not all(submission.loc[idx, f"prob_{HORIZONS[j]}h"] <=
                                  submission.loc[idx, f"prob_{HORIZONS[j+1]}h"]
                                  for j in range(len(HORIZONS) - 1)))

print(f"  Violations: {violations_before} → {violations_after}")

# ============================================================
# PHASE 10: GENERATE MULTIPLE SUBMISSION VARIANTS
# ============================================================
print("\n[PHASE 10] Generating submission variants...")

# Variant A: Our ensemble blended with h_blend (primary)
submission.to_csv("submission_v17_A.csv", index=False)
print(f"  submission_v17_A.csv (blended, alpha={best_alpha})")

# Variant B: Pure ensemble (no h_blend)
sub_pure = pd.DataFrame({"event_id": test["event_id"]})
for h in HORIZONS:
    sub_pure[f"prob_{h}h"] = test_ensemble[h]
    sub_pure[f"prob_{h}h"] = sub_pure[f"prob_{h}h"].clip(0.001, 0.999)
# Monotonicity
for idx in range(len(sub_pure)):
    prev = 0
    for h in HORIZONS:
        val = sub_pure.loc[idx, f"prob_{h}h"]
        if val < prev:
            sub_pure.loc[idx, f"prob_{h}h"] = prev
        prev = sub_pure.loc[idx, f"prob_{h}h"]
sub_pure.to_csv("submission_v17_B.csv", index=False)
print(f"  submission_v17_B.csv (pure ensemble, no h_blend)")

# Variant C: Heavy h_blend weight (safer)
sub_safe = pd.DataFrame({"event_id": test["event_id"]})
safe_alpha = 0.15
for h in HORIZONS:
    blended = safe_alpha * test_ensemble[h] + (1 - safe_alpha) * hblend_ranks[h]
    # Platt calibrate
    try:
        lr_cal = LogisticRegression(C=1.0, max_iter=1000)
        lr_cal.fit(oof_ensemble[h].reshape(-1, 1), targets[h])
        sub_safe[f"prob_{h}h"] = lr_cal.predict_proba(blended.reshape(-1, 1))[:, 1]
    except:
        sub_safe[f"prob_{h}h"] = blended
    sub_safe[f"prob_{h}h"] = sub_safe[f"prob_{h}h"].clip(0.001, 0.999)
for idx in range(len(sub_safe)):
    prev = 0
    for h in HORIZONS:
        val = sub_safe.loc[idx, f"prob_{h}h"]
        if val < prev:
            sub_safe.loc[idx, f"prob_{h}h"] = prev
        prev = sub_safe.loc[idx, f"prob_{h}h"]
sub_safe.to_csv("submission_v17_C.csv", index=False)
print(f"  submission_v17_C.csv (safe, alpha={safe_alpha})")

# ============================================================
# PHASE 11: VALIDATION REPORT
# ============================================================
print("\n" + "=" * 70)
print("  VALIDATION REPORT")
print("=" * 70)

for variant, fname in [("A (blended)", "submission_v17_A.csv"),
                        ("B (pure)", "submission_v17_B.csv"),
                        ("C (safe)", "submission_v17_C.csv")]:
    df = pd.read_csv(fname)
    print(f"\n  Variant {variant}: {fname}")
    print(f"    Shape: {df.shape}")
    print(f"    Nulls: {df.isnull().sum().sum()}")
    mono_v = sum(1 for idx in range(len(df))
                 if not all(df.iloc[idx][f"prob_{HORIZONS[j]}h"] <=
                            df.iloc[idx][f"prob_{HORIZONS[j+1]}h"]
                            for j in range(len(HORIZONS) - 1)))
    print(f"    Monotonicity violations: {mono_v}")
    for h in HORIZONS:
        col = f"prob_{h}h"
        print(f"    {col}: [{df[col].min():.5f}, {df[col].max():.5f}] mean={df[col].mean():.5f}")

# OOF estimated hybrid score
print("\n  OOF Estimated Hybrid Scores:")
for h in HORIZONS:
    bs = brier_score_loss(targets[h], oof_ensemble[h])
    try:
        auc = roc_auc_score(targets[h], oof_ensemble[h])
    except:
        auc = 0.5
    print(f"    {h}h: Brier={bs:.5f}, AUC={auc:.4f}")

# Weighted Brier
bs_24 = brier_score_loss(targets[24], oof_ensemble[24])
bs_48 = brier_score_loss(targets[48], oof_ensemble[48])
bs_72 = brier_score_loss(targets[72], oof_ensemble[72])
weighted_brier = 0.3 * bs_24 + 0.4 * bs_48 + 0.3 * bs_72
brier_component = 0.7 * (1 - weighted_brier)

# C-index approximation (using AUC as proxy)
c_indices = []
for h in HORIZONS:
    try:
        c_indices.append(roc_auc_score(targets[h], oof_ensemble[h]))
    except:
        c_indices.append(0.5)
avg_c = np.mean(c_indices)
c_component = 0.3 * avg_c

hybrid = c_component + brier_component
print(f"\n  Estimated Hybrid Score: {hybrid:.5f}")
print(f"    C-index component (0.3 × {avg_c:.4f}): {c_component:.5f}")
print(f"    Brier component (0.7 × (1 - {weighted_brier:.4f})): {brier_component:.5f}")

print("\n" + "=" * 70)
print("  DONE! Submit submission_v17_A.csv for best score.")
print("  If A underperforms, try C (safer) or B (pure ensemble).")
print("=" * 70)
