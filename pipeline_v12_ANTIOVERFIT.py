"""
pipeline_v12_ANTIOVERFIT.py — Extreme Simplicity & Regularization
WiDS Global Datathon 2026: Wildfire Survival Analysis

Diagnosis: EPV (Events per Variable) was 2.76. We need > 7 to prevent overfit.
Action: Limit to exact 10 physics-based features. 
Models: GBSA, ElasticNet CoxPH, CatBoost. 
Calibration: No complex Isotonic Regression.
"""

import warnings
warnings.filterwarnings('ignore')
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.special import logit, expit
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, roc_auc_score

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from catboost import CatBoostClassifier
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from lifelines import CoxPHFitter

start_time = time.time()

# ============================================================
# CONFIGURATION
# ============================================================
TIME_HORIZONS = [12, 24, 48, 72]
SEEDS = [42, 52, 62, 72, 82, 92, 102, 112, 122, 132]
N_SPLITS = 5
N_TRIALS = 30
MONO_INC = 0.003
CLIP_LO, CLIP_HI = 0.008, 0.992

# ============================================================
# PHASE 1: DATA PIPELINE (EPV > 7)
# ============================================================
print("=" * 70)
print("WiDS 2026 — Pipeline v12 ANTI-OVERFIT")
print("=" * 70)

train = pd.read_csv('d:/WiDS/train.csv')
test = pd.read_csv('d:/WiDS/test.csv')
sample = pd.read_csv('d:/WiDS/sample_submission.csv')

def engineer_features(df):
    out = pd.DataFrame(index=df.index)
    
    # 1. Minimum distance to fire
    out['dist_min_ci'] = df['dist_min_ci_0_5h'].clip(lower=0)
    
    # 2. Log distance (linearizes extreme distances)
    out['log_dist'] = np.log1p(out['dist_min_ci'])
    
    # 3. Alignment (is fire heading directly at us?)
    out['alignment_abs'] = df['alignment_abs']
    
    # 4. Data richness / threat scope proxy
    out['num_perimeters'] = df['num_perimeters_0_5h']
    
    # 5. Raw speed
    safe_speed = df['closing_speed_m_per_h'].clip(lower=0.001)
    out['closing_speed'] = df['closing_speed_m_per_h']
    
    # 6. Projected time to hit (Distance / Speed)
    out['projected_time'] = (df['dist_min_ci_0_5h'] / safe_speed).clip(0, 500)
    
    # 7. Near Miss Margin (Geometry)
    out['near_miss_margin'] = (
        df['dist_min_ci_0_5h'] - df['radial_growth_m'] - df['projected_advance_m'].clip(lower=0)
    )
    
    # 8. Threat Gravity (Momentum)
    out['threat_gravity'] = (df['alignment_abs'] * df['closing_speed_m_per_h']) / ((df['dist_min_ci_0_5h'] + 100)**2)
    
    # 9. Wavefront ETA
    out['wavefront_eta'] = (df['dist_min_ci_0_5h'] - df['radial_growth_m']) / (df['closing_speed_m_per_h'] + df['radial_growth_rate_m_per_h']).clip(lower=0.001)
    
    # 10. Multiplicative Risk Composite
    dist_max = df['dist_min_ci_0_5h'].max() + 1
    speed_max = df['closing_speed_m_per_h'].abs().max() + 1
    out['risk_score'] = (1.0 - df['dist_min_ci_0_5h']/dist_max) * (df['closing_speed_m_per_h']/speed_max).clip(0,1) * df['alignment_abs']
    
    return out

X_train_raw = engineer_features(train)
X_test_raw = engineer_features(test)

FEATURES = list(X_train_raw.columns)
print(f"Using exactly {len(FEATURES)} features. EPV = {69 / len(FEATURES):.2f}")

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# Target matrices
y_surv = np.array(
    [(bool(row['event'] == 1), row['time_to_hit_hours']) for _, row in train.iterrows()],
    dtype=[('event', bool), ('time', float)]
)

train_df_cph = pd.DataFrame(X_train, columns=FEATURES)
train_df_cph['event'] = train['event']
train_df_cph['time'] = train['time_to_hit_hours']

test_df_cph = pd.DataFrame(X_test, columns=FEATURES)

# ============================================================
# PHASE 2: IPCW & BINARY SETUP
# ============================================================
def compute_ipcw(train_df, horizon):
    n = len(train_df)
    labels, weights, mask = np.zeros(n), np.ones(n), np.ones(n, dtype=bool)
    for i in range(n):
        ev = train_df.iloc[i]['event']
        t = train_df.iloc[i]['time_to_hit_hours']
        if ev == 1 and t <= horizon:
            labels[i] = 1
        elif ev == 1 and t > horizon:
            labels[i] = 0
        elif ev == 0:
            if t >= horizon:
                labels[i] = 0
            elif t >= horizon * 0.7:
                labels[i] = 0
                weights[i] = t / horizon
            else:
                mask[i] = False
    weights = np.clip(weights, 0.1, 3.0)
    if mask.sum() > 0:
        weights[mask] /= weights[mask].mean()
    return labels, weights, mask

ipcw_cache = {h: compute_ipcw(train, h) for h in TIME_HORIZONS}

# ============================================================
# PHASE 3: MODEL TRAINING (GBSA, CoxPH, CatBoost)
# ============================================================
print("\n" + "=" * 70)
print("PHASE 3: Model Training (10 Seeds x 5 Folds)")
print("=" * 70)

preds_oof = {m: {h: np.zeros(len(train)) for h in TIME_HORIZONS} for m in ['gbsa', 'coxph', 'catboost']}
preds_test = {m: {h: np.zeros(len(test)) for h in TIME_HORIZONS} for m in ['gbsa', 'coxph', 'catboost']}

# --- 1. GBSA (Optimized) ---
print("\n  [1/3] Training GBSA...")
def gbsa_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 250),
        'learning_rate': trial.suggest_float('lr', 0.01, 0.1),
        'max_depth': trial.suggest_int('depth', 2, 3),
        'min_samples_leaf': trial.suggest_int('min_leaf', 10, 30),
        'dropout_rate': trial.suggest_float('dropout', 0.0, 0.3),
    }
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    brier_sum = 0
    for tr_idx, val_idx in kf.split(X_train):
        mdl = GradientBoostingSurvivalAnalysis(**params, random_state=42)
        mdl.fit(X_train[tr_idx], y_surv[tr_idx])
        surv_fns = mdl.predict_survival_function(X_train[val_idx])
        for i, fn in enumerate(surv_fns):
            def get_p(h):
                if h < fn.x.min(): return 0.0
                return 1.0 - fn(min(h, fn.x.max()))
            p24, p48, p72 = get_p(24), get_p(48), get_p(72)
            y24 = 1 if train.iloc[val_idx[i]]['event']==1 and train.iloc[val_idx[i]]['time_to_hit_hours']<=24 else 0
            y48 = 1 if train.iloc[val_idx[i]]['event']==1 and train.iloc[val_idx[i]]['time_to_hit_hours']<=48 else 0
            y72 = 1 if train.iloc[val_idx[i]]['event']==1 and train.iloc[val_idx[i]]['time_to_hit_hours']<=72 else 0
            brier_sum += 0.3*(p24-y24)**2 + 0.4*(p48-y48)**2 + 0.3*(p72-y72)**2
    return brier_sum

study = optuna.create_study(direction='minimize')
study.optimize(gbsa_objective, n_trials=N_TRIALS)
gbsa_params = {
    'n_estimators': study.best_params['n_estimators'],
    'learning_rate': study.best_params['lr'],
    'max_depth': study.best_params['depth'],
    'min_samples_leaf': study.best_params['min_leaf'],
    'dropout_rate': study.best_params['dropout'],
}
print(f"    Best GBSA params: {gbsa_params}")

for seed in SEEDS:
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for tr_idx, val_idx in kf.split(X_train):
        mdl = GradientBoostingSurvivalAnalysis(**gbsa_params, random_state=seed)
        mdl.fit(X_train[tr_idx], y_surv[tr_idx])
        
        val_fns = mdl.predict_survival_function(X_train[val_idx])
        test_fns = mdl.predict_survival_function(X_test)
        
        for h in TIME_HORIZONS:
            for i, fn in enumerate(val_fns):
                v = 1.0 - fn(min(h, fn.x.max())) if h >= fn.x.min() else 0.0
                preds_oof['gbsa'][h][val_idx[i]] += v / len(SEEDS)
            for i, fn in enumerate(test_fns):
                v = 1.0 - fn(min(h, fn.x.max())) if h >= fn.x.min() else 0.0
                preds_test['gbsa'][h][i] += v / (len(SEEDS)*N_SPLITS)

# --- 2. CoxPH (ElasticNet Regularized) ---
print("\n  [2/3] Training Regularized CoxPH...")
for seed in SEEDS:
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for tr_idx, val_idx in kf.split(train_df_cph):
        # Heavy L1/L2 regularization
        cph = CoxPHFitter(penalizer=0.5, l1_ratio=0.5)
        cph.fit(train_df_cph.iloc[tr_idx], duration_col='time', event_col='event')
        
        val_surv = cph.predict_survival_function(train_df_cph.iloc[val_idx])
        test_surv = cph.predict_survival_function(test_df_cph)
        
        for h in TIME_HORIZONS:
            # Interpolate survival at time h
            if h in val_surv.index:
                vo = 1.0 - val_surv.loc[h].values
                to = 1.0 - test_surv.loc[h].values
            else:
                idx = val_surv.index.searchsorted(h)
                if idx == 0:
                    vo = np.zeros(len(val_idx))
                    to = np.zeros(len(test))
                else:
                    vo = 1.0 - val_surv.iloc[idx-1].values
                    to = 1.0 - test_surv.iloc[idx-1].values
                    
            preds_oof['coxph'][h][val_idx] += vo / len(SEEDS)
            preds_test['coxph'][h] += to / (len(SEEDS)*N_SPLITS)

# --- 3. CatBoost (Heavily Regularized Binary) ---
print("\n  [3/3] Training Extra-Regularized CatBoost...")
for h in TIME_HORIZONS:
    y_h, w_h, mask = ipcw_cache[h]
    X_f, y_f, w_f = X_train[mask], y_h[mask], w_h[mask]
    
    # 1. Very simple model configuration
    params = {
        'iterations': 300,
        'learning_rate': 0.03,
        'depth': 2,               # Brutal restriction
        'l2_leaf_reg': 25,        # Heavy L2
        'min_data_in_leaf': 15,   # Brutal restriction
        'verbose': 0
    }
    
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for tr_idx, val_idx in skf.split(X_f, y_f):
            mdl = CatBoostClassifier(**params, random_seed=seed)
            mdl.fit(X_f[tr_idx], y_f[tr_idx], sample_weight=w_f[tr_idx])
            preds_oof['catboost'][h][np.where(mask)[0][val_idx]] += mdl.predict_proba(X_f[val_idx])[:,1] / len(SEEDS)
            preds_test['catboost'][h] += mdl.predict_proba(X_test)[:,1] / (len(SEEDS)*N_SPLITS)

# ============================================================
# PHASE 4: BLENDING & CALIBRATION (NO ISO)
# ============================================================
print("\n" + "=" * 70)
print("PHASE 4: Blending (Test Distribution Optimization)")
print("=" * 70)

final_test = {h: np.zeros(len(test)) for h in TIME_HORIZONS}

for h in TIME_HORIZONS:
    # Full dataset binary targets for evaluation
    y_full = ((train['event'] == 1) & (train['time_to_hit_hours'] <= h)).astype(float).values
    
    model_scores = {}
    for m in ['gbsa', 'coxph', 'catboost']:
        # Note: CatBoost OOF is only computed on mask, need to pad with zeros
        # Actually gbsa and coxph are full length. Let's just use mask for evaluation.
        _, _, mask = ipcw_cache[h]
        if m == 'catboost':
            p = np.clip(preds_oof[m][h][mask], 0.001, 0.999)
            yf = y_full[mask]
        else:
            p = np.clip(preds_oof[m][h], 0.001, 0.999)
            yf = y_full
            
        brier = brier_score_loss(yf, p)
        auc = roc_auc_score(yf, p) if len(np.unique(yf)) > 1 else 0.5
        model_scores[m] = 0.3*auc + 0.7*(1-brier)
    
    # Softmax weighting with floor
    scores = np.array(list(model_scores.values()))
    w = np.exp(10 * (scores - scores.max()))
    w = np.maximum(w / w.sum(), 0.10)  # High floor (10%) for diversity
    w /= w.sum()
    
    models = list(model_scores.keys())
    for i, m in enumerate(models):
        final_test[h] += w[i] * np.clip(preds_test[m][h], CLIP_LO, CLIP_HI)
        
    print(f"  {h}h blends: " + " | ".join([f"{models[i]}: w={w[i]:.3f} (S={scores[i]:.4f})" for i in range(3)]))

# ============================================================
# PHASE 5: BASE RATE SHIFT & MONOTONICITY
# ============================================================
print("\n" + "=" * 70)
print("PHASE 5: Submission Assembly")
print("=" * 70)

for h in TIME_HORIZONS:
    y_full = ((train['event'] == 1) & (train['time_to_hit_hours'] <= h)).astype(float).values
    train_base = y_full.mean()
    test_base = final_test[h].mean()
    
    # Gentle sigmoid shift (25% strength)
    shift = logit(np.clip(train_base, 0.01, 0.99)) - logit(np.clip(test_base, 0.01, 0.99))
    final_test[h] = expit(logit(np.clip(final_test[h], 0.01, 0.99)) + 0.3 * shift)
    
    # Rank blend for extreme stability
    cal = final_test[h]
    rank = rankdata(cal) / len(cal)
    final_test[h] = 0.97 * cal + 0.03 * rank

# Enforce monotonicity
for i in range(len(test)):
    for j in range(1, len(TIME_HORIZONS)):
        prev, curr = TIME_HORIZONS[j-1], TIME_HORIZONS[j]
        if final_test[curr][i] < final_test[prev][i] + MONO_INC:
            final_test[curr][i] = final_test[prev][i] + MONO_INC

# Assemble
sub10 = sample.copy()
for h in TIME_HORIZONS:
    sub10[f'prob_{h}h'] = np.clip(final_test[h], CLIP_LO, CLIP_HI)
sub10.to_csv('d:/WiDS/submission_10.csv', index=False)

# CREATE THE LB HEDGE
sub09 = pd.read_csv('d:/WiDS/submission_09.csv')
sub_blend = sample.copy()
for h in TIME_HORIZONS:
    c = f'prob_{h}h'
    # 50% anti-overfit (v12) + 50% best LB baseline (v11)
    sub_blend[c] = 0.50 * sub10[c] + 0.50 * sub09[c]

# Re-enforce
for i in range(len(sub_blend)):
    for j in range(1, len(TIME_HORIZONS)):
        prev, curr = TIME_HORIZONS[j-1], TIME_HORIZONS[j]
        c_prev, c_curr = f'prob_{prev}h', f'prob_{curr}h'
        if sub_blend.at[i, c_curr] < sub_blend.at[i, c_prev] + MONO_INC:
            sub_blend.at[i, c_curr] = sub_blend.at[i, c_prev] + MONO_INC

sub_blend.to_csv('d:/WiDS/submission_10_blend.csv', index=False)

print("\nSaved submission_10.csv (pure v12)")
print("Saved submission_10_blend.csv (50% v12 + 50% LB 0.963 baseline) *** SUBMIT THIS ONE ***")

elapsed = time.time() - start_time
print(f"\nExecution time: {elapsed/60:.1f} minutes.")
