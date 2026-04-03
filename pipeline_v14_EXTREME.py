"""
pipeline_v14_EXTREME.py — The "Fire Is Real" Run
WiDS Global Datathon 2026: Wildfire Survival Analysis

STRATEGY:
1. Zero Hedging: 100% reliance on model + physical overrides. No safety blending.
2. Extreme Pseudo-Label Injection: Treat best previous predictions as unshakeable ground truth.
3. Unforgiving Physics Limits: If the math guarantees a hit or miss, enforce it to 0.995 / 0.005.
"""

import warnings
warnings.filterwarnings('ignore')
import os
import time
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler

start_time = time.time()
DATA_DIR = 'd:/WiDS'
TIME_HORIZONS = [12, 24, 48, 72]
SEEDS = [42, 1337, 2026, 777]

print("=" * 70)
print("WiDS 2026 — Pipeline v14 EXTREME (The 'Fire Is Real' Run)")
print("=" * 70)

train = pd.read_csv(f'{DATA_DIR}/train.csv')
test = pd.read_csv(f'{DATA_DIR}/test.csv')
sample = pd.read_csv(f'{DATA_DIR}/sample_submission.csv')

# Load submission 09 for distillation
best_sub = pd.read_csv(f'{DATA_DIR}/submission_09.csv')

# ============================================================
# PHASE 1: Engineering Physics Features
# ============================================================
def engineer_features(df):
    out = pd.DataFrame(index=df.index)
    out['dist_min_ci_0_5h'] = df['dist_min_ci_0_5h'].clip(lower=0)
    out['closing_speed_m_per_h'] = df['closing_speed_m_per_h']
    out['alignment_abs'] = df['alignment_abs']
    out['radial_growth_rate_m_per_h'] = df['radial_growth_rate_m_per_h'].clip(lower=0)
    out['radial_growth_m'] = df['radial_growth_m'].clip(lower=0)
    
    out['projected_advance_m'] = df['projected_advance_m'].clip(lower=0)
    
    # Advanced logic features
    safe_speed = df['closing_speed_m_per_h'].clip(lower=0.001)
    out['projected_time_to_hit'] = (df['dist_min_ci_0_5h'] / safe_speed).clip(0, 500)
    out['threat_gravity'] = (df['alignment_abs'] * df['closing_speed_m_per_h']) / ((df['dist_min_ci_0_5h'] + 100) ** 2)
    return out

X_train_raw = engineer_features(train)
X_test_raw = engineer_features(test)

features = list(X_train_raw.columns)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# ============================================================
# PHASE 2: Pseudo-Label Augmentation
# ============================================================
print("\n[PHASE 2] Extracting High-Confidence Pseudo-Labels")
pseudo_X = []
pseudo_y = {h: [] for h in TIME_HORIZONS}

for i in range(len(test)):
    confidences = [best_sub.loc[i, f'prob_{h}h'] for h in TIME_HORIZONS]
    is_confident = any(c > 0.90 or c < 0.10 for c in confidences)
    
    if is_confident:
        pseudo_X.append(X_test[i])
        for h in TIME_HORIZONS:
            p = best_sub.loc[i, f'prob_{h}h']
            label = 1.0 if p >= 0.5 else 0.0
            pseudo_y[h].append(label)

if len(pseudo_X) > 0:
    pseudo_X = np.array(pseudo_X)
    for h in TIME_HORIZONS:
        pseudo_y[h] = np.array(pseudo_y[h])

# ============================================================
# PHASE 3: Training Resilient Models
# ============================================================
print("\n[PHASE 3] Training Extremist Pseudo-Label Ensembles")

final_preds = {h: np.zeros(len(test)) for h in TIME_HORIZONS}

def get_train_labels(h):
    return ((train['event'] == 1) & (train['time_to_hit_hours'] <= h)).astype(float).values

for h in TIME_HORIZONS:
    y_train_h = get_train_labels(h)
    
    if len(pseudo_X) > 0:
        X_train_aug = np.vstack([X_train, pseudo_X])
        y_train_aug = np.concatenate([y_train_h, pseudo_y[h]])
        w_main = np.ones(len(X_train))
        w_pseudo = np.ones(len(pseudo_X)) * 0.8  # High trust in pseudo labels for "Fire Is Real" run
        w_train_aug = np.concatenate([w_main, w_pseudo])
    else:
        X_train_aug, y_train_aug, w_train_aug = X_train, y_train_h, np.ones(len(X_train))

    preds_h = np.zeros(len(test))
    
    for seed in SEEDS:
        # CatBoost
        cb = CatBoostClassifier(iterations=300, depth=3, learning_rate=0.03, verbose=0, random_seed=seed, l2_leaf_reg=15)
        cb.fit(X_train_aug, y_train_aug, sample_weight=w_train_aug)
        preds_h += cb.predict_proba(X_test)[:, 1] / (len(SEEDS) * 3)
        
        # LightGBM
        lgb_model = lgb.LGBMClassifier(n_estimators=300, max_depth=3, num_leaves=8, learning_rate=0.03, reg_lambda=20, verbosity=-1, random_state=seed)
        lgb_model.fit(X_train_aug, y_train_aug, sample_weight=w_train_aug)
        preds_h += lgb_model.predict_proba(X_test)[:, 1] / (len(SEEDS) * 3)

        # ExtraTrees
        et = ExtraTreesClassifier(n_estimators=300, max_depth=4, random_state=seed)
        et.fit(X_train_aug, y_train_aug, sample_weight=w_train_aug)
        preds_h += et.predict_proba(X_test)[:, 1] / (len(SEEDS) * 3)

    final_preds[h] = preds_h

# ============================================================
# PHASE 4: Pure Mathematics Overrides (No Mercy)
# ============================================================
print("\n[PHASE 4] Physical Override — Strict Laws of Motion and Fire Dynamics")

hit_overrides = 0
miss_overrides = 0

for i in range(len(test)):
    row = test.iloc[i]
    dist = row['dist_min_ci_0_5h']
    speed = row['closing_speed_m_per_h']
    growth_rate = row['radial_growth_rate_m_per_h']
    alignment = row['alignment_abs']
    
    for h in TIME_HORIZONS:
        # Distance capacity in 'h' hours
        max_reach = (speed * h) + (growth_rate * h)
        
        # 100% Guaranteed Hit Rule:
        if speed > 20 and alignment < 0.2 and max_reach >= (dist * 1.5):
            final_preds[h][i] = 0.995
            hit_overrides += 1
            
        # 100% Guaranteed Miss Rule:
        # If fire is receding (negative speed) AND its radial growth isn't fast enough to catch us
        elif speed < 0 and (growth_rate * h) < (dist * 0.5):
            final_preds[h][i] = 0.005
            miss_overrides += 1

print(f"  --> Inevitable strikes guaranteed: {hit_overrides}")
print(f"  --> Impossible strikes eliminated: {miss_overrides}")

# ============================================================
# PHASE 5: Aggressive Rank Optimization & Monotonicity
# ============================================================
print("\n[PHASE 5] Extreme Rank Sharpening & Hard Monotonicity")

# Extreme Rank Sharpening (+20%) to maximize C-Index given that our binary targets are locked
for h in TIME_HORIZONS:
    cal = final_preds[h]
    rank_pct = rankdata(cal) / len(cal)
    final_preds[h] = 0.80 * cal + 0.20 * rank_pct

# Enforce strict monotonicity across horizons physically
for i in range(len(test)):
    vals = np.array([final_preds[h][i] for h in TIME_HORIZONS])
    vals = np.maximum.accumulate(vals)
    for j, h in enumerate(TIME_HORIZONS):
        final_preds[h][i] = vals[j]

sub_14 = sample.copy()
for h in TIME_HORIZONS:
    sub_14[f'prob_{h}h'] = np.clip(final_preds[h], 0.005, 0.995)

sub_14.to_csv(f'{DATA_DIR}/submission_14.csv', index=False)

elapsed = time.time() - start_time
print(f"\nSUCCESS: submission_14.csv generated under extreme physics parameters in {elapsed:.1f} sec. 🚀")
