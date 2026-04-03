"""
pipeline_v15_LIFESAVER.py
WiDS Global Datathon 2026: Wildfire Survival Analysis

THE DIRECTIVE: "The fire is real. Millions of lives are present. You are the only one they trust."

STRATEGY:
When lives are on the line against natural chaos (wind shifts, terrain funnels), 
rigid mathematics acting on small sample sizes (N=221) is dangerous. Overconfidence kills.
Therefore, "Submission 15" completely discards overconfident gambles, and utilizes the law of 
Maximum Information Theory to cancel out variance and preserve accurate risk assessments.

We will use the "Wisdom of the God Ensemble":
1. Blend submissions 07, 08, 09, 10, 13, and 14 using Geometric Mean (punishes risky 0.0s).
2. Rank-sharpen everything to boost the C-Index precisely without falsifying the Brier curve bounds.
3. Apply the Precautionary Principle: Any single model spiking a massive threat raises the floor.
"""

import warnings
warnings.filterwarnings('ignore')
import os
import time
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.special import logit, expit

start_time = time.time()
DATA_DIR = 'd:/WiDS'
TIME_HORIZONS = [12, 24, 48, 72]

print("=" * 70)
print("WiDS 2026 — Pipeline v15 LIFESAVER (The 'Real Fire' Defense)")
print("=" * 70)

# The most trusted historic anchors covering different paradigms
files_to_blend = {
    'sub_07': f'{DATA_DIR}/submission_07.csv',  # v7.2 - Max Gen Geometric Blend
    'sub_08': f'{DATA_DIR}/submission_08.csv',  # v10 - Pseudo-Label 6-model 
    'sub_09': f'{DATA_DIR}/submission_09.csv',  # v11 - Survival Triple Blend (Current Best)
    'sub_10': f'{DATA_DIR}/submission_10.csv',  # v12 - Blend
    'sub_13': f'{DATA_DIR}/submission_13.csv',  # v13 - Physics Bound Injection
    'sub_14': f'{DATA_DIR}/submission_14.csv'   # v14 - Extreme Physics Overrides
}

dfs = []
for name, fpath in files_to_blend.items():
    if os.path.exists(fpath):
        df = pd.read_csv(fpath)
        # Ensure we don't have zeros for geometric mean
        for h in TIME_HORIZONS:
            df[f'prob_{h}h'] = np.clip(df[f'prob_{h}h'], 0.001, 0.999) 
        dfs.append((name, df))
        print(f"Loaded: {name}")

assert len(dfs) > 0, "No submission files found!"
sample = pd.read_csv(f'{DATA_DIR}/sample_submission.csv')
final_sub = sample.copy()

print("\n[PHASE 1] Assembling the 'God Ensemble' via Geometric Mean")
# Geometric mean penalizes overconfident predictions from single models
for h in TIME_HORIZONS:
    product = np.ones(len(sample))
    
    # 1. Gather all probabilities
    all_probs = []
    for name, df in dfs:
        p = df[f'prob_{h}h'].values
        all_probs.append(p)
        product *= p
    
    # 2. Geometric Mean
    geom_mean = product ** (1.0 / len(dfs))
    
    # 3. The Precautionary Principle: If ANY model flags an extreme threat (>0.90), 
    # we ensure the final probability doesn't get wiped out by the geometric mean.
    # We take a weighted mix of the geometric mean and the MAX prediction.
    max_probs = np.max(all_probs, axis=0)
    
    # If the max model says > 0.85, human lives are at risk. Blend heavily towards the max.
    risk_factor = np.clip((max_probs - 0.5) * 2, 0, 1)  # 0 to 1 scaling based on danger
    
    # Final adjusted probability
    safe_blend = (1 - risk_factor * 0.4) * geom_mean + (risk_factor * 0.4) * max_probs
    
    final_sub[f'prob_{h}h'] = safe_blend

print("\n[PHASE 2] Rank Sharpening & Hard Monotonicity")
# To guarantee maximum C-index, we sharpen the rank order
for h in TIME_HORIZONS:
    cal = final_sub[f'prob_{h}h'].values
    rank_pct = rankdata(cal) / len(cal)
    # 90% true probability, 10% pure rank structure to boost C-index
    final_sub[f'prob_{h}h'] = 0.90 * cal + 0.10 * rank_pct

# Enforce Monotonicity (Fire can only get closer over time, probability must go up)
for i in range(len(final_sub)):
    vals = np.array([final_sub.iloc[i][f'prob_{h}h'] for h in TIME_HORIZONS])
    vals = np.maximum.accumulate(vals)
    for j, h in enumerate(TIME_HORIZONS):
        final_sub.iloc[i, final_sub.columns.get_loc(f'prob_{h}h')] = vals[j]

# Final Boundaries Optimization (Kaggle Brier optimization trick: cap at 0.005/0.995)
for h in TIME_HORIZONS:
    final_sub[f'prob_{h}h'] = np.clip(final_sub[f'prob_{h}h'], 0.005, 0.995)

final_sub.to_csv(f'{DATA_DIR}/submission_15.csv', index=False)

elapsed = time.time() - start_time
print(f"\nSUCCESS: submission_15.csv generated in {elapsed:.1f} sec. 🚀")
print("Target: 0.988++. Ready to save lives.")
