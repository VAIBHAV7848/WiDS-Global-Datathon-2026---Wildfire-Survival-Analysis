"""
V25 FINAL ANALYSIS: Compare ALL available submissions for STATIC zone
and create the ABSOLUTE BEST ensemble by combining them all.
"""
import pandas as pd
import numpy as np

test = pd.read_csv("test.csv")
far = test.dist_min_ci_0_5h >= 5000
act = ~far & ((test.radial_growth_rate_m_per_h > 0) | (test.area_growth_rate_ha_per_h > 0))
sta = ~far & ~act

HC = ['prob_12h','prob_24h','prob_48h','prob_72h']

# Load ALL previous submissions
subs = {}
for name in ['v18_BLEND', 'v20_MEGA', 'v20_PURE', 'v20_B40', 'v20_B60',
             'v25_ULTIMATE_SMART', 'v25_ULTIMATE_PERH', 'v25_ULTIMATE_PURE',
             'v25c_B50_STACK', 'v25c_B30_STACK']:
    try:
        subs[name] = pd.read_csv(f"submission_{name}.csv")
    except:
        pass

print("=== STATIC ZONE PREDICTIONS: ALL SUBMISSIONS ===\n")
for name, sub in subs.items():
    static = sub[sta]
    means = static[HC].mean()
    ranges = static[HC].max() - static[HC].min()
    print(f"{name:30s}: 12h={means.iloc[0]:.4f} 24h={means.iloc[1]:.4f} "
          f"48h={means.iloc[2]:.4f} 72h={means.iloc[3]:.4f} | "
          f"range_12h={ranges.iloc[0]:.4f}")

# Statistical comparison - how correlated are submissions?
print("\n=== Correlation of 12h predictions (STATIC only) ===")
df_compare = pd.DataFrame()
for name, sub in subs.items():
    df_compare[name] = sub[sta]['prob_12h'].values

print(df_compare.corr().to_string())

# KEY: Create a MEGA-BLEND: average of ALL available submissions
# But zone-gated: only average STATIC zone from ML-based submissions
print("\n=== CREATING MEGA BLENDS ===")

# Method 1: Average all ML submissions (diversified ensemble)
ml_subs = ['v20_MEGA', 'v18_BLEND', 'v25_ULTIMATE_SMART', 'v25c_B50_STACK']
for blend_name, sub_list, weights in [
    ("MEGABLEND_EQUAL", ml_subs, None),
    ("MEGABLEND_V20H", ['v20_MEGA', 'v18_BLEND', 'v25_ULTIMATE_SMART', 'v25c_B50_STACK'],
     [0.40, 0.20, 0.25, 0.15]),
    # v20 heavy
    ("MEGABLEND_V20D", ['v20_MEGA', 'v25_ULTIMATE_SMART'],
     [0.65, 0.35]),
]:
    mega = pd.DataFrame({'event_id': test['event_id']})
    
    for col in HC:
        mega[col] = 0.0
        mega.loc[far.values, col] = 0.001
        mega.loc[act.values, col] = 0.999
        
        if weights:
            bl = sum(w * subs[s].loc[sta.values, col].values 
                     for w, s in zip(weights, sub_list))
        else:
            preds = [subs[s].loc[sta.values, col].values for s in sub_list]
            bl = np.mean(preds, axis=0)
        
        mega.loc[sta.values, col] = bl
    
    # Monotonicity
    for idx in mega.index:
        prev = 0
        for col in HC:
            if mega.loc[idx, col] < prev:
                mega.loc[idx, col] = prev
            prev = mega.loc[idx, col]
    
    for col in HC:
        mega[col] = mega[col].clip(0.001, 0.999)
    
    mega.to_csv(f"submission_{blend_name}.csv", index=False)
    sm = mega.loc[sta.values, HC].mean()
    print(f"  {blend_name}: 12h={sm.iloc[0]:.4f} 24h={sm.iloc[1]:.4f} "
          f"48h={sm.iloc[2]:.4f} 72h={sm.iloc[3]:.4f}")

# Method 2: Rank-blend of diverse submissions
from scipy.stats import rankdata

for rbl_name, sub_list in [
    ("RANKBLEND_ALL", ['v20_MEGA', 'v18_BLEND', 'v25_ULTIMATE_SMART', 'v25c_B50_STACK']),
    ("RANKBLEND_ML3", ['v20_MEGA', 'v25_ULTIMATE_SMART', 'v25_ULTIMATE_PERH']),
]:
    rb = pd.DataFrame({'event_id': test['event_id']})
    
    for col in HC:
        rb[col] = 0.0
        rb.loc[far.values, col] = 0.001
        rb.loc[act.values, col] = 0.999
        
        static_preds = [subs[s].loc[sta.values, col].values for s in sub_list]
        ranks = np.mean([rankdata(p) / len(p) for p in static_preds], axis=0)
        # Scale rank-avg back to probability range using min/max from v20
        v20_static = subs['v20_MEGA'].loc[sta.values, col].values
        lo, hi = v20_static.min(), v20_static.max()
        scaled = lo + (hi - lo) * ranks
        rb.loc[sta.values, col] = scaled
    
    for idx in rb.index:
        prev = 0
        for col in HC:
            if rb.loc[idx, col] < prev:
                rb.loc[idx, col] = prev
            prev = rb.loc[idx, col]
    
    for col in HC:
        rb[col] = rb[col].clip(0.001, 0.999)
    
    rb.to_csv(f"submission_{rbl_name}.csv", index=False)
    sm = rb.loc[sta.values, HC].mean()
    print(f"  {rbl_name}: 12h={sm.iloc[0]:.4f} 24h={sm.iloc[1]:.4f} "
          f"48h={sm.iloc[2]:.4f} 72h={sm.iloc[3]:.4f}")

print("\n=== BEST SUBMISSION RECOMMENDATION ===")
print("""
With only 25 static test events, the key to 0.989+ is:
1) FAR=0.001 and ACTIVE=0.999 (already perfect)
2) STATIC zone: right RANKING (for C-index) + right CALIBRATION (for Brier)

Our model adds value via WIDER SPREAD of predictions (range 0.32 vs v20's 0.21 at 12h).
This means better DISCRIMINATION of which events hit early vs late.

BUT: we don't know if wider spread = more correct. It depends on whether 
our model correctly identifies which test events hit/miss at each horizon.

SAFEST SUBMISSIONS (descending risk):
  1. submission_MEGABLEND_V20D.csv  -- 65% v20 + 35% new (SAFEST)
  2. submission_MEGABLEND_V20H.csv  -- 40% v20 + component blend
  3. submission_v25_ULTIMATE_SMART.csv -- Per-horizon blend
""")
