"""Create the FINAL BEST submission with the optimal strategy."""
import pandas as pd
import numpy as np

test = pd.read_csv("test.csv")
far = test.dist_min_ci_0_5h >= 5000
act = ~far & ((test.radial_growth_rate_m_per_h > 0) | (test.area_growth_rate_ha_per_h > 0))
sta = ~far & ~act

HC = ['prob_12h','prob_24h','prob_48h','prob_72h']

# Key finding: v20_PURE has ZERO discrimination at 12h (all 0.6153)
# v20_MEGA discrimination at 12h comes 100% from v18_BLEND
# Our ULTIMATE model has BETTER discrimination (range 0.32 vs v20_MEGA's 0.21)

# Load the key submissions
v20m = pd.read_csv("submission_v20_MEGA.csv")
v18b = pd.read_csv("submission_v18_BLEND.csv")
ult_smart = pd.read_csv("submission_v25_ULTIMATE_SMART.csv")
ult_perh = pd.read_csv("submission_v25_ULTIMATE_PERH.csv")
ult_pure = pd.read_csv("submission_v25_ULTIMATE_PURE.csv")

print("=== Event-by-event comparison (STATIC) ===\n")
print(f"{'event_id':>12s}  {'v20_12h':>8s}  {'v18_12h':>8s}  {'ULT_12h':>8s}  {'DIFF':>8s}")
for i in range(len(test)):
    if not sta.iloc[i]:
        continue
    eid = test.iloc[i].event_id
    v20_p = v20m.iloc[i].prob_12h
    v18_p = v18b.iloc[i].prob_12h
    ult_p = ult_smart.iloc[i].prob_12h
    diff = ult_p - v20_p
    print(f"{eid:>12.0f}  {v20_p:8.4f}  {v18_p:8.4f}  {ult_p:8.4f}  {diff:+8.4f}")

# The OPTIMAL strategy: 
# - For 72h: use 0.999 (all static train events hit by 72h)
# - For 48h: mostly high, use conservative v20-dominant blend
# - For 24h: decent v20 discrimination, blend with our model
# - For 12h: v20 has poor discrimination (from v18 blend only), 
#            our model adds GENUINE ML-based discrimination
# Key: 72h predictions determine C-index. Both v20 and our model predict 0.999.
# So C-index depends entirely on FAR=0.001 vs ACTIVE/STATIC=0.999.
# Since ALL far test events should be event=0 and ALL static/active should be event=1,
# the C-index should already be near-perfect with FAR/STATIC/ACTIVE zone gating.

# THE BEST POSSIBLE SUBMISSION:
# Uses RANKING from ML models (for C-index) + CALIBRATED PROBABILITIES (for Brier)

# Strategy: since C-index is about ranking and all static have prob_72h=0.999,
# within-static ranking doesn't affect C-index at all! (all have same prob_72h)
# => C-index is determined ONLY by zone separation: FAR=0.001 << STATIC=0.999
# => We can ONLY improve Brier by better 12h/24h/48h predictions

# For Brier at each horizon, averaged over 95 events:
# - 67 FAR events: contribute ~0 Brier (0.001^2 per event)
# - 3 ACTIVE events: contribute ~0 Brier (0.001^2 per event)
# - 25 STATIC events: ALL the Brier error

# For 12h: ~38% of static events DON'T hit by 12h
# -> Those events should have LOW prob_12h
# -> v20 gives them ~0.55-0.75 (TOO HIGH for non-hits) => Brier ≈ 0.6^2 ≈ 0.36
# -> If we correctly predict ~0.2 for non-hits: Brier ≈ 0.2^2 ≈ 0.04
# -> This one correction could reduce 12h Brier by ~0.30 per event!

print("\n\n=== CREATING THE DEFINITIVE SUBMISSION ===\n")
print("Strategy: maximize 12h discrimination + v20 for 24h/48h/72h\n")

# Final blend per horizon:
# 12h: 50% ULTIMATE_PURE + 50% v18_BLEND (maximize discrimination from diverse sources)
# 24h: 30% ULTIMATE_SMART + 70% v20_MEGA (v20 already decent here)
# 48h: 15% ULTIMATE_SMART + 85% v20_MEGA (v20 very good here)
# 72h: 100% v20_MEGA (= 0.999 for all static, perfect)

sub_final = pd.DataFrame({'event_id': test['event_id']})
for col in HC:
    sub_final[col] = 0.0
    sub_final.loc[far.values, col] = 0.001
    sub_final.loc[act.values, col] = 0.999

# 12h: maximize discrimination from diverse sources
p12_new = 0.35 * ult_pure.loc[sta.values, 'prob_12h'].values + \
          0.35 * ult_smart.loc[sta.values, 'prob_12h'].values + \
          0.30 * v18b.loc[sta.values, 'prob_12h'].values
sub_final.loc[sta.values, 'prob_12h'] = p12_new

# 24h: rely more on v20
p24_new = 0.30 * ult_smart.loc[sta.values, 'prob_24h'].values + \
          0.70 * v20m.loc[sta.values, 'prob_24h'].values
sub_final.loc[sta.values, 'prob_24h'] = p24_new

# 48h: mostly v20
p48_new = 0.15 * ult_smart.loc[sta.values, 'prob_48h'].values + \
          0.85 * v20m.loc[sta.values, 'prob_48h'].values
sub_final.loc[sta.values, 'prob_48h'] = p48_new

# 72h: pure v20 (perfect)
sub_final.loc[sta.values, 'prob_72h'] = v20m.loc[sta.values, 'prob_72h'].values

# Monotonicity enforcement
for idx in sub_final.index:
    prev = 0
    for col in HC:
        if sub_final.loc[idx, col] < prev:
            sub_final.loc[idx, col] = prev
        prev = sub_final.loc[idx, col]

for col in HC:
    sub_final[col] = sub_final[col].clip(0.001, 0.999)

sub_final.to_csv("submission_DEFINITIVE.csv", index=False)

print("submission_DEFINITIVE.csv saved!\n")
sm = sub_final.loc[sta.values, HC]
print(f"STATIC zone stats:")
print(f"  12h: mean={sm.prob_12h.mean():.4f}, min={sm.prob_12h.min():.4f}, max={sm.prob_12h.max():.4f}, range={sm.prob_12h.max()-sm.prob_12h.min():.4f}")
print(f"  24h: mean={sm.prob_24h.mean():.4f}, min={sm.prob_24h.min():.4f}, max={sm.prob_24h.max():.4f}")
print(f"  48h: mean={sm.prob_48h.mean():.4f}, min={sm.prob_48h.min():.4f}, max={sm.prob_48h.max():.4f}")
print(f"  72h: mean={sm.prob_72h.mean():.4f}")

# Also create copies with v20-dominant blends
for name, w12, w24, w48 in [
    ("DEF_V20DOMINANT", 0.25, 0.15, 0.10),
    ("DEF_BALANCED", 0.45, 0.35, 0.20),
]:
    sub = pd.DataFrame({'event_id': test['event_id']})
    for col in HC:
        sub[col] = 0.0
        sub.loc[far.values, col] = 0.001
        sub.loc[act.values, col] = 0.999
    
    # 12h: blend
    p12 = w12 * ult_pure.loc[sta.values, 'prob_12h'].values + \
          (1-w12) * v20m.loc[sta.values, 'prob_12h'].values
    sub.loc[sta.values, 'prob_12h'] = p12
    
    # 24h
    p24 = w24 * ult_smart.loc[sta.values, 'prob_24h'].values + \
          (1-w24) * v20m.loc[sta.values, 'prob_24h'].values
    sub.loc[sta.values, 'prob_24h'] = p24
    
    # 48h
    p48 = w48 * ult_smart.loc[sta.values, 'prob_48h'].values + \
          (1-w48) * v20m.loc[sta.values, 'prob_48h'].values
    sub.loc[sta.values, 'prob_48h'] = p48
    
    # 72h: v20
    sub.loc[sta.values, 'prob_72h'] = v20m.loc[sta.values, 'prob_72h'].values
    
    for idx in sub.index:
        prev = 0
        for col in HC:
            if sub.loc[idx, col] < prev:
                sub.loc[idx, col] = prev
            prev = sub.loc[idx, col]
    
    for col in HC:
        sub[col] = sub[col].clip(0.001, 0.999)
    
    sub.to_csv(f"submission_{name}.csv", index=False)
    sm2 = sub.loc[sta.values, HC]
    print(f"\n{name}: 12h={sm2.prob_12h.mean():.4f} 24h={sm2.prob_24h.mean():.4f} "
          f"48h={sm2.prob_48h.mean():.4f} 72h={sm2.prob_72h.mean():.4f}")

# Validation
print("\n\n=== VALIDATION ===")
for fname in ['submission_DEFINITIVE.csv', 'submission_DEF_V20DOMINANT.csv', 'submission_DEF_BALANCED.csv']:
    sub = pd.read_csv(fname)
    nulls = sub.isnull().any().any()
    mono_ok = True
    for i in range(len(sub)):
        for j in range(3):
            if sub.iloc[i][HC[j]] > sub.iloc[i][HC[j+1]] + 1e-9:
                mono_ok = False
    valid = (sub[HC] >= 0).all().all() and (sub[HC] <= 1).all().all()
    print(f"  {fname}: rows={len(sub)}, nulls={nulls}, mono_ok={mono_ok}, valid_range={valid}")
