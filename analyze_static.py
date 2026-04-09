import pandas as pd
import numpy as np

t = pd.read_csv('train.csv')
far = t.dist_min_ci_0_5h >= 5000
act = ~far & ((t.radial_growth_rate_m_per_h > 0) | (t.area_growth_rate_ha_per_h > 0))
sta = ~far & ~act
s = t[sta]

print('=== STATIC ZONE DEEP ANALYSIS ===')
print(f'All event=1: {(s.event==1).all()}')
print(f'time_to_hit stats:')
print(s.time_to_hit_hours.describe())
print()

for h in [12, 24, 48, 72]:
    hit = s.time_to_hit_hours <= h
    print(f'By {h}h: {hit.sum()}/52 hit ({hit.mean():.3f})')

print()
print('Key feature correlations with time_to_hit:')
for c in ['dist_min_ci_0_5h','alignment_abs','dt_first_last_0_5h',
          'num_perimeters_0_5h','log1p_area_first','closing_speed_abs_m_per_h']:
    print(f'  {c}: r={s[c].corr(s.time_to_hit_hours):.3f}')

print()
print('=== STATIC ZONE: time_to_hit distribution ===')
bins = [0, 1, 3, 6, 12, 24, 48, 72]
for i in range(len(bins)-1):
    n = ((s.time_to_hit_hours > bins[i]) & (s.time_to_hit_hours <= bins[i+1])).sum()
    print(f'  ({bins[i]},{bins[i+1]}]h: {n} events')

print()
print('=== V20 PREDICTIONS FOR STATIC TEST ===')
test = pd.read_csv('test.csv')
v20 = pd.read_csv('submission_v20_MEGA.csv')
tfar = test.dist_min_ci_0_5h >= 5000
tact = ~tfar & ((test.radial_growth_rate_m_per_h > 0) | (test.area_growth_rate_ha_per_h > 0))
tsta = ~tfar & ~tact
v20_static = v20[tsta]
print(f'Test static events: {tsta.sum()}')
print(v20_static[['event_id','prob_12h','prob_24h','prob_48h','prob_72h']].to_string())

print()
print('=== BRIER CONTRIBUTION ANALYSIS ===')
print('If all FAR=0.001, ACTIVE=0.999, and test follows training distribution:')
print(f'  FAR (67 events, all event=0): Brier contribution = {67/95 * 0.001**2:.8f}')
print(f'  ACTIVE (3 events, all event=1 at all h): Brier contribution = {3/95 * 0.001**2:.8f}')
print(f'  Total Brier from FAR+ACTIVE: ~0.000 (essentially perfect)')
print(f'  ALL BRIER ERROR comes from 25 STATIC test events')
print(f'  ---> Improving 25 static predictions is THE ONLY WAY to improve score')
