import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print("=== TRAINING DATA ===")
print("Total rows:", len(train))
print("Event=1 (hit):", train["event"].sum())
print("Event=0 (censored):", (train["event"]==0).sum())
print()

for h in [12, 24, 48, 72]:
    n = ((train['event']==1) & (train['time_to_hit_hours']<=h)).sum()
    print("Hits <= %dh: %d (%.1f%%)" % (h, n, n/len(train)*100))

print()
print("=== TEST DATA BREAKDOWN ===")
dist = test['dist_min_ci_0_5h']

print("Total test events:", len(test))
print("Far (>=5km):", (dist>=5000).sum(), "events")
print("Near (<5km):", (dist<5000).sum(), "events")
print()

active = test[dist < 5000].copy()
print("=== THE 28 NEAR EVENTS (each line = 1 event) ===")
for i, row in active.iterrows():
    speed = max(row['closing_speed_m_per_h'], 0.01)
    eta = row['dist_min_ci_0_5h'] / speed
    growth = row['radial_growth_rate_m_per_h']
    area_g = row['area_growth_rate_ha_per_h']
    is_active = "ACTIVE" if (growth > 0 or area_g > 0) else "STATIC"
    
    # Classification
    if row['dist_min_ci_0_5h'] < 500 or eta < 6:
        cat = "CERTAIN_HIT"
    elif row['dist_min_ci_0_5h'] < 2000 and eta < 24:
        cat = "LIKELY_HIT"
    elif eta < 72:
        cat = "POSSIBLE_HIT"
    else:
        cat = "UNCERTAIN"
    
    print("  %10d: dist=%6.0fm speed=%7.1fm/h align=%.2f eta=%8.1fh type=%-6s class=%s" % (
        int(row["event_id"]), row["dist_min_ci_0_5h"], 
        row["closing_speed_m_per_h"], row["alignment_abs"],
        eta, is_active, cat))

# Summary
print()
print("=== FILE INVENTORY ===")
print("Pipeline versions: pipeline_v20_PHYSICS.py, pipeline_v21_ORACLE.py")
print("Submission files: submission.csv(LB=0.97175), submission_v18_BLEND.csv(LB=0.97216), v20_PURE, v20_50, v20_MEGA, v21_A, v21_B, v21_C")
print("Best known score: 0.97216 (submission_v18_BLEND.csv)")
print("What it did: 2-zone gate (far=0.005, near=50%% ML + 50%% h_blend)")
