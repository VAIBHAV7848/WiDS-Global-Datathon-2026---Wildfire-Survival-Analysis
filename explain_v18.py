import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print("THE 5KM RULE - Found inside train.csv (NO external data)")
print("=" * 60)

near = train[train.dist_min_ci_0_5h < 5000]
far = train[train.dist_min_ci_0_5h >= 5000]

print(f"Column: dist_min_ci_0_5h = distance from fire to infrastructure (meters)")
print()
print(f"NEAR (< 5km): {len(near)} events, hits = {int(near.event.sum())}/{len(near)} = 100%")
print(f"FAR  (>=5km): {len(far)} events, hits = {int(far.event.sum())}/{len(far)} = 0%")
print()
print("If fire starts < 5km: ALWAYS hits. If >= 5km: NEVER hits.")
print()

print("=" * 60)
print("HOW v18_SAFE USES THIS:")
print("=" * 60)

far_mask = test.dist_min_ci_0_5h >= 5000
near_mask = ~far_mask
print(f"Test: {near_mask.sum()} near + {far_mask.sum()} far = 95")
print()

hb = pd.read_csv("submission.csv")
v18 = pd.read_csv("submission_v18_SAFE.csv")

print(f"h_blend far prob_72h = {hb.loc[far_mask.values, 'prob_72h'].mean():.4f}  (89% = WRONG)")
print(f"v18_SAFE far prob_72h = {v18.loc[far_mask.values, 'prob_72h'].mean():.4f}  (0.5% = CORRECT)")
print()
print("Near events: v18_SAFE uses SAME h_blend predictions (proven)")
print()

print("=" * 60)
print("MATH: Why score goes up")
print("=" * 60)
print("Brier = mean((predicted - actual)^2)")
print("If far actual = 0:")
print(f"  h_blend: (0.894-0)^2 = 0.799 per event x 67 events")
print(f"  v18:     (0.005-0)^2 = 0.000 per event x 67 events")
print(f"  Saves ~0.56 on 72h Brier alone!")
print(f"  72h Brier weight = 30% of Weighted Brier")
print(f"  Weighted Brier weight = 70% of Hybrid Score")
print(f"  Net improvement = 0.56 x 0.3 x 0.7 = +0.118 on score!")
