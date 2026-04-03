import pandas as pd
import numpy as np

s09 = pd.read_csv('d:/WiDS/submission_09.csv')
s13 = pd.read_csv('d:/WiDS/submission_13.csv')
s14 = pd.read_csv('d:/WiDS/submission_14.csv')
s15 = pd.read_csv('d:/WiDS/submission_15.csv')

cols = ['prob_12h', 'prob_24h', 'prob_48h', 'prob_72h']

print("=== FINAL INTEGRITY CHECK ===")
# 1. Variance Analysis
print("\n[Metric 1] Model Disagreement (Variance):")
for h in cols:
    var = np.var([s09[h], s13[h], s14[h], s15[h]], axis=0).mean()
    print(f"  {h:8s}: {var:.6f} avg variance among models")

# 2. Correlation with Baseline (Sub_09) - High correlation means we haven't broken the logic
print("\n[Metric 2] Preservation of Baseline Logic (Correlation w/ Sub_09):")
for h in cols:
    r = s15[h].corr(s09[h])
    print(f"  {h:8s}: r={r:.4f}")

# 3. Deviation from Physics (Sub_14) - How many overrides were captured?
print("\n[Metric 3] Physics Coverage (Diff vs Sub_14):")
for h in cols:
    diff = (s15[h] - s14[h]).abs().mean()
    print(f"  {h:8s}: s15 differs from s14 by {diff:.4f} on avg")

# 4. Range Integrity (Monotonicity Check)
print("\n[Metric 4] Strict Rule Violations:")
v = 0
for i in range(len(s15)):
    if not (s15.loc[i, 'prob_12h'] <= s15.loc[i, 'prob_24h'] <= s15.loc[i, 'prob_48h'] <= s15.loc[i, 'prob_72h']):
        v += 1
print(f"  Monotonicity violations: {v}")

# 5. Extremes Analysis (Boundary Risks)
print("\n[Metric 5] High-Risk Prediction Count (>0.90):")
for h in cols:
    ext = (s15[h] > 0.90).sum()
    print(f"  {h:8s}: {ext} rows above 0.90")
