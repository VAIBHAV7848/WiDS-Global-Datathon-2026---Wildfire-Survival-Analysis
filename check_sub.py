import pandas as pd
df = pd.read_csv(r"d:\WiDS\submission_12.csv")

# 1. Check rows and columns
print(f"Shape: {df.shape} (Expected 95, 5)")
print(f"Columns: {df.columns.tolist()}")

# 2. Check monotonicity: 12h <= 24h <= 48h <= 72h
mono_violations = 0
for i, row in df.iterrows():
    if not (row['prob_12h'] <= row['prob_24h'] <= row['prob_48h'] <= row['prob_72h']):
        mono_violations += 1

print(f"Monotonicity Violations: {mono_violations} (Must be 0)")

# 3. Check bounds [0.008, 0.992]
min_val = df[['prob_12h', 'prob_24h', 'prob_48h', 'prob_72h']].min().min()
max_val = df[['prob_12h', 'prob_24h', 'prob_48h', 'prob_72h']].max().max()
print(f"Min probability: {min_val:.5f} (Target lower bound usually 0.005-0.008)")
print(f"Max probability: {max_val:.5f} (Target upper bound usually 0.992-0.995)")
