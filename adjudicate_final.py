import pandas as pd
import numpy as np

# Load Data
train = pd.read_csv('d:/WiDS/train.csv')
s13 = pd.read_csv('d:/WiDS/submission_13.csv')
s14 = pd.read_csv('d:/WiDS/submission_14.csv')
s15 = pd.read_csv('d:/WiDS/submission_15.csv')

horizons = [12, 24, 48, 72]

# 1. Base Rate Calculation
train_means = []
for h in horizons:
    m = ((train['event'] == 1) & (train['time_to_hit_hours'] <= h)).mean()
    train_means.append(m)

# 2. Results Collection
results = []
for h_idx, h in enumerate(horizons):
    t_m = train_means[h_idx]
    
    row = {
        'Horizon': f"{h}h",
        'Train_Mean': t_m,
        'S13_Mean': s13[f'prob_{h}h'].mean(),
        'S14_Mean': s14[f'prob_{h}h'].mean(),
        'S15_Mean': s15[f'prob_{h}h'].mean(),
        'S13_Std': s13[f'prob_{h}h'].std(),
        'S14_Std': s14[f'prob_{h}h'].std(),
        'S15_Std': s15[f'prob_{h}h'].std(),
    }
    results.append(row)

res_df = pd.DataFrame(results)

print("=== ADJUDICATION MATRIX ===")
print("\nMEAN PROBABILITIES (Base Rate Alignment):")
for i, r in res_df.iterrows():
    print(f"{r['Horizon']} | Train: {r['Train_Mean']:.4f} | S13: {r['S13_Mean']:.4f} (diff: {abs(r['S13_Mean']-r['Train_Mean']):.4f}) | S14: {r['S14_Mean']:.4f} | S15: {r['S15_Mean']:.4f} (diff: {abs(r['S15_Mean']-r['Train_Mean']):.4f})")

print("\nSPREAD (Standard Deviation - C-index Proxy):")
for i, r in res_df.iterrows():
    print(f"{r['Horizon']} | S13: {r['S13_Std']:.4f} | S14: {r['S14_Std']:.4f} | S15: {r['S15_Std']:.4f}")

# 3. Correlation check
print("\nCORRELATION WITH PREVIOUS BEST (Sub_09):")
s09 = pd.read_csv('d:/WiDS/submission_09.csv')
for h in horizons:
    c13 = s13[f'prob_{h}h'].corr(s09[f'prob_{h}h'])
    c15 = s15[f'prob_{h}h'].corr(s09[f'prob_{h}h'])
    print(f"{h}h | S13 corr: {c13:.4f} | S15 corr: {c15:.4f}")
