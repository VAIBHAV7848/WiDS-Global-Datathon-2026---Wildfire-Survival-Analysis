import pandas as pd
import numpy as np

t = pd.read_csv('train.csv')
print('Correct targets (event=1 AND time<=h):')
for h in [12,24,48,72]:
    y = ((t.time_to_hit_hours <= h) & (t.event == 1)).astype(int)
    print(f'  y_{h}h: {y.sum()}/{len(t)} positive ({y.mean():.3f})')

print()
print('WRONG targets (time<=h regardless of event):')
for h in [12,24,48,72]:
    y = (t.time_to_hit_hours <= h).astype(int)
    print(f'  y_{h}h: {y.sum()}/{len(t)} positive ({y.mean():.3f})')

print()
print('Difference (censored events incorrectly labeled as hit):')
for h in [12,24,48,72]:
    y_correct = ((t.time_to_hit_hours <= h) & (t.event == 1)).astype(int)
    y_wrong = (t.time_to_hit_hours <= h).astype(int)
    diff = (y_wrong - y_correct).sum()
    print(f'  {h}h: {diff} extra positives from censored events')
