import pandas as pd
train = pd.read_csv(r'd:\WiDS\train.csv')
c = train[train['event']==0]
print('Censored time distribution:')
print(c['time_to_hit_hours'].describe())
print()
for t in [12, 24, 36, 48, 54, 60, 66, 72]:
    n = (c['time_to_hit_hours'] >= t).sum()
    print(f'  censored with obs >= {t}h: {n}')
print()
h = train[train['event']==1]
print('Hit time distribution:')
print(h['time_to_hit_hours'].describe())
for t in [12, 24, 48, 72]:
    n_pos = (h['time_to_hit_hours'] <= t).sum()
    n_neg_hit = (h['time_to_hit_hours'] > t).sum()
    print(f'  {t}h: {n_pos} pos from hits, {n_neg_hit} neg from hits')
