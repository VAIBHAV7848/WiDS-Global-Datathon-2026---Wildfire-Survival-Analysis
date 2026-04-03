import pandas as pd
import numpy as np

files = ['submission_09.csv', 'submission_13.csv', 'submission_14.csv', 'submission_15.csv']
cols = ['prob_12h', 'prob_24h', 'prob_48h', 'prob_72h']

print(f"{'File':25s} | {'H':3s} | {'Mean':6s} | {'Std':6s} | {'X<0.01':6s} | {'X>0.99':6s} | {'Corr w/ Sub09'}")
print("-" * 90)

sub09 = pd.read_csv('d:/WiDS/submission_09.csv')

for f in files:
    fpath = f'd:/WiDS/{f}'
    df = pd.read_csv(fpath)
    for c in cols:
        m = df[c].mean()
        s = df[c].std()
        low = (df[c] < 0.01).sum()
        high = (df[c] > 0.99).sum()
        corr = df[c].corr(sub09[c])
        print(f"{f:25s} | {c[5:8]:3s} | {m:6.4f} | {s:6.4f} | {low:4d} | {high:4d} | {corr:7.4f}")
    print("-" * 90)
