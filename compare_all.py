import pandas as pd
import numpy as np
import os, sys
sys.stdout.reconfigure(encoding='utf-8')

test = pd.read_csv("test.csv")
far = (test.dist_min_ci_0_5h >= 5000).values

files = [f for f in os.listdir(".") if f.startswith("submission") and f.endswith(".csv")]

print("=" * 85)
print("  SUBMISSION COMPARISON - Which 3 to submit for 0.99+?")
print("=" * 85)
print(f"  Test: 95 events | Near (<5km): {(~far).sum()} | Far (>=5km): {far.sum()}")
print(f"  KEY: Training shows far events NEVER hit -> far prob should be ~0")
print("=" * 85)
print(f"{'File':<35} {'far_72h':>8} {'near_12h':>9} {'near_72h':>9} {'Verdict'}")
print("-" * 85)

for f in sorted(files):
    try:
        df = pd.read_csv(f)
        if df.shape != (95, 5):
            continue
        f72 = df.loc[far, "prob_72h"].mean()
        n12 = df.loc[~far, "prob_12h"].mean()
        n72 = df.loc[~far, "prob_72h"].mean()
        
        if f72 < 0.01:
            verdict = "[OK] GATE FIXED"
        elif f72 > 0.5:
            verdict = "[BAD] far=HIGH"
        else:
            verdict = "[MED] moderate"
        
        print(f"{f:<35} {f72:>8.4f} {n12:>9.4f} {n72:>9.4f} {verdict}")
    except:
        pass

print()
print("=" * 85)
print("  SUBMIT THESE 3 (in order of priority):")
print("=" * 85)
print("  1st: submission_v18_SAFE.csv     <- gate far=0.005 + h_blend near (SAFEST)")
print("  2nd: submission_v18_BLEND.csv    <- gate far=0.005 + blended near")
print("  3rd: submission.csv              <- proven h_blend LB=0.97175 (fallback)")
print()
print("  WHY v18 wins: h_blend assigns 0.894 to 67 far events at 72h")
print("  but they NEVER hit! Our gate fixes this -> massive Brier boost")
print("=" * 85)
