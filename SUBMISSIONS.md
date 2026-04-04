# WiDS 2026 - Submission Guide

## Which 3 to Submit (in order)

### 1st Priority: `submission_v18_SAFE.csv`
- **Strategy:** 5km Distance Gate + h_blend near-zone predictions
- **Far zone (67 events, dist >= 5km):** Probability = 0.005 (they NEVER hit in training)
- **Near zone (28 events, dist < 5km):** Uses proven h_blend predictions (0.97175 LB)
- **Why best:** Fixes the massive 72h Brier score error in all other submissions
- **Expected score:** 0.99+

### 2nd Priority: `submission_v18_BLEND.csv`
- **Strategy:** 5km Distance Gate + 50/50 blend of our ML models and h_blend for near zone
- **Far zone (67 events):** Probability = 0.005 (same fix)
- **Near zone (28 events):** Blend of LightGBM predictions + h_blend predictions
- **Why second:** Same far-zone fix but different near-zone approach
- **Expected score:** 0.98-0.99+

### 3rd Priority (fallback): `submission.csv`
- **Strategy:** h_blend of 4 public notebooks (proven LB = 0.97175)
- **All 95 events:** Uses rank-weighted ensemble of 4 top Kaggle notebooks
- **Why fallback:** Proven on leaderboard but has a flaw at 72h for far events
- **Expected score:** 0.97175 (proven)

---

## The Key Insight: 5km Distance Gate

Training data reveals a PERFECT split:
- Under 5km distance: 69/69 events = 100% hit rate
- Over 5km distance: 0/152 events = 0% hit rate

The h_blend (and ALL other submissions except v18) assigns prob_72h = 0.894 to
far-zone events. But they NEVER hit! This creates a massive Brier score penalty.

Our v18 fixes assign 0.005 to far events, which should dramatically improve
the Weighted Brier Score component (worth 70% of the final metric).

---

## All Submission Files

| File | Description | Far 72h | Status |
|------|-------------|---------|--------|
| submission_v18_SAFE.csv | Gate + h_blend near | 0.005 | SUBMIT 1ST |
| submission_v18_BLEND.csv | Gate + blended near | 0.005 | SUBMIT 2ND |
| submission_v18_GATE.csv | Gate + pure ML near | 0.005 | Backup |
| submission.csv | h_blend 4-model LB=0.97175 | 0.894 | SUBMIT 3RD |
| submission_hblend.csv | h_blend backup copy | 0.894 | Same as above |
| submission1.csv | Downloaded reference | varies | Old |
| submission_09.csv | Pipeline v9 | varies | Old |
| submission_5model_avg.csv | 5-model weighted average | 0.979 | Not recommended |
| submission_5model_rank.csv | 5-model rank blend | 1.000 | Not recommended |
| submission_MEGA.csv | Mega blend | 0.894 | Not recommended |
| submission_v17_A.csv | 6 ML models + h_blend | 0.351 | Not recommended |
| submission_v17_B.csv | Pure ML ensemble | 0.575 | Not recommended |
| submission_v17_C.csv | Safe ML blend | 0.360 | Not recommended |
| submission_v17_OPTIMAL.csv | Per-horizon optimal | 0.894 | Not recommended |
