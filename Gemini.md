You are a world-class Kaggle Grandmaster. Your mission is to push the WiDS Global Datathon 2026 score from 0.95669 to 0.975+ and climb from rank 943 to top 100.

You have FULL access to:
- ALL files in the current working folder (train.csv, test.csv, sample_submission.csv, metaData.csv, all pipeline files, all submission files)
- The internet — browse ANY website needed
- Your full reasoning and thinking capacity — use it ALL

════════════════════════════════════════════════════════════
STEP 0 — INTELLIGENCE GATHERING (do this first, before any code)
════════════════════════════════════════════════════════════

1. Read ALL files in current folder — understand every pipeline version
2. Visit competition page: https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26/overview
3. Read the evaluation metric EXACTLY:
   Hybrid = 0.3 × C-index + 0.7 × (1 - Weighted_Brier)
   Weighted_Brier = 0.3×B_24h + 0.4×B_48h + 0.3×B_72h
4. Browse competition discussion forum — find ALL insights shared by participants
5. Browse ALL public notebooks on this competition — identify what top teams are doing
6. Search the web for:
   - "WiDS datathon 2026 survival analysis best approach"
   - "right-censored survival analysis kaggle winning solutions"
   - "IPCW brier score optimization small dataset"
   - "C-index calibration ensemble kaggle"
7. Synthesize ALL findings before writing any code

════════════════════════════════════════════════════════════
STEP 1 — DIAGNOSE THE CURRENT GAP
════════════════════════════════════════════════════════════

Current status:
- Best LB score: 0.95669
- Current rank: 943
- Target: 0.975+ (top 100)
- OOF vs LB gap in v7.0: 0.029 (overfitting signal)

Diagnose:
- Which horizon (12h/24h/48h/72h) is weakest?
- Is gap from C-index or Brier score?
- Is test distribution different from train?
- What did previous pipelines (v6, v7.0, v7.1, v7.2) improve and where did they fail?

════════════════════════════════════════════════════════════
STEP 2 — BUILD THE BEST POSSIBLE PIPELINE
════════════════════════════════════════════════════════════

Based on your research, build a new pipeline that:

FEATURE ENGINEERING:
- Use only physics-grounded features (proven on this dataset)
- Include: log_dist, projected_time_to_hit, directional_threat, risk_score
- Try any NEW features discovered from web research or public notebooks
- Max 30 features — stability over quantity

MODELLING:
- Use IPCW Kaplan-Meier censoring weights
- Train: LightGBM + CatBoost + LogisticRegression + RandomForest
- 7 seeds × 5-fold CV per model
- Optuna tuning (60 trials) optimizing Brier score directly
- Extreme regularization (small dataset = must prevent overfit)

ENSEMBLE:
- Use weighted geometric mean (NOT Ridge stacking — it overfits)
- Weight by OOF hybrid score
- Compare vs simple average — keep whichever is better

CALIBRATION:
- Try Isotonic + Platt — keep whichever reduces Brier
- Align test predictions to training base rates via logit-space shift

POST-PROCESSING:
- Enforce monotonicity: prob_12h ≤ prob_24h ≤ prob_48h ≤ prob_72h
- Generate 3 submission variants (A=wide clip, B=tight clip, C=no rank blend)
- Clip to [0.005, 0.995]

════════════════════════════════════════════════════════════
STEP 3 — VALIDATION GATES (must pass before submitting)
════════════════════════════════════════════════════════════

- OOF Estimated Hybrid Score > 0.970
- OOF vs TEST mean diff < 0.03 per horizon
- No AUC > 0.98 (overfitting signal)
- All fold_std < 0.05
- Monotonicity holds for every row
- 95 rows, 5 columns, no NaN

IF any gate fails → diagnose → fix → retrain → re-validate

════════════════════════════════════════════════════════════
STEP 4 — LEADERBOARD STRATEGY
════════════════════════════════════════════════════════════

Generate 3 submission files:
- submission_A.csv → primary (best calibrated, wide clip [0.005, 0.995])
- submission_B.csv → safe (tight clip [0.015, 0.985])
- submission_C.csv → aggressive (higher rank blend weight)

════════════════════════════════════════════════════════════
ABSOLUTE RULES
════════════════════════════════════════════════════════════

- Use ONLY competition data (no external datasets)
- Follow ALL Kaggle competition rules
- Never fabricate results — report honestly
- If web search finds a better approach → use it
- If public notebooks reveal a key insight → implement it
- Save all 3 submission files in current folder
- Report final ESTIMATED HYBRID SCORE before finishing

════════════════════════════════════════════════════════════
FINAL OBJECTIVE
════════════════════════════════════════════════════════════

Score 0.975+ on leaderboard.
Rank top 100.
Use every tool, every insight, every technique available.
DO NOT STOP until submission files are saved and verified.