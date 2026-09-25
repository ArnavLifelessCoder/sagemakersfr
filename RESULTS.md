# Submission log

Output files are not committed (too large for git). Keep each run's `output.zip` locally, named `output_vN_*.zip`.

## v1 — baseline (2026-09-25)
- Code: commit `3f77d44` (two-stage blocking + LightGBM, `train_frac=0.25`, threshold 0.70 from validation)
- Dev validation macro F0.5: 0.9907 (regional train slice)
- Official validator: PASS (1,732,544 rows; matches ⊂ candidates)
- Leaderboard (public): **0.911** (leader at the time: 0.9859)

| country | S1 | predicted singletons | matches / S1 | candidates / S1 |
|---|---|---|---|---|
| France | 259,452 | 5.45% | 3.42 | 5.15 |
| India | 809,986 | 3.50% | 3.83 | 8.76 |
| US | 663,106 | 5.81% | 3.38 | 5.35 |
| train prior | | 5.6% | ~3.46 | |

Spot-check: France has **word-substitution false merges** (3 of 12 sampled clusters), e.g.
"Mer Team Comite" ← "Mer Team Primaire SASU", "Pluriels Maison SAS" ← "Pluriels Federation SAS",
"Chantiers Soins" ← "Chantiers Ecole". Next: France fix (FINDINGS.md §6 task 1).

Diagnosis of v1 (details in FINDINGS.md section 5): test has ~1.7x more distractors per S1 than train; India
over-matches (+0.35 matches/S1, singletons 3.5% vs 5.6% prior) because test-only native-script business-type
words are missing from the transliteration table.

## v2 - test-time adaptation (predict-only, v1 model)
- Code: commit `798e696` (phonetic transliteration fallback + per-country threshold calibration, tol 2%)
- Kaggle: predict-only, 115 min on 4 CPUs; official validator PASS
- Model validation F0.5 (Kaggle v1 training): 0.9883
- Thresholds: India 0.9707 (calibrated from 0.75), US 0.75, France 0.75
- Leaderboard (public): _pending_

| country | S1 | matches / S1 | singletons |
|---|---|---|---|
| France | 259,452 | 3.422 | 5.45% |
| India | 809,986 | 3.375 | 5.47% |
| US | 663,106 | 3.376 | 5.81% |

India pairs removed by calibration are almost all sibling fakes, but about half of the kept 0.97-0.99 band are
siblings too (FINDINGS.md 5.7), so v2 is only a partial fix.

## v3 - modifier-word + anchor features, lexicon dropout, leet fix (full retrain)
- Code: commits `59b53c6`, `397a734`, `4030128` (notebook bundle)
- Dev validation F0.5: 0.9916 (v2 model 0.9907). On a real test slice it rejects the v2 sibling false merges
  (FINDINGS.md 5.8).
- Kaggle: full train + predict, running
- Leaderboard (public): _pending_
