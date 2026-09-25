# Submission log

Output files are not committed (too large for git). Keep each run's `output.zip` locally, named `output_vN_*.zip`.

## v1 — baseline (2026-09-25)
- Code: commit `3f77d44` (two-stage blocking + LightGBM, `train_frac=0.25`, threshold 0.70 from validation)
- Dev validation macro F0.5: 0.9907 (regional train slice)
- Official validator: PASS (1,732,544 rows; matches ⊂ candidates)
- Leaderboard (public): _pending_

| country | S1 | predicted singletons | matches / S1 | candidates / S1 |
|---|---|---|---|---|
| France | 259,452 | 5.45% | 3.42 | 5.15 |
| India | 809,986 | 3.50% | 3.83 | 8.76 |
| US | 663,106 | 5.81% | 3.38 | 5.35 |
| train prior | | 5.6% | ~3.46 | |

Spot-check: France has **word-substitution false merges** (3 of 12 sampled clusters), e.g.
"Mer Team Comite" ← "Mer Team Primaire SASU", "Pluriels Maison SAS" ← "Pluriels Federation SAS",
"Chantiers Soins" ← "Chantiers Ecole". Next: France fix (FINDINGS.md §6 task 1).
