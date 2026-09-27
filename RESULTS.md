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
- Leaderboard (public): **0.934** (+0.023 over v1: confirms the India over-matching diagnosis)

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
- Kaggle: train 55 min + predict 120 min; model validation F0.5 0.9887 (threshold 0.725)
- Per country: France 3.263 matches/S1 (6.1% singletons), India 3.365 (5.6%), US 3.384 (5.8%)
- Leaderboard (public): **0.947**
- **Bug found in the log:** the training slice held 18 US states and 1 tiny India state (716 queries) -
  v1-v3 were trained essentially without India (47% of test).

## v4 - stratified training + collective peer features + test-like threshold (4 parallel seeds)
- Code: commit `8fc0d46` (+ weighted `src/ensemble.py`)
- Training slice stratified per country; `train_frac` 0.3 (seeds 0, 1, 2, 3)
- Seed 3 was first run with `train_frac` 0.5 and ran out of memory after ~49 min (Kaggle 30 GB);
  rerun at 0.3. Keep `train_frac` <= 0.3 on 4-CPU / 30 GB sessions.
- New: peer features (siblings repeat their modifier word / new number across their records),
  threshold tuned with false merges x1.7 (test distractor density)
- Dev: F0.5 0.9919 (test-like 0.9911); `peer_num_share_p` is the #4 feature
- **Perf bug (commit after `d477d24`):** the pairwise `peer_both` loop was quadratic per S1; India test has S1s with
  thousands of candidates (country-wide no-state search), so predict stalled for hours after India stage 1.
  Fixed with a linear (token, number)-combination count; 520k pairs incl. a 20k-candidate group in 19 s.
  Training (seed 0 log): 74.6 min, slice India 4 states / 271k S1 + US 15 states / 454k S1, validation F0.5
  0.98453 (now includes India, so not comparable with v3's US-only 0.9887), threshold 0.75.
- Plan: submit the first finished seed alone, then a weighted ensemble of all four (`src/ensemble.py`)
- Leaderboard (public): **0.971** for each of seeds 0, 2, 3 (ensemble pending)

### v4 results (seeds 0, 2, 3; each run = train + predict on Kaggle 4 CPU / 30 GB)

| seed | training slice (India / US) | val F0.5 | test-like val F0.5 | threshold | train | predict | public LB |
|---|---|---|---|---|---|---|---|
| 0 | 7 states 272k S1 / 15 states 405k S1 | 0.98635 | 0.98579 | 0.800 | 70 min | 126 min | **0.971** |
| 2 | 10 states 268k S1 / 18 states 530k S1 | 0.98805 | 0.98752 | 0.750 | 80 min | 118 min | **0.971** |
| 3 | 8 states 292k S1 / 16 states 439k S1 | 0.98707 | 0.98645 | 0.775 | 73 min | 125 min | **0.971** |

- All three score **0.971** on the public LB (+0.024 over v3's 0.947): the gain comes from the
  India training data and the collective features, not from a lucky seed.
- Blocking recall on the training slices: 0.981-0.985 (large states; dev slice was 0.994).
  Stage 1 keeps 0.98-0.99 pairs per query at 99.95% recall of blocked positives.
- Top features (all seeds): p1_q_gap, p1, lex_extra_min, **peer_num_share_p (#4)**, legal_q_only,
  ex_qdf_max, p1_s_gap_top, p1_s_ncand, hn_logdiff, first_tok_eq, num_q_only, anc_p1.
- Predict-time calibration (target = validation predicted/S1 3.34-3.36):
  - **US now over-matches**: 3.45-3.47 matches/S1 at the base threshold, so calibration raised the US
    threshold to **0.960-0.976**.
  - India 3.27-3.28 and France 3.09-3.17 are *below* target (calibration never lowers thresholds).
  - Singletons: US 6.0%, India 6.5%, **France 7.0-7.2%** (prior 5.6%), which suggests v4 under-matches
    in France / India (recall) while the US over-matches before calibration.
- Stage-1 candidate pairs on test: France 1.38-1.41M, India 4.60-5.85M, US 3.65-3.68M.

### v4 ensemble (seeds 0+2+3, `src/ensemble.py`, equal weights)
- Leaderboard (public): **0.971**, identical to each single seed. The seeds make the same mistakes, so the
  remaining error is systematic, not variance; more seeds will not help.

## v5 - neighbour-state blocking (predict-only on v4 models)
- Code: commit `cde27ca`
- Diagnosis on a labelled full-size TRAIN slice (IN-TG + IN-AP + US-OH, 158k S1, `experiments/diag/`):
  loss was mostly FN from blocking (0.045 of F0.5); 82% of India's blocking misses were Telangana records
  written "Andhra Pradesh" (18.5% of all Telangana true pairs in train; DC<->WA similar, small).
- Fix: S1 partition of a state also searches the queries of its neighbour states; `state_eq` uses state
  groups so existing models accept these pairs. Slice: India TG/AP ~0.87-0.90 -> 0.984, US 0.990 unchanged.
- Other ideas tested on the slice and rejected: per-S1 expected-F0.5 decoding (+0.0003 only); rescuing
  renames / empty-address exact names in France (only ~87% precision, below the ~0.78 break-even of
  the rejected subset).
- Runs: predict-only with v4 seed-0 artifacts (SEED 100) and seed-2 artifacts (SEED 102), running.
- Leaderboard (public): _pending_
- Seed 100 turned out to run the OLD code (identical to v4 seed 0, 8 of 5.67M pairs differ); seed 102 has the
  fix (India 3.284 -> 3.326 matches/S1 = recovered Telangana records). Submission file: submissions/v5_seed102.

## v6 - band-aware per-country thresholds + structural rescue (post-processing on v5 seed-102 scores)
- Code: `src/postprocess.py`; analysis scripts `experiments/diag/bands.py`, `band_look*.py`, `nearnum_*.py`
- Evidence: records per S1 per probability band, test vs a labelled train-like slice (same pipeline):
  US top band 3.294 vs 3.295 (identical), but US 0.775-0.97 bands inflated 2-4x -> est. precision 25-55%;
  India close to training (0.775-0.9 est. precision 0.66-0.69); France top band 2.895 (0.4/S1 lower) and
  0.775-0.99 inflated 4-6x. Hand check of 44 accepted France pairs in 0.775-0.99: 28 siblings (Développement,
  Groupe, & Fils, France, word swaps, SASU with new number), 16 true, all of three kinds: initials,
  rename at the exact address, empty address with exactly the S1 name.
- A 'near-number sibling' hard rule was rejected: on train-like data the flagged accepted pairs are 98.7% true.
- v6: thresholds France 0.99, India 0.9, US 0.96 (= seed-102 calibration); rescue down to p >= 0.5 of
  initials / rename-at-exact-address / empty-address-exact-name (France 29k, India 12k, US 15k pairs).
- Matches per S1: France 3.172 -> 3.006, India 3.326 -> 3.312, US 3.360 -> 3.385. Validator PASS.
- Leaderboard (public): 0.974 (file: submissions/v6_bandfix) - thresholding alone can't remove siblings the
  model is confident about.

## v8 - retrain with synthetic test-style sibling distractors (augment.py)
- Idea: test has a sibling type training barely contains (US: same core name + legal form changed + house number
  within +-20, about 35x over-represented in the 0.5-0.9 band; France: same address with one name word swapped).
  The model learned training odds and accepts them. Fix at the source: make such records from REAL true S2/S3
  records of training S1s (so they keep the source's noise) and add them to training as unmatched records.
  - type A (rate 0.15 per S1): legal form stripped/replaced + house number shifted 1..20
  - type B (rate 0.10 per S1): one core name word swapped for a common name word of the same country
- Dev run (train slice): 21,247 synthetic records (0.18 per S1; US 11.8k, India 9.4k); blocking recall 0.994;
  validation including the synthetics F0.5 0.9900 plain, 0.9889 test-like, at threshold 0.75.
- Kaggle: SEED 7, FORCE_TRAIN, train_frac 0.3. Matches per S1: France 3.087, India 3.334, US 3.344.
  Leaderboard (public): **0.977** (+0.001 over v7): the model rejects the synthetic siblings, but real test
  siblings are mostly not of the generated kinds.

## v7 - raw seed 202 (v4 seed-2 model + wider blocking + OCR fold, model's own calibration)
- Matches per S1: France 3.209, India 3.350, US 3.360. Leaderboard (public): **0.976**.
- Lesson: v6 (same model family, older blocking, blanket thresholds France 0.99 / US 0.96 / India 0.9) scored 0.974;
  the blocking/OCR change is worth ~+0.002 in-distribution, so the blanket thresholds likely COST ~0.002.
  France's 0.5-0.99 band holds true word-swap matches as well as siblings (FINDINGS 5.16). Trust the model's calibration.
- Built but not submitted: raw 202 + camp-B drop (src/camp_drop.py); estimated +0.001-0.002 (ceiling ~+0.004).


## v9 - v8 + pseudo-labelled France (domain adaptation for the country without training data)
- France is the biggest leak (FINDINGS 5.16): the model never saw French names/addresses/tokens. v9 adds ~half of the
  France test S1 as training entities: records the seed-202 model assigned with p >= 0.99 are matches, records whose
  best score is < 0.05 (or never scored) are distractors, the uncertain middle is left out. Pseudo S1 never enter
  validation (threshold/calibration on real labels only). augment.py then also builds French siblings from real
  French names (type A French legal forms + number shift, type B French word swap).
- Local end-to-end test (train_frac 0.01, 5% of France): 12,921 pseudo S1, 2.93 matches/S1, 1.91 distractors/S1;
  2,909 French synthetic siblings; blocking recall 0.9986; validation (real only) F0.5 0.9904.
- Kaggle: SEED 9, train_frac 0.25, pseudo_frac 0.5, seed-202 bundle attached, predict France only
  (`predict_countries`; ~1h45m instead of ~3h20m). Final file = v9 France + v8 US/India
  (`src/merge_countries.py`). Leaderboard: _pending_
- Also in v9: a third blocking channel, combined name+address TF-IDF (cosine = (name_cos + addr_cos) / 2). France
  names are generic (one core name shared by many businesses: 35% of France's never-scored records have an S1 with
  the identical core name, vs 14% in the US) and addresses hold many businesses, so the separate top-K lists are full
  of ties; the S1 agreeing on both now ranks first. Idea also found independently by a teammate's session
  (France 'no match' S1 6.87% -> 6.10%). Local end-to-end test passed (validation F0.5 0.9908 on a tiny slice).
- Checked and rejected tonight: India never-scored records (0.61/S1, 57% native-script names) have no S1 twin -
  distractors, not a recall leak; true singleton rate 5.58% in train vs 5.9% predicted - no singleton lever.

### v9 outcome (run locally: train_frac 0.05 + 40% of France pseudo-labelled, France-only predict, 20:29-22:33)
- Validation (real labels) F0.5 0.98828. France: 3.038 matches/S1, 6.95% singletons (v8 3.087 / 6.71%).
- Hand check of 28 France pairs where v9 and v8 disagree: v9 DROPS mostly true matches (empty-address exact names,
  street typos, initials - the old model's pseudo-label biases) and ADDS mostly siblings ('Et Fils', '& Fils',
  'Groupe'). Net ~13k fewer true matches -> v9 France is worse than v8. NOT submitted.
- France has 501k of 1.43M records without a recognised region (departments: Gironde, Nord, Loire-Atlantique ...):
  they are searched country-wide - slow and tie-heavy. Mapping departments to regions is the obvious next fix.

## v10 - v8 + the combined channel's new France candidates
- v9-accepted France pairs that were NOT v8 candidates and whose record v8 left unassigned: 4,319 (0.017/S1).
  24/24 sampled look true: street typos ('Propraétaires', 'Mguuet', 'RTSPAIL', 'Wisnton') that the address channel
  missed while the name channel was full of same-name businesses. Only these are added to v8; nothing removed.
- France 3.087 -> 3.104 matches/S1, singletons 6.71% -> 6.57%. Validator PASS. File: submissions/v10_v8_plus_newcand.
- Expected ~+0.0003 over v8 (0.977) -> not submitted. FINAL BEST: v8 = 0.977.
- Rejected tonight: exact-twin rescue (same name + number + street, unassigned) - samples are mostly siblings with a
  changed sub-number (N5490 vs N5479, Flat A/1025 vs A/1014) that the model correctly rejects.


---

# Second line of work (Claude, 2026-09-26/27): v5, v6, cross-encoder, ensemble -> public LB 0.984

Code: `solution/business_entity_resolution` (submitted line; the final package's code), `solution/v6_variant` (seed-1 variant with
neighbour-state search and frequency features), model artifacts in `solution/artifacts/`. Full method: `APPROACH.md`; audit and plan: `PLAN.md`.

## v4-local - same code, trained locally on the stratified 30% slice (2026-09-26)
- Machine: 18-core Snapdragon laptop, Windows x64 Python; train 127 min under heavy CPU contention
  (three jobs at once), predict see below. Config `runs/cfg_s0.json` = `{"train_frac": 0.3, "seed": 0}`.
- Training slice: India 7 states / 272,504 S1, US 15 states / 405,400 S1 (0.31 of each country).
- **Blocking recall on this slice: 0.98106** (28.7M pairs, 44,435 true pairs missed) against 0.994 on the
  small 9-state dev slice: big partitions (Delhi 44% of the misses, Texas, Uttar Pradesh, Ohio) and
  empty-address records (46% of the misses) are where recall goes at scale.
- Stage-1 keeps 0.98 pairs per record (recall kept 0.99952); stage-2 best iteration 1015.
- **Validation macro F0.5 0.98584** (test-like, fp x1.7) at threshold 0.75 on 135,052 held-out entities;
  predicted 3.348 matches per S1 against 3.458 true (ratio 0.968). The 9-state dev slice said 0.9918:
  every earlier dev number was optimistic by about 0.006.
- Leaderboard (public): _submission #1 of 2026-09-26, pending_

## v5 - blocking x2, composite identifiers, France city->region backfill, expected-F0.5 decision (2026-09-26)
- Code: `v5/business_entity_resolution` (to be merged into `code/`).
- Blocking: top-20 per channel, keep the top-10 by name, by address and by their sum (new combined channel),
  40 by name for records without a state; composite identifiers ("17/34" -> 1734) as `ID_` tokens in the
  address channel. Dev recall 0.99427 -> 0.99677 (0.99690 with the combined channel on two states).
- Features: composite identifier jaccard / edit distance / replaced vs added (India: "H.no 332", "4-1-10/1/2").
- France: 35% of France S2/S3 addresses end with the city and no region; a city -> region map learned from
  S1 addresses backfills them so they get state partitions and the state-agreement feature.
- Decision: per-entity expected-F0.5 (empty prediction included, 2% unfound allowance) chosen automatically
  when it beats the fixed threshold on validation; per-country calibration shrinks the odds instead of
  raising a threshold. Negative weighting 1.9 was tried and gave nothing on top of it (dev B).
- Dev (9 states): F0.5 0.99239 (v4 0.99182 with the same decision rule), test-like fp x1.9 0.99174 (v4 0.99116).
  Loss decomposition: blocking loss 0.00186 -> 0.00099, distractor false merges 0.00081 -> 0.00068.
- Full 30% slice (same 22 states as v4-local): **blocking recall 0.99215** (v4 0.98106; 62.7M pairs, 18,406 true
  pairs missed instead of 44,435), stage 1 keeps 1.06 pairs per record; stage-2 best iteration 1084;
  **validation F0.5 0.99060, test-like (fp x1.9) 0.99001** with the expected-F0.5 decision (fixed threshold 0.775:
  0.98983) on the same 135,052 entities where v4 scored 0.98584; predicted 3.378 matches per S1 (ratio 0.977)
- Leaderboard (public): _pending_


## v6 / v7 options (2026-09-27, night)
- v6 = v5 + label-free name / address frequency features (how many S1 of the country share the exact core name
  or address key; ambiguity is 16% of the full-scale loss). Dev: blocking recall 0.99721, F0.5 0.99242, test-like
  0.99180 (v5 0.99239 / 0.99174). Seeds 1 and 2 train overnight (`v5/overnight.sh`), ensemble in `v6/out_ens`.
- Prediction fix: `peer_features` had a per-entity loop quadratic in the candidate count; at test time popular names
  give entities thousands of candidates and both predictions sat for hours in it. Groups above 300 candidates now
  use a counter approximation. Predictions also checkpoint each country (`OUT/scores_<country>.parquet`).
- France early look (checkpoints): v4 3.172 matches/S1, 6.9% singletons; v5 3.297 matches/S1, **5.60% singletons**
  (prior 5.6%). v5 rejects 15.6k pairs v4 accepted, almost all French siblings (SASU/SAS swaps, "Groupe",
  "Développement", "& Associés", changed numbers) and accepts 48k v4 missed (mostly true: domain names, typos,
  records v4 never retrieved). Remaining weakness: French word substitutions (Comite -> Club, Amis -> Gestion).
- v7 option `predict --selftrain-lex` (v6 code): a per-country lexicon learned from the run's own confident
  decisions (p >= 0.97 vs p <= 0.03), training lexicon kept for known words. Mini France slice (Pornic + Calais,
  24k S1): 56 new extra words, the most distractor-like being holding, participations, developpement, groupe,
  parents, fetes, sante, anciens, SNC, EI, foyer, culture, loisirs; 2.4% of pairs crossed 0.5.

## Submissions of 2026-09-27 (deadline 21:00 IST)
Teammate's remote: v4 seeds each 0.971 public LB (ensemble 0.971). All files below are v5 scores assembled with
`src.assemble` from per-country checkpoints and validated (PASS).

| file | decision | France m/S1, single | India m/S1, single | US m/S1, single | public LB |
|---|---|---|---|---|---|
| matching_results_v5_odds19.tsv | expected-F0.5, odds/1.9 | 3.236, 5.85% | 3.299, 6.17% | 3.433, 5.67% | **0.975** |
| matching_results_v5_raw.tsv | expected-F0.5, raw (US calibrated odds/1.55) | 3.297, 5.60% | 3.317, 6.08% | 3.444, 5.65% | _pending_ |
| matching_results_v5_bands.tsv | fixed thresholds FR 0.99 / IN 0.9 / US 0.968 | 3.006, 7.16% | 3.285, 6.35% | 3.353, 5.97% | _pending_ |
| matching_results_v7_fr_*.tsv | v5 + France self-trained lexicon | | | | _pending_ |

## Cross-encoder (xlm-roberta-base, MIT) on an L40S, 2026-09-27
- Trained 1 epoch on 1.5M stage-1 pairs of the v5 30% slice (negatives capped at 3 per positive, hardest kept),
  max length 128, batch 64, lr 2e-5, fp16: 33 min. Held-out pairs: logloss 0.0437, accuracy 0.9842.
- Blend in logit space with the LightGBM probability, evaluated on the 135k validation entities with the
  expected-F0.5 decision (`src/blend_ce.py`):

| weight on CE | F0.5 | test-like (fp x1.9) |
|---|---|---|
| 0.00 (LightGBM only) | 0.99060 | 0.99001 |
| 0.20 | 0.99174 | 0.99127 |
| 0.35 | 0.99180 | 0.99124 |
| 0.50 | 0.99117 | 0.99035 |
| 1.00 (CE only) | 0.98666 | 0.98409 |

- Weight 0.3 chosen for the test blend (`v5/ce_w.txt`); test scoring of the 12.2M pairs runs on the L40S.

### Label-free band calibration (`src.assemble --band-ref`, 2026-09-27 11:30)
Records per S1 per probability band on the validation entities of the v5 slice (train-like reference, same
pipeline) against the test: inflation of the test bands [0.3-0.5, 0.5-0.6, 0.6-0.7, 0.7-0.775, 0.775-0.85,
0.85-0.9, 0.9-0.95, 0.95-0.97, 0.97-0.99] = France 6.4 4.7 4.4 5.7 6.6 5.6 4.1 2.9 2.1; India 2.4 1.8 1.6 1.6
1.9 1.9 1.8 1.6 1.5; US 2.9 2.7 2.6 3.7 4.3 3.9 2.7 1.8 1.4 (top band >= 0.99 trusted). Extra records in a band
are distractors, so p -> p / inflation before the expected-F0.5 decision. Result (v5 scores):
France 3.010 matches/S1 (6.9% singletons), India 3.174 (6.3%), US 3.302 (5.9%): submissions/v5_bandcal.

### Blended variants (v5 LightGBM + cross-encoder w=0.3), 2026-09-27 12:50, all validated
| file | decision | France | India | US | public LB |
|---|---|---|---|---|---|
| matching_results_blend_odds19.tsv | expected-F0.5, odds/1.9 | 3.252, 5.72% | 3.301, 6.17% | 3.379, 5.75% | |
| matching_results_blend_bandcal.tsv | band calibration (blended reference) + expected-F0.5 | 3.009, 6.53% | 3.166, 6.23% | 3.351, 5.82% | **0.973** (conservative decisions cost; teammate's blanket thresholds 0.974 vs their raw 0.978) |
| matching_results_blend_v7_bandcal.tsv | same + France self-trained lexicon | 3.007, 6.57% | 3.166 | 3.351 | |
Band inflation with the blended reference: France 6.0 4.7 3.6 3.3 4.9 6.4 4.5 3.0 1.6; India 2.0 1.8 1.6 1.4 1.5 1.9 1.8 1.7 1.5;
US 3.8 2.6 1.9 1.6 1.7 1.7 1.3 1.2 1.0 (the blend already sharpened the US mid bands).
xlm-roberta-large (MIT) training started 12:47 on the L40S for the final blend (37,500 steps, ~95 min, then scoring).

### Two-seed ensemble (v5 s0 + v6 s1, mean per pair; union 7.0M India / 3.96M US / 1.92M France pairs), 13:00
| file | decision | France | India | US |
|---|---|---|---|---|
| matching_results_ens_blend_odds19.tsv | CE blend + expected-F0.5 odds/1.9 | 3.225, 5.79% | 3.341, 5.81% | 3.379, 5.73% |
| matching_results_ens_blend_bandcal.tsv | CE blend + band calibration (reference not recomputable for the ensemble: over-conservative, France 2.853) | 2.853, 7.56% | 3.153 | 3.352 |
v6 seed 1: training slice India 4 states 272k S1 / US 15 states 454k S1 (seed 1), blocking recall 0.99135, validation
F0.5 0.98932 / test-like 0.98864 on its own 144,853 entities (states differ from seed 0; not comparable directly).

### xlm-roberta-large (16:07): no gain over the base cross-encoder
Held-out logloss 0.0469 (base 0.0437); blend at w=0.2: F0.5 0.99173 / test-like 0.99123 (base model 0.99174 / 0.99127).
Not used. Final candidates remain the base-model blends: blend_bandcal / blend_v7_bandcal (calibrated line) and
ens_blend_odds19 (ensemble line); the choice depends on the leaderboard score of upload 4 (blend_bandcal).

### Structural post-rules on the blended validation (16:20): no gain
Dropping mid-confidence records with a replaced identifier, a sibling group (shared new number + word), an added legal
form, or a modifier word before the expected-F0.5 decision is neutral or worse (baseline 0.99168 / test-like 0.99115;
best rule 0.99168 / 0.99117): the model's features and the per-entity decision already cover them. Not applied.

### Final upload (16:40): matching_results_ens_blend_raw.tsv
Two-seed ensemble (v5 s0 + v6 s1) + cross-encoder blend (w 0.3) + raw expected-F0.5 decision with the per-country
shape guard only. Chosen because both leaderboard readings (ours 0.973 < 0.975, teammate 0.974 < 0.978) show that
extra conservatism costs, and the raw decision is also the best on the full-scale validation.
Package: submissions/FINAL_ens_blend_raw_package.zip.

### Final upload (18:35): matching_results_ens3_blend_raw.tsv
Three-model ensemble (mean per pair over the runs that scored it): our v5 seed 0, our v6 seed 1, and the teammate's
seed 202 (public LB 0.978 alone; pair scores from their download_me_seed202.zip), blended with the cross-encoder
(w 0.3, covers 89-97% of pairs), raw expected-F0.5 decision with the shape guard. France 3.281 / India 3.358 /
US 3.395 matches per S1, singletons 5.6-5.8%; union of 13.26M candidate pairs; validator PASS with --check-ids.
Package: submissions/FINAL_ens3_blend_raw_package.zip. **Public LB: 0.984** (from 0.978).
