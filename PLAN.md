# Plan to 0.99+ (public and private leaderboard)

Written 2026-09-26 after an independent audit of the code, the docs and the data
(everything below was measured on this machine; scripts are in `experiments/`).
Status of the inherited work: v3 = 0.947 public LB, v4 (stratified training, peer features,
4 seeds) submitted but result not yet recorded here.

---

## 1. What I verified today

### 1.1 Reproduction
- Dataset downloaded from Kaggle (`data/student_resource`, 2.5 GB, 10 files) and copied into WSL.
- Dev subset rebuilt (`dev/`: 116,791 S1, 546k S2/S3 in 9 small states).
- v4 dev training reproduced locally in **5.5 min** (GPT-6 needed 11): F0.5 0.9918, test-like 0.9910.
  Same numbers as RESULTS.md, so the code and the environment are sound.
- Environment: Windows x64 Python 3.11 (emulated on the Snapdragon) is the fastest for LightGBM
  training (8.8 s / 1M rows x 60 x 200 rounds vs 45 s for the native ARM build); WSL Ubuntu has a
  full native env too (`~/ber`, micromamba `ber`) and is 2-9x faster for string ops, prediction and
  fork-based multiprocessing. Full local training and prediction are feasible: 18 cores, 48 GB.

### 1.2 The generator (train, both countries identical)
| quantity | value |
|---|---|
| true records per S1 | 3.46 (S2 1.67 + S3 1.79), singletons 5.58% |
| cluster size 0/1/2/3/4/5/6+ | 5.6 / 5.4 / 17.0 / 24.1 / 21.9 / 14.6 / 11.4 % |
| singleton is a separate draw | P(n2=0, n3=0) = 5.6% vs 1.6% if independent; given non-singleton n2, n3 nearly independent |
| distractors (unmatched S2/S3) per S1 | 1.21 |
| empty address | **4.5% of true records, 0.3% of distractors** (a 15x prior, not used explicitly yet) |
| native-script names (India) | S2 23.5%, S3 13.2% |

### 1.3 The train -> test shift, decomposed (`experiments/eda_shift.py`)
For the same states, every S2/S3 record was matched to its nearest S1 (name and address TF-IDF,
like the blocking) and binned by best-name cosine and house-number agreement.

| US bins: strong name and same number / strong name and other number / mid / weak | share |
|---|---|
| train, true records | 0.655 / 0.134 / 0.192 / 0.018 |
| train, distractors | 0.011 / 0.598 / 0.370 / 0.021 |
| **test, all records** | 0.393 / 0.329 / 0.255 / 0.022 |

Solving the mixture: test = **59% true + 41% distractor** in both the US and India slices
(alpha = 0.593 and 0.588; every bin fits). With 5.6-5.8 records per S1 that is 3.3 true records
per S1 (exactly the training prior) plus **2.3 distractors per S1, 1.9x the training density**,
and the distractors have the *same* feature distribution as training distractors.
Orphans (records with no similar S1 at all) are 0.05-0.3% and can be ignored.

Consequences: a model trained at 1x density is over-confident at 1.9x; the F0.5-optimal
threshold is higher on test; sibling groups are larger on test, which makes collective
(peer) evidence *stronger*, not weaker; the empty-address rate dilutes exactly as predicted
(2.3-3.1% in test S2/S3), confirming the extra records are distractors, not lost true records.

### 1.4 Where the dev loss goes (`experiments/loss_decomp.py`, v4 model, threshold 0.75)
Validation loss 0.0084 (F0.5 0.9916) on 23,384 S1:

| category | share of loss | what it is |
|---|---|---|
| true record below threshold | 56% | random renames at the S1 address ("Nylaquo" p=0.08), added generic words ("Enterprises", "Services", "Center") with empty or partial address, dropped words |
| lost in blocking | 22% | 0.57% of true pairs; 60% are empty-address records searched country-wide, the rest are sub-number edits ("V/482" vs "V/782"), truncations ("1021" vs "021"), random renames with a partial address |
| false merge with a distractor | 10% | sibling with legal form added plus new number; identical name with a far house number ("360 Tower Road" vs "64 Tower Rd" accepted at p=0.99) |
| singleton given a match | 6% | same patterns, but the S1 has no true records so the score is 0 |
| record belongs to another S1 | 5% | two S1 at one address or with one name; the record has an empty address or a random name (mostly irreducible, abstaining is right) |
| lost in stage 1 | 1% | |

On test the false-merge categories roughly double (1.9x distractors) while the recall
categories stay, so precision work and recall work are about equally valuable.

### 1.5 Structural weaknesses found
- **India house numbers**: the parsed house number agrees for only **17%** of true India pairs
  (83% in the US). Indian identifiers are composite ("4-1-10/1/2", "Plot No-176", "H.no 332",
  "No 6-1887/7775", "KH NO. -570/13") and the parser only reads a leading integer. Half of the
  test is India, so the number logic that separates siblings is nearly blind there.
- **Dev slice is unrepresentative**: 9 small states, no big cities (name collisions), no France,
  1x distractor density. Every dev number is optimistic. v1-v3 were also trained almost without India.
- **Empty-address records** (0.16 per S1) are searched name-only against the whole country and
  compete with every same-name business; they are the biggest blocking loss and a source of
  cross-S1 confusions. Their strong prior (15x more likely true) is not used.
- **Decision rule**: one global threshold, then per-country shape calibration. The first record
  attached to an S1 carries singleton risk (a wrong first record scores 0), later records do not;
  a fixed threshold cannot express that.

---

## 2. Strategy

Three tracks, in order of expected leaderboard value per day of work:

A. **Make validation test-like** so every later decision is measured correctly.
B. **Close the systematic gaps** (blocking recall, India identifiers, calibrated decisions, training data).
C. **Exploit test structure** (collective sibling reasoning, transductive adaptation), then
   **capacity** (ensembles, a multilingual cross-encoder) if still short of 0.99.

Everything is scored three ways before it is trusted: dev-v2 F0.5 with 1.9x distractors,
the per-category loss decomposition, and the label-free test proxies per country
(matches per S1 near 3.3-3.4, singletons near 5.6%, modifier-word false-merge share).

---

## 3. Work items

### Phase 0 - infrastructure and a trustworthy validation set (first)
| # | item | why | done when |
|---|---|---|---|
| 0.1 | `dev-v2`: stratified slice with big states (CA, TX, NY, FL; MH, DL, KA, TN, UP, GJ) plus the small ones, ~25% of S1 per country, validation by S1 hash | name collisions and scale effects are invisible in the 9-state slice | built, train runs locally in < 1 h |
| 0.2 | distractor augmentation to 1.9x: a re-noised copy of every unmatched record (case, abbreviation, zero-padding, wrapper junk, dropped component) | reproduces test density for both features (peer/context) and the metric; keeps `fp_weight` as a cross-check | dev-v2 matches/S1 and singleton behaviour move the way test v1 did |
| 0.3 | proxy report per country on the full test after each candidate model (`src/proxy_report.py`, extended with the mixture estimate of true share) | the only label-free view of France | table in RESULTS.md per submission |
| 0.4 | loss decomposition per country as a standard step (`experiments/loss_decomp.py`, done) | know what each change fixes and breaks | |

### Phase 1 - systematic fixes (expected: the bulk of the gap)
| # | item | expected gain | notes |
|---|---|---|---|
| 1.1 | **Blocking recall 99.4% -> 99.8%+**: third channel keyed on (state/city, number token) exact match; word-level name TF-IDF channel; larger k and a name-token channel for empty-address queries; measure on big states | up to +0.002 dev, more on test | misses today: 60% empty-address, rest number edits and renames |
| 1.2 | **Address identifiers v2**: composite identifier tokens ("4-1-10/1/2", "E-4/39"), labelled numbers (plot/door/h.no/no/flat/ward/kh/gala/survey), French "bis/ter" and postal codes; features for added vs replaced vs edited sub-number, edit distance between identifiers, and "the S1 has no identifier" | large for India (half the test) | true records *add* labelled numbers ("Door No 773"), siblings *replace* sub-numbers |
| 1.3 | **Test-density training and decisions**: negative weight 1.9 (or the augmentation of 0.2) so probabilities are calibrated for test; replace the global threshold by a per-S1 expected-F0.5 decision (sort candidates by p, pick the prefix maximising expected F0.5 including the empty prediction); per-country shape calibration kept as a guard | +0.002-0.005 test-like | directly optimises the metric and the singleton risk |
| 1.4 | **Train on much more data**: 50-60% of S1 per country locally (48 GB), all states represented, seeds on disjoint states | v4 used 30% on Kaggle | memory-light feature frames (float32, per-partition) |
| 1.5 | **Empty-address records**: features for name uniqueness in the country (count of S1 with the same core name, best minus second-best name score, whether the S1's other records are confident), plus the 15x prior; abstain when ambiguous | recall on 0.16 records/S1 | biggest FN bucket after renames |
| 1.6 | **Random renames**: address-uniqueness features (how many S1 share the (number, street, city) key; whether the S1's confident records share the query's address form) so a unique exact address alone can carry a match | largest FN bucket | today p=0.08 for an identical-address rename |
| 1.7 | **Far-number identical names**: explicit feature for "identical name, different unrelated number, no other record agrees" so p=0.99 cases like Lantern Inc (360 vs 64) are rejected | precision, singletons | |

### Phase 2 - collective and transductive (expected: the France and India shift)
| # | item | expected gain | notes |
|---|---|---|---|
| 2.1 | **Stage-3 S1-level model**: aggregate an S1's candidates (best/second p, count of strong, agreement of numbers among strong candidates, sibling-group detection: records sharing a new number or modifier word), output a singleton probability and per-record adjustments; trained on OOF stage-2 outputs at 1.9x density | singletons and sibling groups; larger on test than on dev | GPT-6 items 1 and 4 |
| 2.2 | **Test-time self-training**: confident test pairs as pseudo-labels -> per-country lexicon and modifier vocabulary (France words, native-script business words) -> rescore; one or two rounds; proxies must stay in range | France | label-free, allowed |
| 2.3 | **Transliteration audit**: learn from all 7.6M pairs, keep the phonetic fallback, add a char-level fallback; report unmapped-token share on the full test | India recall | v2 mapped 144 tokens; check what is left |
| 2.4 | **France-specific checks**: département/region/postal-code state mapping, French legal forms and stopwords, manual review of 100 accepted and 100 rejected pairs | France is 15% of test with no labels | |

### Phase 3 - capacity (only if still below 0.99 test-like)
| # | item | notes |
|---|---|---|
| 3.1 | Seed and feature-bagging ensemble of the final LightGBM plus a CatBoost model (`src/ensemble.py` exists) | +0.001-0.002 typically |
| 3.2 | Multilingual cross-encoder (xlm-roberta-base or mdeberta-v3-base, both MIT, far below 8B) fine-tuned on pairs with hard negatives, used as a stage-2 feature; needs a remote GPU (A100 class, a few hours to train, about 1-2 h to score the 10M test pairs) | pretrained knowledge of French and Indic words attacks the exact weakness (unknown modifier words); rule check: pretrained weights are a model, not a lookup |

### Phase 4 - submission hygiene (every submission)
- `candidate_pairs.tsv` = exactly the stage-1 survivors scored by the final model (rule).
- Official validator with `--check-ids`.
- Per-country proxy table and the decision thresholds logged in RESULTS.md.
- No public-LB probing beyond sanity: the private split decides.
- Documentation template and the reproducible package updated with each structural change.

---

## 4. Order of execution and checkpoints
1. Phase 0 (0.1-0.3), then re-measure v4 on dev-v2 at 1.9x density: this is the new baseline.
2. 1.1 blocking and 1.2 identifiers (independent, can run in parallel), then 1.3 decisions and 1.4 data.
3. Full local run, submit **v5**, record proxies and LB.
4. 1.5-1.7 and 2.1, submit **v6**.
5. 2.2-2.4, submit **v7**.
6. Phase 3 if v7 test-like is below 0.99.

## 5. What I need from you
- v4 leaderboard scores (each seed and the ensemble) and the four `download_me_seed*.zip` files if they exist: the pair scores let me measure the test proxies without re-running.
- The challenge deadline and the number of submissions per day (sets how many of v5-v7 get a leaderboard reading).
- A remote GPU only when Phase 3.2 starts (A100 40 GB or better, a few hours).
- Optional: raise the WSL memory cap (`C:\Users\rrsri\.wslconfig`, `memory=40GB`) if I move the heavy stages there; I will ask before shutting WSL down for it.
