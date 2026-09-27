# Sibling-aware entity resolution: the full approach

Amazon ML Challenge 2026, business entity resolution. This is the complete description of what the
submitted system does, why each piece exists, what was measured, and what did not work. Numbers are
from this repository's runs (RESULTS.md, FINDINGS.md, `experiments/`). Final public leaderboard
files: v5 0.975 (density-corrected decision), three-model ensemble submitted last (see RESULTS.md).

---

## 1. Task, metric and what the metric implies

Three sources of business records (`entity_id, business_name, business_address, country`). Source 1 (S1)
is deduplicated; for every S1 entity, output the set of Source-2/3 records that describe the same
business. Every S2/S3 record belongs to at most one S1 (0 of 7.64M training matches are shared), so the
problem is a many-to-one assignment, not general clustering.

Score: F0.5 per S1 entity, macro-averaged over all S1 including singletons (an empty prediction for a
true singleton scores 1, any prediction for it scores 0). With P = precision and R = recall of one entity,
F0.5 = 1.25·P·R / (0.25·P + R) = 1.25·tp / (m + 0.25·|T|) for m predicted and |T| true records.

Cost arithmetic that drives every decision below (entity with 3 true records):

| error | entity score | loss |
|---|---|---|
| one wrong record attached | 0.789 | 0.21 |
| one true record missed | 0.909 | 0.09 |
| any record attached to a true singleton | 0 | 1.00 |

Reaching 0.99 means an average loss of 0.01 per entity: roughly one false merge per 20 entities or one
miss per 11. Precision work and recall work are worth about the same per record, but the first record
attached to an entity carries singleton risk and later ones do not.

Rules that matter: no external data or lookups; any model MIT/Apache and under 8B parameters (pretrained
weights are a model, not a lookup); `candidate_pairs.tsv` must be exactly the set the final model scored;
the public leaderboard is a subset of the test entities, the private split decides.

## 2. What the data is (measured)

Training: S1 2,206,821 (US 60%, India 40%), S2 5,034,616, S3 5,285,603. Test: S1 1,732,544 (India 47%,
US 38%, France 15%), S2 4,887,273, S3 5,082,316. France has no training data.

The generator, identical in the US and India:

- 3.46 true records per S1 (S2 1.67, S3 1.79); singletons 5.58%; cluster sizes 0/1/2/3/4/5/6+ =
  5.6/5.4/17.0/24.1/21.9/14.6/11.4%. Singleton status is a separate draw (P(n2=0, n3=0) = 5.6% against
  1.6% if independent); given a non-singleton, the S2 and S3 counts are nearly independent.
- 26% of S2/S3 records are distractors: 1.21 per S1. They are *siblings* of a real S1 (same street,
  near-identical name), made by adding or swapping a real vocabulary word (holdings, exports, eastgate,
  groupe, développement, bakery, jewellers, also in native script), changing the legal form (Private
  Limited → Limited, SAS → SASU) and editing one sub-number of the house identifier (25/1923 → 25/1932,
  506 → 527, 17/34 → 17/39). Siblings usually have several records that agree with each other.
- True-record noise is different in kind: typos, accents, case, dropped/shuffled words and legal words,
  wrapper junk (`--`, `(ID: 72277)`, `| www.x.com`), domain names, DBA / "formerly known as" prefixes,
  completely random new names at the same address, native-script transliterations, address
  abbreviations and reordering, zero-padding, truncated numbers, an added labelled number ("Hn 402"),
  or an empty address.
- Empty address: 4.5% of true records, 0.3% of distractors (a 15x prior). Native-script names: 23.5% of
  India S2, 13.2% of India S3.
- Indian house identifiers are composite ("4-1-10/1/2", "Plot No-176", "H.no 332", "KH NO. -570/13");
  a leading-integer house number agrees on only 17% of true India pairs (83% in the US).

### 2.1 The train → test shift, decomposed

Test holds 5.5 to 5.8 S2/S3 records per S1 against 4.68 in training. For matching states, every S2/S3
record was matched to its nearest S1 with the blocking's own TF-IDF channels and binned by best-name
cosine and house-number agreement (`experiments/eda_shift.py`). Test bins are a mixture of the labelled
training bins with weight 0.59 on true records in both the US and India (every bin fits): test = 3.3 true
records per S1 (the training prior) + **2.3 distractors per S1, 1.9x the training density**, from the
same distractor distribution. Orphans (records with no similar S1) are 0.05 to 0.3%. The empty-address
rate in test S2/S3 (2.3 to 3.1%) dilutes exactly as this predicts.

Consequences: probabilities learned at 1x density over-accept on test; sibling groups are larger on
test, so collective evidence gets stronger; and France, which contributes no labels, needs label-free
signals.

### 2.2 France specifics

35% of France S2/S3 addresses end with the city and no region ("17 RUE YVES BODIGUEL, NANTES"); the US
and India are 100% state-recognised. Without a region a record is searched country-wide and its
state-agreement feature is unknown. A (country, city) → region map learned from the S1 addresses only
(26 entries, majority vote, purity ≥ 0.9) backfills 455,385 of those records at test time. French
siblings use Groupe, Développement, Participations, Holding, "& Associés", "et Fils", legal-form swaps
(SAS/SASU/SA/SCI/SNC/EI) and word swaps between common category words (Comité → Club, Amis → Gestion,
Primaire → Collège).

### 2.3 Scale effects the small dev slice hid

The inherited 9-state dev slice gives blocking recall 0.994 and F0.5 0.992. A stratified 30% slice with
Delhi, Uttar Pradesh, Texas and Ohio gives, for the same v4 code, blocking recall **0.981** and validation
F0.5 **0.986**. On that slice the loss is 52% blocking misses (Delhi alone 44% of them; 46% of all misses
are empty-address records searched name-only against 400k S1), 19% true records below threshold, 16%
records attached to the wrong same-name or same-address S1, 8% distractor false merges, 4% singletons
given a match. Everything after v4 was judged on this slice (135,052 held-out entities).

## 3. Pipeline

`train`: parse → transliteration table → stratified training slice → blocking → stage-1 filter (OOF) →
stage-2 features → out-of-fold lexicon → LightGBM → decision rule and calibration target.
`predict`: per country: parse → city→region backfill → blocking → stage-1 → stage-2 → checkpoint;
then assignment. Code: `v5/business_entity_resolution/src` (the `v6` copy adds frequency features,
neighbour-state search and the transductive lexicon).

### 3.1 Normalisation (`normalize.py`, `prep.py`)

Names: NFKC + unidecode, lower case; `M/S` and ID/reference numbers removed; 6+ digit numbers removed;
dotted initials joined; `&`/`+` → "and"; domain names unpacked ("fudit.com" → "fudit"); DBA / "formerly
known as" / "a.k.a." split into a main and an alternative name; legal forms canonicalised (llc, inc, corp,
co, ltd, pvt, lp, llp, pllc, pc, plc, pa, sarl, sas, sasu, sa, eurl, sci, snc, scop, gie, opc, gmbh, ei,
cie), including misspelled long forms by edit distance ("provate", "lncorporated"); honorifics and French
articles dropped; leet digits inside words (0→o, 1→l, 3→e, 5→s, 4→a, 7→t, 6→g, 8→b, 9→g, 2→z). Native-script
tokens are transliterated with a table learned from matched training pairs (positional alignment when
token counts agree, 1,498 entries on the full data) plus a phonetic fallback for unknown tokens:
consonant skeleton of the romanisation matched to the most frequent Latin name token of the same data
(144 test-only business words such as ज्वेलर्स → jewellers).

Addresses: components split on commas; null placeholders dropped; the right-most component that is a
state name (English, code, or native script for India; region or département for France) is the
canonical state; house number = leading integer of the first numbered component with its letter suffix
and a "1/2" flag; number set (all digit runs, zero-stripped); canonical street tokens (abbreviations,
ordinals, drop-words); a street guess and a city guess (last digit-free component, with common Indian
city aliases); **composite identifiers** (`ident_tokens`): tokens such as "4-1-10/1/2", "e-4/39", "b-201",
"1003a", "63/982" with separators removed and label prefixes (no, hno, plot, door, flat, shop, sco, kh, tc)
stripped, ordinals skipped.

### 3.2 Blocking (`blocking.py`)

Run from the S2/S3 side inside each (country, canonical state) partition; the S1 partition of a state also
searches the queries of its neighbour states (Telangana ↔ Andhra Pradesh, DC ↔ Washington, learned from
the training ground truth by the teammate: 18.5% of Telangana pairs are written "Andhra Pradesh").
Records without a recognised state are searched country-wide. Three TF-IDF channels with exact sparse
top-k retrieval (`sparse_dot_topn`):

1. character 3-grams of the core name (sublinear tf, max_df 0.05);
2. canonical address tokens plus a composite `HS_<housenumber>_<street>` token and `ID_<identifier>`
   tokens for composite identifiers;
3. the two vectors side by side (cosine = mean of the two), which finds records moderately noisy on both
   fields that fall out of both single-field lists.

Top-20 per channel, then per query the union of the top-10 by name cosine, by address cosine and by their
sum; 40 by name for records without a state; S1 records without a state are reachable from every query of
the country.

| setting | dev recall (9 states) | candidates / record |
|---|---|---|
| inherited: top-10, keep 5 | 0.99427 | 8.9 |
| top-20, keep 10, no-state 40 | 0.99677 | 18.5 |
| + combined channel (two states) | 0.99638 → 0.99690 | 19.6 |
| (state, number) exact-key channel | +0.001 | +35 (rejected) |

At scale (30% slice): v4 0.98106 → v5 0.99215 (18,406 instead of 44,435 true pairs missed).

### 3.3 Stage-1 filter

A small LightGBM (63 leaves, learning rate 0.1, 300 rounds) on cheap vectorised features only: the two
channel cosines, rapidfuzz ratio / token-sort / token-set / partial / Jaro-Winkler on core names and on
concatenated names, address token-set and token-sort similarity, street and city ratios, native / domain
/ empty-address / source flags, house-number equality, and per-query and per-S1 rank, gap and count
context of a blocking score. Trained two-fold out-of-fold by S1-id hash so its probability `p1` can be a
stage-2 feature. Threshold chosen to keep 99.95% of the blocked true pairs: keeps 1.06 pairs per record
(5% of candidates). These survivors are exactly `candidate_pairs.tsv`.

### 3.4 Stage-2 features (about 110)

- Name: exact and fuzzy (Jaro-Winkler ≥ 0.88 or ratio ≥ 75) extra / missing token counts, token jaccard,
  first-token equality, exact and sorted equality, subset relations, token counts, alternative-name
  similarity, short-token extras.
- Legal forms: conflict, equality, one-sided counts; the differing forms also enter the lexicon as
  `L:<form>` tokens (losing "Private" is the single most sibling-like token).
- Address: house-number relation (both missing / one missing / equal / equal with suffix / truncated /
  small / mid / large difference) and log difference, number-set jaccard and one-sided counts, query
  number found in the S1 number set, min log distance, address-token jaccard and one-sided counts,
  state agreement (with state groups), "1/2" flag, suffix difference; **composite identifiers**: jaccard,
  one-sided counts, min edit distance between uncovered identifiers, set equality, `ident_replaced` (S1
  identifiers the record does not carry while it carries others) and `ident_added` (record identifiers
  beyond a fully covered S1: the "Hn 402" pattern of true records).
- Lexicon (labelled): smoothed log-odds of each extra / missing token appearing in true vs false
  near-duplicate pairs (token-set ratio ≥ 80), learned two-fold out-of-fold on the training rows with
  40% of the distractor entries randomly hidden per fold so the model also learns from label-free
  evidence (max / min / sum for extras and for missing tokens).
- Vocabulary and modifier features (label-free, computed per country on the data being scored): log
  document frequency of the differing tokens among the country's S1 names, counts of real-vocabulary vs
  rare differing tokens, the substitution flag, and the S2/S3-vs-S1 frequency ratio of the differing
  tokens (distractor generators add words that are far more frequent in S2/S3 names than in S1 names).
  This is what transfers to France and to native-script business words.
- Stage-1 context: `p1`, per-query rank / gap / count and per-S1 rank / count / gap-to-top of `p1`.
  `p1_q_gap` (margin over the record's next-best S1) is the top feature in every run.
- Anchor features: comparison with the S1's best *other* candidate (its `p1`, whether it is strong, house
  number agreement with the record and with the S1, name similarity, number-set jaccard), counts of strong
  peers and how many of them agree on the record's or the S1's house number, whether the record is the
  anchor itself.
- Peer features (collective): for the record's extra name tokens and its new numbers, how many other
  candidates of the same S1 carry them (counts and `p1`-weighted), how many carry both, and the group size.
  Siblings repeat their modifier word and their new number across their records; per-record noise does
  not. `peer_num_share_p` is the #4 feature.
- Frequency features (v6): how many S1 of the country share the S1's or the record's exact core name
  (rate and log count) or address key, and how many of the record's candidates carry exactly its name
  (cross-entity ambiguity is 16% of the full-scale loss).

### 3.5 Matcher

LightGBM binary, 127 leaves, learning rate 0.05, feature fraction 0.8, bagging 0.8, L2 1.0, early stopping
(100 rounds) on the held-out entities, best iteration ~1,000. Training data: a stratified slice of about
31% of the S1 entities per country in whole states (states larger than half a country's target are skipped
so several states are represented; seed 0: India 7 states / 272,504 S1 and US 15 states / 405,400 S1),
with all their true records, the unmatched records of the same states and a proportional sample of
empty-address unmatched records; 20% of entities held out by hash for early stopping, threshold selection
and the calibration target. Negative up-weighting (1.9) was tried and gave nothing on top of the decision
rule below.

Validation on the 135,052 held-out entities of the 30% slice: v4 0.98584 (test-like) → v5 0.99060 plain,
0.99001 with false merges weighted 1.9x.

### 3.6 Cross-encoder (`cross_encoder.py`, `export_pairs.py`, `blend_ce.py`)

`xlm-roberta-base` (MIT, 278M parameters) fine-tuned for one epoch on 1.5M stage-1 pairs of the training
slice (text = "name | address | country" of both records, negatives capped at three per positive with the
hardest kept, max length 128, batch 64, lr 2e-5, fp16, 33 minutes on one L40S). Held-out pairs: logloss
0.0437, accuracy 0.9842. Blended with the LightGBM probability in logit space:

| weight on the cross-encoder | validation F0.5 | test-like |
|---|---|---|
| 0 | 0.99060 | 0.99001 |
| 0.2 | 0.99174 | 0.99127 |
| 0.35 | 0.99180 | 0.99124 |
| 0.5 | 0.99117 | 0.99035 |
| 1.0 | 0.98666 | 0.98409 |

Weight 0.3 is used. Scoring the 12.2M test pairs took 49 minutes. A one-epoch `xlm-roberta-large` gave
no gain (0.99173 at its best weight) and was dropped.

### 3.7 Decision rule (`model.assign_expected`, `assemble.py`)

Each record goes to its most probable S1. Then, per S1, its records are sorted by probability and the
prefix with the highest expected F0.5 is kept, the empty prediction included, under independent truths:
E[F] = Σ P(tp) P(fn) · 1.25·tp / (m + 0.25·(tp + fn)) with tp ~ Poisson-binomial over the kept records,
fn over the dropped ones plus one hidden true record with probability 0.02 (blocking misses); m = 0 scores
Π(1 − p). Entities whose records are all ≥ 0.985 are kept whole without the search. Per country, a shape
guard shrinks the odds (bisection on a divisor) only if predicted matches per S1 exceed the validation
rate (3.378) by more than 2%.

Evidence on the decision, in order:

| decision | where | result |
|---|---|---|
| fixed global threshold | validation | 0.99172 (test-like) |
| expected-F0.5, raw probabilities | validation | 0.99174 and best plain F0.5 |
| expected-F0.5 with odds / 1.9 | validation (fp x1.9) | worse than raw |
| expected-F0.5 with odds / 1.9 (v5) | public LB | **0.975** (v4 0.971) |
| band calibration + blend | public LB | 0.973 |
| teammate: blanket thresholds FR 0.99 / IN 0.9 / US 0.96 | public LB | 0.974, their raw model 0.978 |

Band calibration (rescaling each probability by how much its band is inflated on the test relative to
the validation slice: France 4 to 6.6x, US 2.6 to 4.3x, India 1.5 to 2.4x in the mid bands) is a
principled label-free correction and it lost 0.002 on the leaderboard: the mid bands hold more true
records than the inflation implies (French word-swap matches, renames, initials). Both leaderboard
readings point the same way, so the final files use the raw decision with only the shape guard.

### 3.8 Ensembling (`ens_ckpt.py`)

Predictions checkpoint one parquet per country. Runs are averaged per pair over the runs that scored the
pair (a run's blocking difference is mostly extra recall, so no penalty for single-run pairs), then
blended with the cross-encoder and decided as above. Final: our v5 seed 0, our v6 seed 1 (neighbour-state
search, frequency features) and the teammate's seed 202 (0.978 alone): union of 13.26M candidate pairs,
France 3.281 / India 3.358 / US 3.395 matches per S1, singletons 5.6 to 5.8%.

### 3.9 Transductive France lexicon (`predict --selftrain-lex`)

After a first pass, a per-country lexicon is learned from the run's own confident decisions (p ≥ 0.97
vs p ≤ 0.03) with the training lexicon kept for known words, and the country is rescored. On France it
learns exactly the sibling vocabulary (holding, participations, développement, groupe, parents, fêtes,
foyer, culture, loisirs, SNC, EI) and moves 2.4% of pairs across 0.5; it makes France more selective
(3.30 → 3.22 matches per S1). Kept as an option; not in the final file because every leaderboard reading
favoured the less selective decision.

### 3.10 Engineering that mattered

- `peer_features` had a per-entity loop quadratic in the candidate count; on test, popular names give
  entities thousands of candidates and both predictions sat for hours in it. Groups above 300 candidates
  now use a (token, number)-combination counter.
- pandas 3 stores strings as Arrow arrays and rapidfuzz iterates them element by element; feeding it
  object arrays made the cheap-feature stage several times faster.
- Per-country checkpoints (`scores_<country>.parquet`) let a run resume, let countries run in parallel
  processes (`--countries`), and make every decision variant a few minutes of assembly.
- Timings on an 18-core laptop: training on the 30% slice 35 minutes (12 workers), prediction of the full
  test 60 to 90 minutes, cross-encoder training 33 minutes and test scoring 49 minutes on an L40S.

## 4. Results

| version | change | validation | public LB |
|---|---|---|---|
| v3 (inherited) | US-only training, single threshold | 0.9887 (US slice) | 0.947 |
| v4 | stratified India + US training, peer features | 0.98584 (30% slice) | 0.971 |
| v5 | blocking x2 + combined channel, composite identifiers, France backfill, expected-F0.5 | 0.99060 / 0.99001 | 0.975 (odds / 1.9) |
| v5 + cross-encoder + band calibration | | 0.99180 (blend) | 0.973 |
| teammate seed 202 raw | wider blocking, OCR fold | | 0.978 |
| final | 3-model ensemble + cross-encoder blend + raw decision | | submitted last |

## 5. What did not work

- Negative weighting 1.9 in training: nothing on top of the decision rule.
- (state, number) exact-key blocking channel: +0.001 recall for 35 extra candidates per record.
- Structural post-rules before the decision (drop mid-confidence records with a replaced identifier, a
  sibling group, an added legal form or a modifier word): neutral or worse; the trees already carry it.
- `xlm-roberta-large`: no gain over the base model.
- Any decision more conservative than the model's own calibration: lost on the leaderboard twice.

## 6. What would move it to 0.99

1. Train with synthetic test-style siblings made from real true records (legal change + number ±20;
   word swap), so the model learns the test odds at the source (teammate's v8).
2. Feed the cross-encoder probability to the trees as a feature (out-of-fold on training pairs) instead
   of a late blend, and train on the full data rather than a 30% slice.
3. A stage-3 entity-level model on out-of-fold stage-2 outputs: singleton probability, sibling-group
   detection, agreement among strong candidates.
4. Empty-address records: name-uniqueness features and an abstain rule; they are 46% of blocking misses
   at scale and a source of cross-entity confusion.
5. Neighbour-state search in every run and a transliteration audit on the full test.

## 7. Reproduction

```bash
cd v5/business_entity_resolution
python -m src.pipeline train   --data-dir DATASET_DIR --work art --config cfg.json --workers 12 --threads 12
python -m src.pipeline predict --data-dir DATASET_DIR --work art --out out --workers 12 --threads 12
# cross-encoder (GPU box): export pairs, train, score, blend, assemble
python -m src.export_pairs train --dump DUMP --data-dir DATASET_DIR --out pairs_train.parquet
python cross_encoder.py train --pairs pairs_train.parquet --out ce_model
python cross_encoder.py score --pairs pairs_test.parquet --model-dir ce_model --out ce_test.parquet
python -m src.blend_ce test --scores-dir out --ce ce_test.parquet --w 0.3 --out out_blend
python -m src.ens_ckpt --runs out_blend_run0 out_blend_run1 --out out_ens
python -m src.assemble --scores-dir out_ens --test-dir DATASET_DIR/test --out final --meta art/meta.json --decision expected
python utils/validate_submission.py --matching final/matching_results.tsv --candidate final/candidate_pairs.tsv --test-dir DATASET_DIR/test --check-ids
```
