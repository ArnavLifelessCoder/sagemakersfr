# Business Entity Resolution — findings, status and handoff

Amazon ML Challenge 2026. Goal: beat the current leaderboard top of **0.976808** (macro F0.5)
and hold up on the **private** leaderboard.

Code: [`code/business_entity_resolution/`](code/business_entity_resolution/) (runnable pipeline,
laid out the way the final submission zip needs it). Exploration scripts: [`experiments/`](experiments/).
**The dataset is NOT in this repo** (1.1 GB challenge zip). Anyone running code needs
`student_resource/dataset/{train,test}/*.tsv` locally.

---

## 1. The task, in short

- 3 sources of business records: `entity_id, business_name, business_address, country`.
- Source 1 (S1) is deduplicated. For every S1 entity, output all matching S2/S3 records (0..many).
- Score: F0.5 computed **per S1 entity, then averaged** (singletons included: an empty
  prediction for a true singleton scores 1.0, and any prediction for it scores 0).
  Precision counts twice as much as recall.
- Output: `matching_results.tsv` (scored) and `candidate_pairs.tsv` (the exact set the final
  model scores; matches must be a subset of it). Both are tab-separated with one row per test S1.
- Rules: no external data or APIs, and any model must be MIT/Apache and ≤ 8B parameters.
  Test adds **France**, which has no training data.

## 2. Data facts (measured)

| | train | test |
|---|---|---|
| S1 | 2.21M (US 60%, India 40%) | 1.73M (US 38%, **France 15%**, India 47%) |
| S2 + S3 | 10.3M | 9.97M |

- **Each S2/S3 record matches at most one S1** (0 of 7.64M train matches are shared).
  The problem is many-to-one, so each S2/S3 record can be assigned to its single best S1.
- **26% of S2/S3 records match nothing.** These are distractors.
- **5.6% of S1 are singletons**, the same share in US and India. Clusters average ~3.5 matches,
  and 80% of S1 have matches in both S2 and S3.
- About 3% of S2/S3 have an **empty address**. About 5% of names are in **Indic script**
  (Devanagari, Gujarati, Kannada, Malayalam, Odia, Gurmukhi …).
- True pairs agree on state 95.7% of the time. 4.3% have an empty S2/S3 address, and only 0.07% disagree,
  so **partitioning blocking by (country, state) is safe**.
- State formats: US uses 2-letter codes (S1, S2) or full names (S3). India uses English, 2-letter codes
  (MH, DL, KA …) or native script (महाराष्ट्र, ಕರ್ನಾಟಕ …). France has 3 regions plus 4 départements
  (Nord, Pas-de-Calais → Hauts-de-France; Gironde → Nouvelle-Aquitaine; Loire-Atlantique → Pays de la Loire).
- Null placeholders inside addresses: `NULL`, `null`, `N/A`, `<NULL>`.

### Noise patterns in true matches
- Names: typos, accents (Sólutions), case, dropped words ("Cozy Ice"), duplicated words,
  shuffled words and legal suffixes ("LLC Hernandez …", "Pvt X Ltd."), junk wrappers
  (`--`, `>>`, `<<`, `#`, `([Limited])`, `(ID: 72277)`, `- 4941186041`, `| www.x.com`),
  domain names (`fudit.com`, `#aksarainfraestate`, `Hministries.Com`), **DBA / "formerly known as"**
  with a random new name in front ("Solcirax formerly known as Heartland Ministries"),
  and **completely random names** ("Nylaevogild") where only the address links.
- Legal typos: "Provate", "Priavlte", "lncorporated".
- Addresses: abbreviations (Rd/Road), component reordering, zero-padding (0014), `#`/`##`
  prefixes, **truncated house numbers** (3195→319, 5711→571), `1/2` suffixes, dropped units,
  and native-script state names.

### Distractors are deliberately hard negatives (the core difficulty)
Unmatched S2/S3 records are "siblings" of a real S1: same street, near-identical name. Vermont sample:

| house number vs S1 | true matches | distractors |
|---|---|---|
| identical | **79%** | 1% |
| zero-pad / truncated / missing | 15% | 2% |
| off by ≤ 20 | 1.7% | **58%** |

- Distractor names **add or substitute real vocabulary words**, for example eastgate, southside, downtown,
  midtown, riverside, highland, central, north, holdings, partners, group, India, global, public.
  France: Groupe, Développement, International, Holding, Distribution, Participations, "et Fils",
  "+ Associés"; substitutions such as Primaire→College, Centre→Lycée, Club→Ecole.
- **Legal-form changes are a distractor signal.** India: "Private Limited" → "Limited" (a different
  company type), whereas true matches drop "Limited" but keep "Private". US: LLC → Co, adding or
  removing LLC. France: SAS → SA/SASU/SCI.
- True-match extras are typos, `com`, `dba`, `formerly`, legal words, `center`, `services`.
- Irreducible cases exist, such as an identical name with a truncated number (530→53). Truncation also
  happens in true matches.

## 3. Pipeline (implemented, working end to end)

`python -m src.pipeline train …` then `python -m src.pipeline predict …` (see the code README).

1. **Normalise** (`normalize.py`): NFKC plus unidecode. Junk wrappers and IDs are stripped. DBA/formerly
   is split into main and alternative names. Domain names are unpacked. Legal forms are canonicalised,
   including fuzzy typos and French forms (SARL, SAS, SASU, EURL, SCI, SNC, EI, Cie).
   Honorifics and French articles are dropped. Address handling: canonical state (right-most
   component wins), house number plus suffix plus `1/2` flag, number set, canonical street tokens
   (street/st, road/rd, rue/r, boulevard/bd …, ordinals), a street-token guess and a city guess.
2. **Transliteration table** learned from training pairs (native token → Latin token, positional
   alignment when token counts match), with a unidecode fallback.
3. **Blocking** (`blocking.py`), run from the S2/S3 side per (country, state). Records with no
   state search their whole country. Two channels:
   name char-3-gram TF-IDF and address-token TF-IDF plus a `HS_<housenumber>_<street>` token.
   Exact multithreaded sparse top-K (`sparse_dot_topn`), union of both channels, top-5 per channel kept.
   - Dev recall: k=1 96.6%, k=3 98.9%, **k=5 99.3% (8.9 cand/query)**, k=10 99.55%, k=20 99.65%.
     Most remaining misses are unrecoverable (empty address plus mangled name).
4. **Stage-1 filter**: LightGBM on cheap vectorised features (rapidfuzz `cpdist` name/address
   scores, cosines, per-query and per-S1 rank/gap context). The threshold keeps 99.95% of true pairs.
   **It keeps 0.98 pairs per query (11%)**. These survivors are what goes in `candidate_pairs.tsv`.
5. **Stage-2 matcher**: LightGBM on everything. Python token logic covers fuzzy extra/missing tokens,
   legal conflicts, house-number relation category and log difference, number-set overlap and
   address token overlap. It adds the **token lexicon** (log-odds of a token appearing as an
   extra/missing word in true vs false pairs among name-similar pairs, including `L:<legal>` tokens,
   out-of-fold on training data), **vocabulary features** (label-free document frequency of the
   differing tokens among that country's S1 names, which carries over to France), stage-1 `p1`, and p1 rank/gap context.
6. **Assignment**: each S2/S3 record goes to its argmax S1 if p ≥ threshold (tuned on validation S1 entities).

## 4. Results so far (dev = regional slice of TRAIN: 9 states, 117k S1, 546k S2/S3)

| change | validation macro F0.5 |
|---|---|
| baseline single-stage model | 0.9882 |
| + fuzzy legal typos + `L:<legal>` lexicon tokens | 0.9898 |
| + vocabulary features, French stopwords | 0.9901 |
| **two-stage (stage-1 filter + p1 context)** | **0.9907** (threshold 0.70) |

Top features: `p1_q_gap` (margin over the query's next-best S1), `p1`, lexicon features,
`legal_q_only`, `hn_logdiff`, `hn_rel`. The most distractor-like lexicon token is `L:pvt`
(losing "Private").

Caveats: the dev slice is small and regional, and France is untested. **No leaderboard submission yet.**
The official validator passes on a dev-sized run.

Remaining error types (dev validation): FPs are mostly distractors with a legal change plus a truncated
or nearby number, or an identical name with an empty address. FNs are mostly empty addresses with a weakened
name ("Swati Limited Center"), random renamed records whose address is also noisy, and native-script
names with sparse addresses.

## 5. Running it on Kaggle
- Upload the challenge zip and this code folder as two private Datasets. Run
  `code/business_entity_resolution/kaggle_run.ipynb`. Internet must be ON (pip installs rapidfuzz,
  sparse_dot_topn and Unidecode only).
- Prefer a **TPU VM** session for its CPU cores; nothing uses a GPU. On a 4-core CPU, expect about 5–7 h in total.
- `train_frac=0.25` of training S1 (whole states) is used, with 20% of it held out for validation.

---

## 6. HANDOFF — next tasks for the cloud session (priority order)

> The data is not in the repo. If the session has no data, work on code and unit-testable logic only,
> and never commit the dataset.

1. **France self-training lexicon** (highest expected value: 15% of test, no labels).
   In `cmd_predict`, for France: score once, take confident pairs (p2 > 0.95 as positive, and
   p2 < 0.05 among name-similar pairs with `n_tset ≥ 80` as negative), `learn_lexicon` on those,
   blend it with the trained lexicon for French tokens only, recompute `add_lex`, and rescore.
   Log how many French matches change.
2. **Cluster-consistency (stage-3) features.** After stage 2, for each (q, s): the number of other
   queries assigned to s with p > 0.5, the share of those sharing q's house number and s's house number,
   max name similarity between q and those records, and whether q's source (S2/S3) already has
   a strong match to s. Train on out-of-fold stage-2 predictions.
3. **More training data plus robustness.** Try `train_frac` 0.4–0.6 if memory allows. Tune the
   threshold per country (US vs India). For France, fall back to the global threshold,
   or a slightly higher one to protect precision.
4. **Speed.** `prep.parse_frame` (about 130 µs per record) is the main single-process cost left.
   Profile and vectorise the hot regexes. Stage-1 OOF training could subsample negatives.
5. **Documentation.** Fill `Documentation_template.md` (from the challenge zip) using sections 2–4 above.
6. Ideas not tried yet: a small multilingual char/embedding model (MIT/Apache, ≤ 8B) as an
   extra name-similarity feature for native-script names; a per-S1 "expected cluster size"
   prior; learned address aliases (Bombay/Mumbai, etc.) from training co-occurrence.

Local dev loop (needs the zip):
```bash
cd code/business_entity_resolution
python -m src.make_dev_subset --zip PATH/6ab10eb3b23ba_student_resource.zip --out ../../dev
python -m src.pipeline train --dev ../../dev --work ../../art_dev      # ~11 min on 12 threads
```
