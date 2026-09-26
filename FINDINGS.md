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

Caveats: the dev slice is small and regional, and France is not covered. **The first leaderboard
submission scored 0.911 (see §5): dev validation did NOT reflect the test distribution.**

Remaining error types (dev validation): FPs are mostly distractors with a legal change plus a truncated
or nearby number, or an identical name with an empty address. FNs are mostly empty addresses with a weakened
name ("Swati Limited Center"), random renamed records whose address is also noisy, and native-script
names with sparse addresses.

## 5. Leaderboard reality check: the train → test shift (EDA after v1 = 0.911)

v1 scored **0.911** on the public LB against 0.9907 on dev validation. The leader was at 0.9859.
Scripts are in `experiments/diag/`.

**5.1 The generator is identical across countries, but test has more distractors.**
- Training truth has *exactly* the same cluster shape in US and India: mean 1.67 S2 + 1.79 S3 matches per S1,
  the same (nS2, nS3) mix, and 5.6% singletons.
- Records per S1: train about 4.68 (S2 2.28 + S3 2.40), **test about 5.5–5.8 (S2 2.7–2.9 + S3 2.8–3.0) in all
  three countries**, i.e. about 1.1 extra records per S1.
- Distractor "marker" words are about **1.7× more frequent in test S2/S3** than in train S2/S3, at identical
  S1 rates. US: eastgate/southside 51 vs 30 per 10k names, holdings 344 vs 235. India: bakery/pharmacy/hardware
  about 12 vs 7.5 per 10k. So test holds about 1.7× more sibling distractors per S1, and probabilities
  calibrated on training odds over-accept on test.

**5.2 v1 predictions vs the training cluster-size prior (label-free diagnostic).**

| | mean S2 matches / S1 | mean S3 matches / S1 | singletons |
|---|---|---|---|
| train truth (US = India) | 1.672 | 1.786 | 5.6% |
| test v1 US | 1.632 | 1.743 | 5.81% |
| test v1 France | 1.661 | 1.756 | 5.45% |
| **test v1 India** | **1.908** | **1.910** | **3.50%** |

US and France sit at about 97.5% of the prior, which is normal recall loss. **India over-matches by about 0.35 per
S1 (about 280k false merges)** and under-predicts singletons, so every singleton that received a sibling scores 0.

**5.3 Root cause for India: test-only native-script distractor words.**
- The learned transliteration table covers **100%** of native tokens in held-out *train* but only **96.4%** in
  *test*. The uncovered tokens are not random names. They are a fixed list of about 20 **business-type words**,
  each about 3,600 times, evenly: मोटर्स motors, स्टोर्स stores, एजेंसीज agencies, ज्वेलर्स jewellers, जनरल general,
  बेकरी bakery, प्रोविजन provision, स्वीट्स sweets, फार्मेसी pharmacy, मेडिकल्स medicals, ऑटोमोबाइल्स automobiles,
  ट्रेडर्स traders, हार्डवेयर hardware, गारमेंट्स garments, स्टील steel, इलेक्ट्रॉनिक्स electronics, फर्नीचर furniture,
  रेस्टोरेंट restaurant, टेक्सटाइल्स textiles (plus the same words in Kannada, Telugu, …).
- In Latin script these words are known distractor modifiers (train S2/S3 rate about 20× the S1 rate). In native
  script they never occur in training, so v1 romanised them into gibberish that looked like a harmless typo.
  "जय माँ टेक्नोलॉजी ज्वेलर्स लिमिटेड" was accepted as "Jay Maa Technology Limited".
- Sampled India false merges follow the sibling pattern: name plus an added or swapped word (Steel, Technologies,
  Foundation, Overseas, Hardware) and a **replaced house identifier** (E-4/39 → E-4/60, C1-39/23 → 39/30,
  128B → 133B/5). Siblings often come in pairs that share their own number.

**5.4 France** uses the same generator with French vocabulary. Test-S2/S3-only modifier words are international,
développement, groupe, holding, participations, distribution, associés and SNC (240–375 per 10k in S2/S3 vs 1–80
in S1). There are word substitutions too (Comite → Primaire, Maison → Federation, Soins → Ecole). v1 France's
match rate looks right, but the spot-check showed substitution false merges in 3 of 12 clusters.

**5.5 Things checked and rejected**
- A hard "replaced number" rule flags 4% of true matches (noise also rewrites digits, e.g. 37931 → 37932). Not usable.
- A "two records replace the S1 number with the same new number" rule flags 2.3% of true matches, because the
  source formatting applies the same rewrite to several records. Not usable alone.
- A plain word-substitution rule: 5.6% of true matches also contain one (generic words). Not usable alone.

**5.6 v2 (test-time adaptation, v1 model reused, predict-only)**
- `prep.extend_translit`: unknown native tokens are romanised and matched by *consonant skeleton* to the most
  frequent Latin name token of the same data. All 7 checked test words map correctly (ज्वेलर्स → jewellers,
  ಬೇಕರಿ → bakery, इलेक्ट्रॉनिक्स → electronics …).
- `pipeline.calibrate_thresholds`: if a country's predicted matches per S1 exceed the validation rate by more than
  2%, its threshold is raised (never lowered) until it meets the rate. Expected to act on India only.
- Predict also writes `pair_scores.parquet` so thresholds can be retuned offline.

**5.7 v2 run on Kaggle (v1 model + test-time adaptation)**
- The phonetic fallback mapped **144** unknown native tokens in India (0 in US and France, as expected).
- The Kaggle v1 model's own validation F0.5 was **0.9883** (larger and harder states than the dev slice).
- Calibration: India 3.793 matches/S1 at the base threshold 0.75 → threshold **0.9707** (3.375). US (3.376)
  and France (3.422) were left unchanged. Singletons are now 5.45% / 5.47% / 5.81% (FR / IN / US) vs the 5.6% prior.
- Inspection of India pairs by probability band (`experiments/diag/v2_inspect.py`):
  - **removed** (0.75–0.97, 338k pairs): almost all sibling fakes (Brine Traders → "…Industries",
    Harvest Brothers → "…Exports" with flat 508→512, Channel Collective → Channel Chemicals 32/8→32/21,
    Vijay Care → "…स्टोर्स" (stores), Private → Public). So the cut was right.
  - **kept** at 0.97–0.99: still about half siblings (Chennai Cosmetics → Chennai Producer 10/140→10/144,
    Kamdhenu Plasto → Gardens 70/144→70/165, Pink Farms → Landmarks 267/2→267/9, Balaji Management → एस्टेट).
    Threshold calibration alone cannot separate them.
- The sibling pattern is **one content word swapped for another real word + a changed sub-number** of the house identifier.

**5.8 Why v3 fixes it (evidence)**
- On training data (dev), name-similar pairs with a *content-word substitution + changed number* are
  **99.7% false** (India 243,860 pairs, true share 0.003; US 0.002). True noise swaps in *generic* words
  (services, center, partners) or spelling variants (laxmi/lakshmi), and rarely changes the number at the same time.
- A real **test slice** (Tamil Nadu + Rajasthan, 87,590 S1, 498k S2/S3; `mk_test_slice.py`, `tslice_look.py`)
  was scored with the v2 model and with the v3 model trained only on the dev slice:

| S2/S3 record (vs its S1) | v2 p | v3 p | truth |
|---|---|---|---|
| Chennai **Producer** Pvt Ltd, 10/144 (S1: Chennai Cosmetics, 10/140) | 0.989 | 0.000 | sibling |
| Chennai **Sai** Pvt Ltd, 10/144 | 0.991 | 0.000 | sibling |
| Chennai Cosmetics **Exports**, 10/144 | 0.895 | 0.000 | sibling |
| Channel **Chemicals**, 32/21 (S1: Channel Collective, 32/8) | 0.900 | 0.000 | sibling |
| Kamdhenu **Gardens**, 70/165 (S1: Kamdhenu Plasto, 70/144) | 0.983 | 0.002 | sibling |
| Cresco Technologies **Infratech**, Shop No. 3 | 0.962 | 0.000 | sibling |
| "Mirawex" (random rename, same address) | 0.988 | 0.999 | true |
| `TECHPRIVATECRESCO.COM` | 0.006 | 0.988 | true |

  The **anchor features** drive this: the S1's real records agree on the house identifier, while a sibling
  disagrees with them and with the S1.
- Caveat: the dev-trained v3 predicts 2.87 matches per S1 on that slice (below about 3.4). Part of that is its small
  transliteration table (607 vs about 1,500 tokens on full training), which costs recall on Tamil and Hindi
  names. The full Kaggle v3 run is the real measurement.
- Also fixed in v3: leet digits 6→g, 8→b, 9→g, 2→z (`6lobal`, `8lue`, `8rothers` are common in the data).

**5.9 France: v2 vs v3 on the same records** (Pays de la Loire test slice: 72,734 S1; `fslice_cmp.py`)
- Both models accept 153,399 pairs, v2 alone 13,293, v3 (dev-trained) alone 1,997.
- **v2-only accepts are almost all French siblings** (about 18% of S1 in the slice): added modifiers such as
  "HQ Amicale Participations" (2→23), "Fabrique (France) Centre Participations" (3→24), "Croix Club
  Développement" (11→12), "Association de Soutien Groupe", "Sce SAS & Associés", "Dans SARL Et Fils";
  word swaps such as Foyer→Comite, Comite→Collège (15→16), College→Anciens, Club→Fetes (4→11).
- **v3-only accepts are mostly true matches v2 missed**: random renames at the exact address ("Tavocalo One",
  "Zetanylatavo", "ORBIXYLO", "Vantagewex") and initials ("DC", "AS", "PD"). A few are residual siblings
  (Medico→Compagnie 16→19, SASU→SNC 1→10).
- So France was also losing heavily in v1/v2, and v3 should recover most of it.

**5.10 Leaderboard so far and the training-slice bug**

| version | public LB | main change |
|---|---|---|
| v1 | 0.911 | baseline two-stage LightGBM |
| v2 | 0.934 | phonetic transliteration fallback + per-country threshold calibration (predict-only) |
| v3 | 0.947 | anchor / cluster features, label-free modifier words, lexicon dropout, leet fix |

- The v3 Kaggle log showed the **training slice was 18 US states + 1 tiny India state (716 queries)**:
  `select_training_slice` drew whole states from the pooled list until 25% of S1 was reached, and
  India's states are huge. **v1-v3 were trained essentially without India** (47% of test). Only the
  transliteration table was learned from all pairs, which hid the problem.
- v2 vs v3 final matches (`v2v3.py`): France both 840k, v2-only 47.6k (mostly siblings v3 removed), v3-only 6.3k.
  India both 2.60M, v2-only 138k (siblings with "Exports/Infratech/Overseas" + changed sub-numbers), v3-only
  130k, **of which many are Indian legal-type siblings** (LLP→Limited, Private→Public with a new number),
  a pattern the US-only model never saw. US: v3 adds some next-door siblings (2831→2832, 5501→5506).
- The field: about 200 teams above 0.98, top 50 above 0.99 (leader 0.9859 on day 1, higher later).

**5.11 v4 design (collective ER + proper training data)**
- **Stratified training slice**: states sampled per country, about `train_frac` of each country's S1; states
  larger than half a country's target are skipped so several states are represented.
- **Collective peer features** (`features.peer_features`): a sibling is a hidden business with its own
  records, so its modifier word and its new house number repeat across several of them, while per-record noise
  (typos, "services", zero-padding) does not. For each (q, s), count the other candidates of s sharing q's
  extra token / new number (plus p1-weighted versions and "shares both"). On dev, `peer_num_share_p` is the
  #4 feature.
- **Test-like threshold tuning**: `fbeta_macro(fp_weight=1.7)` counts each false merge 1.7x (test holds
  about 1.7x more distractors per S1), and the threshold is chosen on that score.
- **Parallel seeds + ensemble**: four Kaggle runs (seeds 0/1/2 at `train_frac` 0.3, seed 3 at 0.5) on
  different random states; `src/ensemble.py` averages pair probabilities (weights allowed), one
  country at a time on hashed ids (about 3 min for the full test on a laptop), then per-country assignment and
  calibration.
- Dev: F0.5 0.9919 (test-like 0.9911).

## 6. Running it on Kaggle
- Upload the challenge zip and this code folder as two private Datasets. Run
  `code/business_entity_resolution/kaggle_run.ipynb`. Internet must be ON (pip installs rapidfuzz,
  sparse_dot_topn and Unidecode only).
- Attach a previous run's output (it contains `artifacts/`) to skip training and run predict only (about 3 h).
  Set `FORCE_TRAIN = True` to retrain.
- Nothing uses a GPU. Accelerator None is fine, and a TPU VM is faster only because of its CPU cores.
- Parallel runs: set `SEED` (and optionally `train_frac` in `CONFIG`) in the first cell; each run writes
  `download_me_seed<SEED>.zip` (matching_results.tsv, pair_scores.parquet, meta.json, run_log.txt).
- Ensemble locally: `python -m src.ensemble --runs dir0 dir1 dir2 dir3 --weights 1 1 1 2
  --test-dir DATASET_DIR/test --out ens_out`.

---

## 7. NEXT, priority order

Done in v3 (commit `59b53c6` + `397a734`): label-free modifier-word features (item 2), lexicon
dropout, anchor / cluster-consistency features (item 4), group features computed on the full
candidate set, leet fix, `src/proxy_report.py`, and the notebook `download_me.zip` bundle.
Still open:

1. **S1-level singleton model.** Decide per S1 whether it has *any* match before attaching records, using
   S1-level features: best / second-best p2, number of candidates, whether the best candidate carries a
   distractor word or a replaced house identifier, and agreement among its candidates. Singletons are worth a
   full 1.0 each, and v1 India predicted 3.5% vs the 5.6% prior.
2. **Label-free distractor-word score.** For each extra or missing token: log(df in S2/S3 / df in S1) computed on
   the data being scored (train at train time, test at test time). It flags French and new modifiers without
   labels. Add **lexicon dropout** in training (randomly hide lexicon values) so the model learns to use this
   score when the lexicon doesn't know a word (France).
3. **Train on a test-like mix**: up-weight negatives about 1.7x (or subsample positives) so probabilities are
   calibrated to test odds instead of relying on calibration.
4. **Cluster-consistency (stage-3) features** from out-of-fold stage-2 predictions: peers of q under the same S1
   sharing q's house identifier but not the S1's, number of high-p peers, and cross-source agreement.
5. **Composite house identifiers** for India ("E-4/39", "C1-39/23", "DE 141/B2") as whole tokens, with
   replaced-identifier features.
6. France self-training lexicon (pseudo-labels from confident test pairs) if France still looks weak.
7. Speed: `prep.parse_frame` is about 130 µs per record.

Local dev loop (needs the zip):
```bash
cd code/business_entity_resolution
python -m src.make_dev_subset --zip PATH/6ab10eb3b23ba_student_resource.zip --out ../../dev
python -m src.pipeline train --dev ../../dev --work ../../art_dev      # ~11 min on 12 threads
```
