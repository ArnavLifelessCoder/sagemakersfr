# Business Entity Resolution — reproducible pipeline (final submission, public LB 0.984)

The scored file `output/matching_results.tsv` is the average of three prediction runs of the same
pipeline family, blended with a fine-tuned cross-encoder and decided per entity with an expected-F0.5
rule. Everything needed to regenerate both output files from the challenge data is in this folder.
Only the provided training data and MIT/Apache models are used; no external data or services.

| run | code | model | what it adds |
|---|---|---|---|
| A: v5 seed 0 | `src/` | `artifacts/v5_seed0` (LightGBM x2, lexicon, transliteration table, meta) | three-channel blocking, composite identifiers, France city→region backfill |
| B: v6 seed 1 | `src/v6_variant/` (same pipeline + neighbour-state search + name/address frequency features) | `artifacts/v6_seed1` | trained on a different stratified 30% slice (seed 1) |
| C: seed 202 | `src/partner_line/` (the team's other code line: v4 model family + wider blocking + OCR fold) | trained on Kaggle with `kaggle_run.ipynb`, SEED 202, `train_frac` 0.3 | model diversity |
| cross-encoder | `src/cross_encoder.py` | `xlm-roberta-base` fine-tuned on 1.5M training pairs (weights not included: 1.1 GB; retrain in 33 min on one GPU) | multilingual pair scorer, blended at weight 0.3 |

The three runs' probabilities are averaged per pair (`src/ens_ckpt.py`), blended with the cross-encoder in
logit space (`src/blend_ce.py`), and decided with the expected-F0.5 rule (`src/assemble.py`).
`output/candidate_pairs.tsv` is the union of the three runs' stage-1 survivors: exactly the pairs the
final ensemble scored (13,264,903 pairs).

## Layout

```
src/
  normalize.py      name / address normalisation (legal forms, states, composite identifiers, transliteration)
  prep.py           parallel parsing, transliteration table + phonetic fallback, city->state map
  blocking.py       per-(country, state) sparse TF-IDF top-k retrieval, three channels
  features.py       pairwise, anchor, peer (collective) features
  model.py          lexicon, label-free vocabulary/modifier features, LightGBM, expected-F0.5 decision, metric
  pipeline.py       train / predict entry points (per-country checkpoints, --countries, --selftrain-lex)
  data.py           TSV I/O, submission writers
  ens_ckpt.py       average per-country score checkpoints of several runs
  blend_ce.py       blend cross-encoder scores with the LightGBM probabilities (val: choose the weight; test: apply)
  assemble.py       decision rule + per-country calibration -> matching_results.tsv (any variant in minutes)
  export_pairs.py   export text pairs for the cross-encoder (train from a dump, test from scores)
  cross_encoder.py  fine-tune / score the cross-encoder (GPU)
  ensemble.py, proxy_report.py, make_dev_subset.py   older tools kept for reference
  v6_variant/       run B's code (verbatim)
  partner_line/     run C's code (verbatim, with its own README.md, requirements.txt and kaggle_run.ipynb)
artifacts/          trained models for runs A and B (retrainable with the commands below)
APPROACH.md         full method write-up with measurements
requirements.txt    pinned CPU environment; requirements-gpu.txt for the cross-encoder
```

## Environment

```bash
pip install -r requirements.txt            # Python 3.11; CPU pipeline
pip install -r requirements-gpu.txt        # only for the cross-encoder (CUDA GPU)
```
Runs A and B were produced on an 18-core laptop (48 GB RAM): training 35 min each with `--workers 12
--threads 12`, prediction of the full test 60-90 min each. Run C was produced on Kaggle (4 CPUs, 30 GB):
about 75 min training + 2 h prediction. The cross-encoder took 33 min to train and 49 min to score the
12.2M test pairs on one L40S.

`DATASET_DIR` is the `student_resource/dataset` folder (holds `train/` and `test/`).

## Reproduce end to end

### 1. Run A (v5 seed 0)
```bash
cd code/business_entity_resolution
echo '{"train_frac": 0.3, "seed": 0}' > cfg_s0.json
python -m src.pipeline train   --data-dir DATASET_DIR --work art_A --config cfg_s0.json --workers 12 --threads 12 --dump dump_A
python -m src.pipeline predict --data-dir DATASET_DIR --work art_A --out out_A --workers 12 --threads 12
```
`train` writes `stage1.txt`, `stage2.txt`, `lexicon.json`, `translit.json`, `meta.json` (feature lists,
thresholds, decision rule, calibration target, config); the shipped `artifacts/v5_seed0` can be used as
`--work` to skip training. `predict` writes `out_A/scores_<country>.parquet` (one checkpoint per
country), `candidate_pairs.tsv`, `matching_results.tsv` and `pair_scores.parquet`. `--dump` keeps every
scored training pair with its features and labels (used for the cross-encoder export and the blend weight).

### 2. Run B (v6 seed 1)
```bash
echo '{"train_frac": 0.3, "seed": 1}' > cfg_s1.json
python -m src.v6_variant.pipeline train   --data-dir DATASET_DIR --work art_B --config cfg_s1.json --workers 12 --threads 12
python -m src.v6_variant.pipeline predict --data-dir DATASET_DIR --work art_B --out out_B --workers 12 --threads 12
```
(shipped model: `artifacts/v6_seed1`).

### 3. Run C (seed 202, partner line)
Follow `src/partner_line/README.md`: train the v4-family model with seed 2 (`train_frac` 0.3) and predict
with the seed-202 configuration of `src/partner_line/kaggle_run.ipynb` (wider blocking, OCR fold; the model's
own calibration, no post-processing). The run yields `pair_scores.parquet` with columns `s1, q, p, country`;
split it into per-country checkpoints:
```bash
python - <<'EOF'
import pandas as pd, os
d = pd.read_parquet("out_C/pair_scores.parquet"); os.makedirs("out_C", exist_ok=True)
for c, g in d.groupby("country"):
    g.reset_index(drop=True).to_parquet(f"out_C/scores_{c}.parquet", index=False)
EOF
```

### 4. Cross-encoder (GPU)
```bash
python -m src.export_pairs train --dump dump_A --data-dir DATASET_DIR --out pairs_train.parquet --max-neg-per-pos 3
python src/cross_encoder.py train --pairs pairs_train.parquet --out ce_model --model xlm-roberta-base --epochs 1 --bs 64 --max-train 1500000
# validation pairs = the is_val rows of pairs_train.parquet -> choose the blend weight (0.3 was chosen: F0.5 0.9906 -> 0.9918)
python src/cross_encoder.py score --pairs pairs_val.parquet  --model-dir ce_model --out ce_val.parquet --bs 256
python -m src.blend_ce val --dump dump_A --ce ce_val.parquet --gt GT_PARQUET       # GT_PARQUET: train_ground_truth rows of the dumped S1 (see APPROACH.md)
# test pairs = union of the three runs' scored pairs with their texts
python -m src.ens_ckpt --runs out_A out_B out_C --out out_ens
python - <<'EOF'
import pandas as pd
pd.concat([pd.read_parquet(f"out_ens/scores_{c}.parquet") for c in ("France","India","US")]).to_parquet("out_ens/pair_scores.parquet", index=False)
EOF
python -m src.export_pairs test --scores out_ens/pair_scores.parquet --data-dir DATASET_DIR --out pairs_test.parquet
python src/cross_encoder.py score --pairs pairs_test.parquet --model-dir ce_model --out ce_test.parquet --bs 256
```

### 5. Ensemble, blend, decide, validate
```bash
python -m src.blend_ce test --scores-dir out_ens --ce ce_test.parquet --w 0.3 --out out_ens_blend
python -m src.assemble --scores-dir out_ens_blend --test-dir DATASET_DIR/test --out output --meta art_A/meta.json --decision expected --odds-div 1.0
python - <<'EOF'
# candidate_pairs.tsv = every pair the ensemble scored
import pandas as pd, sys
sys.path.insert(0, "."); from src.data import write_id_lists
d = pd.concat([pd.read_parquet(f"out_ens/scores_{c}.parquet", columns=["s1","q","p"]) for c in ("France","India","US")]).sort_values("p", ascending=False)
s1 = pd.read_csv("DATASET_DIR/test/test_source1.tsv", sep="\t", dtype=str, keep_default_na=False, usecols=["entity_id"], quoting=3)
write_id_lists("output/candidate_pairs.tsv", s1.entity_id.values, d.groupby("s1", sort=False).q.apply(list).to_dict(), "candidate_entity_ids")
EOF
python STUDENT_RESOURCE/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir DATASET_DIR/test --check-ids
```
Expected per-country statistics of the final file: France 3.281 matches per S1 (5.57% singletons),
India 3.358 (5.76%), US 3.395 (5.68%). LightGBM training and the cross-encoder fine-tuning are not
bit-reproducible across machines; a regenerated file matches the shipped one up to that noise.

### Decision variants (minutes each, no re-scoring)
`src.assemble` reads the checkpoints and applies: `--decision threshold|expected`, `--odds-div D`
(density correction), `--thr France=0.99 ...` (fixed thresholds), `--band-ref DUMP --band-gt GT`
(label-free band calibration), `--override France=DIR` (take one country from another run).
On the public leaderboard the raw expected-F0.5 decision beat every more conservative variant
(0.975 raw-ish vs 0.973 band-calibrated for the single v5 model).

## Fast local iteration
```bash
python -m src.make_dev_subset --zip STUDENT_RESOURCE.zip --out dev
python -m src.pipeline train --dev dev --work art_dev --dump dump_dev          # ~6 min on 12 cores
```
