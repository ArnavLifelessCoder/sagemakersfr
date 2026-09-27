# Business Entity Resolution — reproducible pipeline

Matches every Source-1 business to its Source-2/3 records (many-to-one) with
three-channel blocking → stage-1 filter → stage-2 LightGBM matcher → per-entity expected-F0.5 decision
calibrated per country to the cluster-size prior.
Only the provided training data is used; no external data, APIs or pretrained models.

## Layout

```
src/
  normalize.py      name / address normalisation (legal forms, states, abbreviations, house numbers)
  prep.py           parallel record parsing + transliteration table learned from training pairs
  blocking.py       per-(country, state) sparse TF-IDF top-K retrieval (name char-3grams, address tokens)
  features.py       pairwise features (rapidfuzz vectorised + python token/number logic + context)
  model.py          token lexicon, vocabulary features, LightGBM, assignment, macro F0.5 metric
  pipeline.py       train / predict entry points
  data.py           TSV I/O, submission writers
  ensemble.py       average pair probabilities of several seed runs, then the same decision rule
  proxy_report.py   label-free quality proxies on the test set (matches per S1, singletons, modifier words)
  export_pairs.py / cross_encoder.py  optional neural pair scorer (not used in the submitted run unless stated)
  make_dev_subset.py  optional: small regional train slice for fast local iteration
kaggle_run.ipynb    notebook that runs everything on Kaggle
requirements.txt    pinned versions used for development
```

## Reproduce

`DATASET_DIR` is the `student_resource/dataset` folder (contains `train/` and `test/`).

```bash
pip install -r requirements.txt
python -m src.pipeline train   --data-dir DATASET_DIR --work artifacts
python -m src.pipeline predict --data-dir DATASET_DIR --work artifacts --out output
python ../../student_resource/utils/validate_submission.py \
       --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv \
       --test-dir DATASET_DIR/test
```

`train` writes to `artifacts/`: `stage1.txt`, `stage2.txt` (LightGBM), `lexicon.json`,
`translit.json`, `meta.json` (feature lists, thresholds, validation F0.5, config).
`predict` writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`
(the stage-1 survivors, i.e. exactly the pairs scored by the final model).

Options: `--threads N` / `--workers N` (CPU parallelism), `--config cfg.json` to override
`DEFAULT_CFG` in `pipeline.py` (e.g. `{"train_frac": 0.4, "seed": 1}`), `predict --decision threshold|expected`,
`predict --calibrate shape|none`, `train --dump DIR` (writes every scored pair for error analysis).

Runtime (4-core Kaggle CPU): train ≈ 2–3 h with `train_frac=0.25`, predict ≈ 3–4 h;
a Kaggle TPU-VM session (many CPU cores) is several times faster. Peak RAM < 30 GB.

## Fast local iteration

```bash
python -m src.make_dev_subset --zip path/to/student_resource.zip --out dev
python -m src.pipeline train --dev dev --work art_dev
```
