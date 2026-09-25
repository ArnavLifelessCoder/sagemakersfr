import json, sys

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md("""
# Amazon ML Challenge 2026 — Business Entity Resolution
### Two-stage blocking + LightGBM matcher · end-to-end Kaggle runner

This notebook reproduces the full pipeline: **data → normalisation → blocking → stage-1 filter → stage-2 matcher → assignment → submission files**.

| Step | What happens | Output |
|---|---|---|
| 0 | Locate inputs, install 3 small libraries | — |
| 1 | **Train** on a slice of the labelled training data, tune threshold on held-out S1 entities | `artifacts/` (models, lexicon, meta) |
| 2 | **Predict** on the full test set | `output/matching_results.tsv`, `output/candidate_pairs.tsv` |
| 3 | **Validate** with the official `validate_submission.py` | PASS / FAIL |
| 4 | **Submit** | upload `matching_results.tsv` |

**Before running**
1. *Add Input* → the challenge zip (private dataset) **and** the `ber-code` dataset (the `business_entity_resolution` folder).
2. *Settings* → **Internet ON** (only for `pip install`), accelerator **TPU VM** if available (used purely for its many CPU cores), otherwise GPU/None — the pipeline is CPU-bound.
3. *Run All*. No external data or APIs are used anywhere — compliant with the challenge rules.
""")

md("""
---
## 0 · Setup
### 0.1 Locate the dataset, the code and the official validator
Paths are discovered automatically under `/kaggle/input` (the `__MACOSX` junk folder inside the zip is ignored).
""")
code("""
import glob, os, time, json

def find(pattern):
    hits = [p for p in glob.glob('/kaggle/input/**/' + pattern, recursive=True) if '__MACOSX' not in p]
    assert hits, f'could not find {pattern} under /kaggle/input - did you add both datasets?'
    return hits[0]

DATA      = os.path.dirname(os.path.dirname(find('train/train_source1.tsv')))  # .../student_resource/dataset
CODE_SRC  = os.path.dirname(os.path.dirname(find('src/pipeline.py')))         # .../business_entity_resolution
VALIDATOR = find('utils/validate_submission.py')
DOC_TMPL  = find('Documentation_template.md')

WORK = '/kaggle/working'
OUT  = f'{WORK}/output'      # submission TSVs

# Re-use a previously trained model if a run's `artifacts/` folder is attached as input
# (Add Input -> your earlier notebook's output). Set FORCE_TRAIN = True to retrain anyway.
FORCE_TRAIN = False
prev = [os.path.dirname(p) for p in glob.glob('/kaggle/input/**/artifacts/meta.json', recursive=True)]
if prev and not FORCE_TRAIN:
    ART, TRAIN = prev[0], False
else:
    ART, TRAIN = f'{WORK}/artifacts', True
sh = get_ipython().system

print('dataset   :', DATA)
print('code      :', CODE_SRC)
print('validator :', VALIDATOR)
print('CPU cores :', os.cpu_count())
print('artifacts :', ART, '(re-using trained model, training will be skipped)' if not TRAIN else '(will train)')
""")
md("""
### 0.2 Copy code to a writable folder and install dependencies
Only `rapidfuzz`, `sparse_dot_topn` and `Unidecode` are missing from the Kaggle image (pandas / numpy / scikit-learn / LightGBM are preinstalled).
""")
code("""
!rm -rf {WORK}/ber && cp -r "{CODE_SRC}" {WORK}/ber
!pip install -q rapidfuzz==3.14.5 sparse_dot_topn==1.2.0 Unidecode==1.4.0
%cd {WORK}/ber
!ls src
""")

md("""
---
## 1 · Train
Stages inside `src.pipeline train`:

1. **Slice** ~25 % of training S1 entities (whole states, so distractor density stays realistic) + all their true matches + nearby unmatched records.
2. **Transliteration table** learned from matched pairs (Indic-script names → Latin).
3. **Normalise** names & addresses (legal forms incl. typos, DBA / "formerly known as", domains, states, house numbers, street abbreviations).
4. **Blocking** per (country, state): sparse TF-IDF top-K on name char-3-grams and address tokens.
5. **Stage-1 LightGBM** on cheap features → threshold keeping 99.95 % of true pairs (≈1 candidate per record).
6. **Stage-2 LightGBM** on the full feature set (token lexicon, vocabulary, house-number logic, context).
7. **Threshold tuning** for macro F0.5 on a 20 % held-out set of S1 entities.

Watch for the line **`best validation F0.5 … at threshold …`** at the end.
*Skipped automatically when trained artifacts were found in section 0.*
""")
code("""
if TRAIN:
    t0 = time.time()
    sh(f'python -u -m src.pipeline train --data-dir "{DATA}" --work {ART} 2>&1 | grep -v Warning')
    print(f'train finished in {(time.time()-t0)/60:.1f} min')
else:
    print('training skipped - using', ART)
""")
md("### 1.1 Training summary")
code("""
meta = json.load(open(f'{ART}/meta.json'))
print(f"validation macro F0.5 : {meta['val_f05']:.5f}")
print(f"decision threshold    : {meta['threshold']:.3f}")
print(f"stage-1 threshold     : {meta['stage1_threshold']:.5f}")
print(f"stage-2 features      : {len(meta['features2'])}")
print(f"calibration target    : {meta.get('target_mps', 3.375):.3f} matches per S1")
sh(f'ls -la {ART}')
""")
md("""
> **12 h session limit:** if time is tight, stop here, *Save Version* so `artifacts/` becomes a notebook output,
> then run section 2 in a new session with that output added as an input (set `ART` to its path).
""")

md("""
---
## 2 · Predict on the test set
Runs country by country (India, US, **France** — unseen in training) to keep memory low:
parse → block → stage-1 filter (survivors are written to `candidate_pairs.tsv`) → stage-2 scoring →
each S2/S3 record is assigned to its best S1 if the probability clears the threshold.

**Test-time adaptation (no labels used):**
* *phonetic transliteration fallback* — native-script words unknown to the learned table are mapped by sound to
  the most frequent Latin name word of the same country (e.g. ज्वेलर्स → jewellers);
* *per-country threshold calibration* — the test set holds ~1.7x more distractors per S1 than training, so each
  country's threshold is raised (never lowered) until predicted matches per S1 equal the rate measured on validation
  (training clusters have the same size distribution in every country).

`pair_scores.parquet` (all scored pairs) is also saved so thresholds can be re-tuned offline.
""")
code("""
t0 = time.time()
!python -u -m src.pipeline predict --data-dir "{DATA}" --work {ART} --out {OUT}
print(f'predict finished in {(time.time()-t0)/60:.1f} min')
""")
md("### 2.1 Quick look at the predictions")
code("""
import pandas as pd
m = pd.read_csv(f'{OUT}/matching_results.tsv', sep='\\t', dtype=str, keep_default_na=False)
n_match = m.matched_entity_ids.str.count(',') + (m.matched_entity_ids != '')
print('S1 rows               :', len(m))
print('predicted singletons  :', round((n_match == 0).mean(), 4), '(train prior ~0.056)')
print('mean matches per S1   :', round(n_match.mean(), 3))
s1c = pd.read_csv(f'{DATA}/test/test_source1.tsv', sep='	', dtype=str, keep_default_na=False, usecols=['entity_id', 'country'])
m = m.merge(s1c, left_on='source1_entity_id', right_on='entity_id')
m['n'] = n_match.values
print(m.groupby('country').agg(S1=('n', 'size'), matches_per_S1=('n', 'mean'), singleton_share=('n', lambda x: (x == 0).mean())).round(4))
""")

md("""
---
## 3 · Validate with the official checker
Must print **PASS** before uploading. `--check-ids` also verifies every ID exists in the test set.
""")
code("""
!python "{VALIDATOR}" --matching {OUT}/matching_results.tsv --candidate {OUT}/candidate_pairs.tsv --test-dir "{DATA}/test" --check-ids
""")

md("""
---
## 4 · Submit
Download **`output/matching_results.tsv`** (right panel → *Output*) and upload it on the portal.
That is the only file the leaderboard needs.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(sys.argv[1], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(len(cells), "cells ->", sys.argv[1])
