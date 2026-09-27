#!/bin/bash
# Overnight chain: v5 seeds 1 and 2 (train + predict), then the 3-seed ensemble. Run from anywhere:
#   bash v5/overnight.sh > v5/overnight.log 2>&1
set -u
export PYTHONUTF8=1
ROOT="C:/Users/rrsri/Desktop/PeakAlways/sagemakersfr"
CODE="$ROOT/v6/business_entity_resolution"
DATA="$ROOT/data/student_resource/dataset"
cd "$CODE"
for SEED in 1; do
  echo "=== seed $SEED train $(date)"
  echo "{\"train_frac\": 0.3, \"seed\": $SEED}" > "../../v6/cfg_full_s$SEED.json"
  python -m src.pipeline train --data-dir "$DATA" --work "../../v6/art_full_s$SEED" --config "../../v6/cfg_full_s$SEED.json" --workers 12 --threads 12 > "../../v6/train_full_s$SEED.log" 2>&1
  grep -E "blocked pairs|best test-like|expected-F|decision rule|validation:" "../../v6/train_full_s$SEED.log"
  echo "=== seed $SEED predict $(date)"
  python -m src.pipeline predict --data-dir "$DATA" --work "../../v6/art_full_s$SEED" --out "../../v6/out_v6_s$SEED" --workers 12 --threads 12 > "../../v6/predict_v6_s$SEED.log" 2>&1
  grep -E "candidate pairs|decide|wrote" "../../v6/predict_v6_s$SEED.log"
done
echo "=== ensemble $(date)"
python -m src.ensemble --runs ../../v5/out_v5_s0 ../../v6/out_v6_s1 --test-dir "$DATA/test" --out ../../v6/out_ens > ../../v6/ensemble.log 2>&1
cat ../../v6/ensemble.log
cp ../../v5/out_v5_s0/candidate_pairs.tsv ../../v6/out_ens/candidate_pairs.tsv
python "$DATA/../utils/validate_submission.py" --matching ../../v6/out_ens/matching_results.tsv --candidate ../../v6/out_ens/candidate_pairs.tsv --test-dir "$DATA/test" | tail -3
echo "=== done $(date)"
