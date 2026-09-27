#!/bin/bash
# Waits for the v5 India + US checkpoints, then assembles submission variants, validates them and packages the primary.
set -u
export PYTHONUTF8=1
ROOT="C:/Users/rrsri/Desktop/PeakAlways/sagemakersfr"
DATA="$ROOT/data/student_resource/dataset"
V5="$ROOT/v5/out_v5_s0"
META="$ROOT/v5/art_full_s0/meta.json"
SUB="$ROOT/submissions"
mkdir -p "$SUB"
cd "$ROOT/v6/business_entity_resolution"
until [ -f "$V5/scores_India.parquet" ] && [ -f "$V5/scores_US.parquet" ]; do sleep 30; done
echo "checkpoints ready at $(date)"
# candidate_pairs.tsv: every scored pair (the stage-1 survivors), written once from the checkpoints
python - <<'EOF'
import os, sys, pandas as pd
sys.path.insert(0, ".")
from src.data import write_id_lists
V5 = "C:/Users/rrsri/Desktop/PeakAlways/sagemakersfr/v5/out_v5_s0"
D = "C:/Users/rrsri/Desktop/PeakAlways/sagemakersfr/data/student_resource/dataset/test"
d = pd.concat([pd.read_parquet(f"{V5}/scores_{c}.parquet", columns=["s1", "q", "p"]) for c in ("France", "India", "US")], ignore_index=True)
d = d.sort_values("p", ascending=False)
s1 = pd.read_csv(f"{D}/test_source1.tsv", sep="\t", dtype=str, keep_default_na=False, usecols=["entity_id"], quoting=3)
write_id_lists(f"{V5}/candidate_pairs.tsv", s1.entity_id.values, d.groupby("s1", sort=False).q.apply(list).to_dict(), "candidate_entity_ids")
print("candidate pairs", len(d))
EOF
run_variant () {  # name, extra args...
  local name=$1; shift
  echo "=== variant $name: $*"
  python -m src.assemble --scores-dir "$V5" --test-dir "$DATA/test" --out "$SUB/$name" --meta "$META" "$@" 2>&1 | tail -5
  cp "$V5/candidate_pairs.tsv" "$SUB/$name/candidate_pairs.tsv"
  python "$DATA/../utils/validate_submission.py" --matching "$SUB/$name/matching_results.tsv" --candidate "$SUB/$name/candidate_pairs.tsv" --test-dir "$DATA/test" 2>&1 | tail -1
  cp "$SUB/$name/matching_results.tsv" "$SUB/matching_results_$name.tsv"
}
run_variant v5_raw --decision expected --odds-div 1.0
run_variant v5_odds19 --decision expected --odds-div 1.9
run_variant v5_bands --decision threshold --thr France=0.99 India=0.9 US=0.968 --calibrate none
if [ -f "$ROOT/v6/out_v7/scores_France.parquet" ]; then
  run_variant v7_fr_odds19 --decision expected --odds-div 1.9 --override "France=$ROOT/v6/out_v7"
  run_variant v7_fr_raw --decision expected --odds-div 1.0 --override "France=$ROOT/v6/out_v7"
fi
python "$ROOT/v5/make_package.py" --out-dir "$SUB/v5_odds19" --zip "$SUB/v5_submission_package.zip" --code "$ROOT/v5/business_entity_resolution" --doc "$ROOT/v5/Documentation_template.md" 2>&1 | head -2
echo "=== finished at $(date)"
ls -la "$SUB"
