#!/bin/bash
# Waits for the xlm-roberta-large test scores, downloads them, waits for the chosen weight (v5/ce_large_w.txt),
# then blends (single model and ensemble), rebuilds the blended validation reference, assembles and validates
# the final candidate files.
set -u
export PYTHONUTF8=1
ROOT="C:/Users/rrsri/Desktop/PeakAlways/sagemakersfr"
H=s_01m3gjwv7yn1etrmamntbpwn6j@ssh.lightning.ai
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no"
cd "$ROOT"
until $SSH $H 'grep -q "test scored" ~/chain2.log' 2>/dev/null; do sleep 120; done
echo "large test scores ready at $(date)"
R=$($SSH $H 'stat -c %s ~/ce_large_test.parquet' 2>/dev/null)
for i in 1 2 3 4; do
  scp -o BatchMode=yes -o StrictHostKeyChecking=no $H:~/ce_large_test.parquet v5/ce_large_test.parquet 2>&1 | grep -v "known hosts"
  L=$(stat -c %s v5/ce_large_test.parquet); echo "attempt $i: $L of $R"
  [ "$L" = "$R" ] && break
done
until [ -f v5/ce_large_w.txt ]; do sleep 30; done
W=$(cat v5/ce_large_w.txt); echo "weight $W"
cd v6/business_entity_resolution
python -m src.blend_ce test --scores-dir ../../v5/out_v5_s0 --ce ../../v5/ce_large_test.parquet --w $W --out ../../v5/out_lblend 2>&1 | tail -1
python -m src.blend_ce test --scores-dir ../../v5/tmp_v7 --ce ../../v5/ce_large_test.parquet --w $W --out ../../v5/out_lblend_v7 2>&1 | tail -1
python -m src.blend_ce test --scores-dir ../../v5/out_ens2 --ce ../../v5/ce_large_test.parquet --w $W --out ../../v5/out_ens2_lblend 2>&1 | tail -1
python - "$W" <<'EOF'
import pandas as pd, numpy as np, os, shutil, sys
w = float(sys.argv[1])
D = pd.read_parquet("../../v5/dump_full_s0/pairs.parquet", columns=["s1_id", "q_id", "p2", "y", "is_val"])
ce = pd.read_parquet("../../v5/ce_large_val.parquet")
D = D.merge(ce, on=["s1_id", "q_id"], how="left")
m = D.ce.notna().values
def logit(p): p = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
z = (1 - w) * logit(D.p2.values[m]) + w * logit(D.ce.values[m])
p = D.p2.values.astype(np.float64); p[m] = 1 / (1 + np.exp(-z))
D["p2"] = p.astype(np.float32)
os.makedirs("../../v5/dump_lblend", exist_ok=True)
D.drop(columns=["ce"]).to_parquet("../../v5/dump_lblend/pairs.parquet", index=False)
shutil.copy("../../v5/dump_full_s0/s1_parsed.parquet", "../../v5/dump_lblend/s1_parsed.parquet")
print("large-blended reference dump written")
EOF
DATA=../../data/student_resource/dataset; SUB=../../submissions
run_variant () {
  local name=$1; local dir=$2; local cand=$3; shift 3
  echo "=== $name: $*"
  python -m src.assemble --scores-dir "$dir" --test-dir $DATA/test --out $SUB/$name --meta ../../v5/art_full_s0/meta.json --decision expected "$@" 2>&1 | grep -E "^  (France|India|US) "
  cp "$cand" $SUB/$name/candidate_pairs.tsv
  python $DATA/../utils/validate_submission.py --matching $SUB/$name/matching_results.tsv --candidate $SUB/$name/candidate_pairs.tsv --test-dir $DATA/test 2>&1 | tail -1
  cp $SUB/$name/matching_results.tsv $SUB/matching_results_$name.tsv
}
run_variant lblend_bandcal ../../v5/out_lblend ../../v5/out_v5_s0/candidate_pairs.tsv --odds-div 1.0 --band-ref ../../v5/dump_lblend --band-gt ../../runs/full_s0_dev/gt.parquet
run_variant lblend_v7_bandcal ../../v5/out_lblend_v7 ../../v5/out_v5_s0/candidate_pairs.tsv --odds-div 1.0 --band-ref ../../v5/dump_lblend --band-gt ../../runs/full_s0_dev/gt.parquet
run_variant lblend_odds19 ../../v5/out_lblend ../../v5/out_v5_s0/candidate_pairs.tsv --odds-div 1.9
run_variant ens_lblend_odds19 ../../v5/out_ens2_lblend ../../v5/out_ens2/candidate_pairs.tsv --odds-div 1.9
echo "final candidates done at $(date)"
