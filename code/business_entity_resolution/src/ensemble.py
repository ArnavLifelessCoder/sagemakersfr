"""Ensemble several runs (different training seeds) from their pair_scores.parquet files.

    python -m src.ensemble --runs run0_dir run1_dir run2_dir --test-dir DATASET_DIR/test --out out_dir
                           [--threshold T] [--calibrate shape|none]

Each run dir holds pair_scores.parquet (+ meta.json). A pair's ensemble probability is the mean over
runs; a run that did not score the pair (dropped by its stage-1 filter) contributes 0. Then the usual
assignment: every S2/S3 record goes to its best S1 if p >= per-country threshold.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from .data import write_id_lists


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, default=None,
                    help="one weight per run (e.g. more for a model trained on more data)")
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--calibrate", choices=["shape", "none"], default="shape")
    ap.add_argument("--tol", type=float, default=0.02)
    a = ap.parse_args()
    frames, thrs, targets = [], [], []
    for k, r in enumerate(a.runs):
        d = pd.read_parquet(os.path.join(r, "pair_scores.parquet"))
        d = d.drop_duplicates(["s1", "q"])
        frames.append(d.set_index(["s1", "q"])[["p"]].rename(columns={"p": f"p{k}"}))
        meta_p = os.path.join(r, "meta.json")
        if os.path.exists(meta_p):
            m = json.load(open(meta_p))
            thrs.append(m["threshold"])
            targets.append(m.get("target_mps", 3.375))
        print(f"run {r}: {len(d)} pairs")
    allp = pd.concat(frames, axis=1, join="outer").fillna(0.0)
    w = np.array(a.weights if a.weights else [1.0] * len(a.runs), dtype=float)
    w = w / w.sum()
    allp["p"] = allp[[f"p{k}" for k in range(len(a.runs))]].values @ w
    allp = allp.reset_index()
    thr = a.threshold if a.threshold is not None else float(np.dot(w[:len(thrs)], thrs) / w[:len(thrs)].sum()) if thrs else 0.7
    target = float(np.dot(w[:len(targets)], targets) / w[:len(targets)].sum()) if targets else 3.375
    s1 = pd.read_csv(os.path.join(a.test_dir, "test_source1.tsv"), sep="\t", dtype=str,
                     keep_default_na=False, usecols=["entity_id", "country"], quoting=3)
    ctry = dict(zip(s1.entity_id, s1.country))
    allp["country"] = allp.s1.map(ctry)
    best = allp.sort_values("p", ascending=False).drop_duplicates("q")
    n_s1 = s1.country.value_counts().to_dict()
    thr_c = {c: thr for c in n_s1}
    if a.calibrate == "shape":
        for c, n in n_s1.items():
            p = np.sort(best.p.values[best.country.values == c])[::-1]
            k = int(round(target * n))
            above = int((p >= thr).sum())
            if 0 < k and above > k * (1 + a.tol):
                thr_c[c] = float(max(thr, p[k - 1]))
            print(f"  {c}: {above / n:.3f} matches/S1 at thr {thr:.3f} -> thr {thr_c[c]:.4f}")
    best = best[best.p.values >= best.country.map(thr_c).values]
    match = best.groupby("s1", sort=False).q.apply(list).to_dict()
    os.makedirs(a.out, exist_ok=True)
    write_id_lists(os.path.join(a.out, "matching_results.tsv"), s1.entity_id.values, match, "matched_entity_ids")
    stats = pd.DataFrame({"country": s1.country, "n": [len(match.get(x, ())) for x in s1.entity_id]})
    print(stats.groupby("country").n.agg(matches_per_S1="mean", singletons=lambda x: (x == 0).mean()).round(4))
    print("wrote", os.path.join(a.out, "matching_results.tsv"), "thresholds", thr_c)


if __name__ == "__main__":
    main()
