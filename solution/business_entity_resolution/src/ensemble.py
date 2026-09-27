"""Ensemble several runs (different training seeds) from their pair_scores.parquet files.

    python -m src.ensemble --runs run0_dir run1_dir run2_dir --test-dir DATASET_DIR/test --out out_dir
                           [--weights 1 1 1 2] [--threshold T] [--calibrate shape|none]

Each run dir holds pair_scores.parquet (+ meta.json). A pair's ensemble probability is the weighted
mean over runs; a run that did not score the pair (dropped by its stage-1 filter) contributes 0.
Then the usual assignment: every S2/S3 record goes to its best S1 if p >= per-country threshold.
Memory-light: processed one country at a time on 64-bit hashes of the ids.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .data import write_id_lists


def _h(arr):
    return pd.util.hash_array(np.asarray(arr, dtype=object)).astype(np.int64)


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
    ap.add_argument("--decision", choices=["threshold", "expected"], default="expected")
    ap.add_argument("--p-unfound", type=float, default=0.02)
    a = ap.parse_args()
    w = np.array(a.weights if a.weights else [1.0] * len(a.runs), dtype=float)
    w = w / w.sum()
    thrs, targets = [], []
    for r in a.runs:
        mp = os.path.join(r, "meta.json")
        m = json.load(open(mp)) if os.path.exists(mp) else {}
        thrs.append(m.get("threshold", 0.7))
        targets.append(m.get("target_mps", 3.375))
    thr = a.threshold if a.threshold is not None else float(np.dot(w, thrs))
    target = float(np.dot(w, targets))
    print("runs", a.runs, "weights", np.round(w, 3), "base threshold", round(thr, 4), "target", round(target, 3))
    s1 = pd.read_csv(os.path.join(a.test_dir, "test_source1.tsv"), sep="\t", dtype=str,
                     keep_default_na=False, usecols=["entity_id", "country"], quoting=3)
    match = {}
    for c in sorted(s1.country.unique()):
        n_c = int((s1.country == c).sum())
        parts = []
        for k, r in enumerate(a.runs):
            t = pq.read_table(os.path.join(r, "pair_scores.parquet"), columns=["s1", "q", "p"],
                              filters=[("country", "=", c)])
            d = t.to_pandas()
            parts.append(pd.DataFrame({"sk": _h(d.s1.values), "qk": _h(d.q.values),
                                       "p": (d.p.values * w[k]).astype(np.float32)}))
            del d, t
        allp = pd.concat(parts, ignore_index=True)
        del parts
        allp = allp.groupby(["sk", "qk"], sort=False).p.sum().reset_index()
        best = allp.sort_values("p", ascending=False).drop_duplicates("qk")
        del allp
        k = int(round(target * n_c))
        if a.decision == "expected":
            from .model import assign_expected
            s_code, _ = pd.factorize(best.sk.values)
            pv = best.p.values.astype(np.float64)
            q_code = np.arange(len(best))
            kept = assign_expected(q_code, s_code, pv, p_unfound=a.p_unfound)[0]
            raw_n = len(kept)
            div = 1.0
            if a.calibrate == "shape" and 0 < k and len(kept) > k * (1 + a.tol):
                lo, hi = 1.0, 8.0
                for _ in range(7):
                    mid = (lo + hi) / 2
                    km = assign_expected(q_code, s_code, pv, p_unfound=a.p_unfound, odds_div=mid)[0]
                    if len(km) > k * (1 + a.tol):
                        lo = mid
                    else:
                        hi, kept = mid, km
                div = hi
            best = best.iloc[np.sort(kept)]
            print(f"  {c}: expected-F0.5 {raw_n / n_c:.3f} matches/S1 raw -> odds/{div:.3f}: {len(best) / n_c:.3f} matches/S1 (target {target:.3f})")
        else:
            thr_c = thr
            p = best.p.values
            above = int((p >= thr).sum())
            if a.calibrate == "shape" and 0 < k and above > k * (1 + a.tol):
                thr_c = float(max(thr, np.sort(p)[::-1][k - 1]))
            best = best[best.p.values >= thr_c]
            print(f"  {c}: {above / n_c:.3f} matches/S1 at thr {thr:.3f} -> thr {thr_c:.4f}, "
                  f"{len(best) / n_c:.3f} matches/S1")
        # map hashes back to ids (only for the kept pairs)
        keep = set(zip(best.sk.values, best.qk.values))
        for r in a.runs:
            d = pq.read_table(os.path.join(r, "pair_scores.parquet"), columns=["s1", "q"],
                              filters=[("country", "=", c)]).to_pandas()
            sk, qk = _h(d.s1.values), _h(d.q.values)
            for s, q_, a_, b_ in zip(d.s1.values, d.q.values, sk, qk):
                if (a_, b_) in keep:
                    match.setdefault(s, []).append(q_)
                    keep.discard((a_, b_))
            if not keep:
                break
    os.makedirs(a.out, exist_ok=True)
    write_id_lists(os.path.join(a.out, "matching_results.tsv"), s1.entity_id.values, match, "matched_entity_ids")
    stats = pd.DataFrame({"country": s1.country, "n": [len(match.get(x, ())) for x in s1.entity_id]})
    print(stats.groupby("country").n.agg(matches_per_S1="mean", singletons=lambda x: (x == 0).mean()).round(4))
    print("wrote", os.path.join(a.out, "matching_results.tsv"))


if __name__ == "__main__":
    main()
