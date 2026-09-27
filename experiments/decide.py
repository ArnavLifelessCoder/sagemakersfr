"""Decision rules on top of stage-2 probabilities, evaluated on a training dump.

Compares, on the validation S1 entities:
  * fixed global threshold (the current rule)
  * per-S1 expected-F0.5 rule: keep the prefix (by p) of an S1's argmax records that maximises
    the expected F0.5 under independent Bernoulli truths, the empty prediction included
Both with the raw probability and with a prior correction for the test distractor density
(odds divided by `c`), scored with fp_weight 1.0 (train density) and 1.9 (test-like).

    python experiments/decide.py DUMP_DIR [--dev DEV_DIR] [--val-pct 20]
"""
import argparse
import collections
import os
import sys
import zlib

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "code", "business_entity_resolution"))
from src.data import gt_pairs  # noqa: E402
from src.model import fbeta_macro, to_mapping  # noqa: E402


def pb_pmf(ps):
    """Poisson-binomial PMF of the number of successes among independent Bernoulli(p)."""
    pmf = np.zeros(len(ps) + 1)
    pmf[0] = 1.0
    for i, p in enumerate(ps):
        pmf[1:i + 2] = pmf[1:i + 2] * (1 - p) + pmf[0:i + 1] * p
        pmf[0] *= (1 - p)
    return pmf


def expected_f(ps, m, p_unfound=0.0, fp_weight=1.0):
    """Expected F0.5 of predicting the top-m of candidates with probabilities ps (sorted desc).

    F = 1.25 tp / (fp_weight * fp + tp + 0.25 (tp + fn)) with tp ~ PB(ps[:m]), fn ~ PB(ps[m:]) (+ unfound).
    For m = 0 the score is 1 iff there is no true record at all."""
    if m == 0:
        return float(np.prod(1 - np.asarray(ps))) * (1 - p_unfound)
    tp_pmf = pb_pmf(ps[:m])
    fn_pmf = pb_pmf(ps[m:]) if m < len(ps) else np.array([1.0])
    if p_unfound > 0:  # one extra hidden true record with probability p_unfound
        fn_pmf = np.convolve(fn_pmf, [1 - p_unfound, p_unfound])
    tp = np.arange(m + 1)[:, None]
    fn = np.arange(len(fn_pmf))[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(tp > 0, 1.25 * tp / (fp_weight * (m - tp) + tp + 0.25 * (tp + fn)), 0.0)
    return float((tp_pmf[:, None] * fn_pmf[None, :] * f).sum())


def decide_expected(best, p_unfound=0.0, fp_weight=1.0, min_p=0.05):
    """best: DataFrame [s1_id, q_id, p] with one row per record (its argmax S1). Returns kept (s1_id, q_id) lists."""
    keep_s, keep_q = [], []
    b = best[best.p >= min_p].sort_values(["s1_id", "p"], ascending=[True, False])
    for s, g in b.groupby("s1_id", sort=False):
        ps = g.p.values.astype(float)
        qs = g.q_id.values
        best_m, best_e = 0, expected_f(ps, 0, p_unfound, fp_weight)
        for m in range(1, len(ps) + 1):
            e = expected_f(ps, m, p_unfound, fp_weight)
            if e > best_e:
                best_m, best_e = m, e
        keep_s.extend([s] * best_m)
        keep_q.extend(qs[:best_m])
    return keep_s, keep_q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--dev", default=os.path.join(ROOT, "dev"))
    ap.add_argument("--val-pct", type=int, default=20)
    a = ap.parse_args()
    D = pd.read_parquet(os.path.join(a.dump, "pairs.parquet"), columns=["s1_id", "q_id", "p2", "y", "is_val"])
    gt = pd.read_parquet(os.path.join(a.dev, "gt.parquet")) if os.path.exists(os.path.join(a.dev, "gt.parquet")) else None
    if gt is None:
        raise SystemExit("needs the dev gt.parquet (full-data dumps: pass --dev with a gt.parquet built from train_ground_truth)")
    pairs = gt_pairs(gt)
    truth = to_mapping(pairs.s1.values, pairs.other.values)
    val = [s for s in gt.source1_entity_id.values if zlib.crc32(s.encode()) % 100 < a.val_pct]
    val_set = set(val)
    # argmax assignment per record (over all rows, as at test time), keep only validation S1
    best = D.sort_values("p2", ascending=False).drop_duplicates("q_id")
    best = best[best.s1_id.isin(val_set)][["s1_id", "q_id", "p2"]].rename(columns={"p2": "p"}).reset_index(drop=True)
    print(f"validation S1 {len(val)}  argmax records on them {len(best)}")
    for fpw in (1.0, 1.9):
        print(f"\n=== metric fp_weight {fpw}")
        rows = []
        for t in np.arange(0.5, 0.96, 0.05):
            k = best[best.p >= t]
            sc = fbeta_macro(to_mapping(k.s1_id.values, k.q_id.values), truth, val, fp_weight=fpw)
            rows.append(("threshold", round(float(t), 2), sc))
        bt = max(rows, key=lambda r: r[2])
        print(f"fixed threshold: best {bt[2]:.5f} at {bt[1]}   (0.75 -> {[r[2] for r in rows if r[1] == 0.75][0]:.5f})")
        for c in (1.0, 1.5, 1.9, 2.5, 3.5):
            b2 = best.copy()
            b2["p"] = b2.p / (b2.p + c * (1 - b2.p))
            for pu in (0.0, 0.02):
                ks, kq = decide_expected(b2, p_unfound=pu, fp_weight=1.0)
                sc = fbeta_macro(to_mapping(ks, kq), truth, val, fp_weight=fpw)
                print(f"expected-F rule: odds/{c:<4} p_unfound {pu:<5} -> {sc:.5f}   kept {len(ks)} records")
    print("\nnote: fp_weight 1.9 emulates the test distractor density on the metric only")


if __name__ == "__main__":
    main()
