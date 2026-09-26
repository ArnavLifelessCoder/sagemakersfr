"""Per-entity F0.5 loss decomposition on a labelled slice + decoding experiments.
usage: err_decomp.py SLICE_DIR OUT_DIR"""
import sys, os, collections
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
SL, OUT = sys.argv[1], sys.argv[2]
gt = pd.read_csv(os.path.join(SL, "gt.tsv"), sep="\t", dtype=str, keep_default_na=False)
truth = {s: set(x for x in m.split(",") if x) for s, m in zip(gt.source1_entity_id, gt.matched_entity_ids)}
owner = {q: s for s, v in truth.items() for q in v}
s1 = pd.read_csv(os.path.join(SL, "test", "test_source1.tsv"), sep="\t", dtype=str, keep_default_na=False, quoting=3)
ctry = dict(zip(s1.entity_id, s1.country))
ps = pd.read_parquet(os.path.join(OUT, "pair_scores.parquet"))
cand = collections.defaultdict(set)
for s, q in zip(ps.s1.values, ps.q.values):
    cand[s].add(q)
best = ps.sort_values("p", ascending=False).drop_duplicates("q")


def f05(P, T):
    if not T and not P:
        return 1.0
    if not T or not P:
        return 0.0
    tp = len(P & T)
    if tp == 0:
        return 0.0
    pr, rc = tp / len(P), tp / len(T)
    return 1.25 * pr * rc / (0.25 * pr + rc)


def evaluate(pred, label, show=False):
    loss = collections.Counter()
    tot = collections.Counter()
    for s in s1.entity_id.values:
        P, T = pred.get(s, set()), truth.get(s, set())
        f = f05(P, T)
        c = ctry[s]
        tot[c] += f
        tot[c + "_n"] += 1
        l = 1 - f
        if l <= 0:
            continue
        if not T:
            loss["FP on singleton"] += l
            continue
        fp = P - T
        fn = T - P
        fn_block = {x for x in fn if x not in cand[s]}
        fn_model = fn - fn_block
        wrong_owner = {x for x in fp if x in owner}
        kinds = []
        if fp - wrong_owner:
            kinds.append("FP distractor")
        if wrong_owner:
            kinds.append("FP stolen from other S1")
        if fn_model:
            kinds.append("FN model")
        if fn_block:
            kinds.append("FN blocking")
        for k in kinds:
            loss[k] += l / len(kinds)
    n = len(s1)
    macro = sum(tot[c] for c in ("India", "US", "France") if c in tot) / n
    print(f"\n=== {label}: macro F0.5 {macro:.5f}  " +
          "  ".join(f"{c} {tot[c] / tot[c + '_n']:.5f}" for c in ("India", "US") if tot[c + '_n']))
    if show:
        for k, v in loss.most_common():
            print(f"   loss {k:26s} {v / n:.5f}")
    return macro


for thr in (0.5, 0.65, 0.8, 0.9):
    b = best[best.p >= thr]
    pred = collections.defaultdict(set)
    for s, q in zip(b.s1.values, b.q.values):
        pred[s].add(q)
    evaluate(pred, f"threshold {thr}", show=(thr == 0.65))
