"""Per-entity loss decomposition of a training dump (pipeline train --dump DIR).

Reports, on the validation S1 entities, how much macro F0.5 is lost and to what:
  * singleton S1 given a match (score 0)
  * false merges on non-singletons (precision loss)
  * missed true records: lost in blocking / lost in stage-1 / rejected by stage-2 / assigned to another S1
and prints examples of each category with the decisive features.

    python experiments/loss_decomp.py DUMP_DIR [threshold] [--country X] [--examples N]
"""
import argparse
import collections
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "code", "business_entity_resolution"))
from src.data import gt_pairs  # noqa: E402


def f05(tp, fp, fn):
    if tp == 0:
        return 0.0
    p, r = tp / (tp + fp), tp / (tp + fn)
    return 1.25 * p * r / (0.25 * p + r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("threshold", type=float, nargs="?", default=None)
    ap.add_argument("--dev", default=os.path.join(ROOT, "dev"))
    ap.add_argument("--country", default=None)
    ap.add_argument("--examples", type=int, default=8)
    a = ap.parse_args()
    D = pd.read_parquet(os.path.join(a.dump, "pairs.parquet"))
    s1 = pd.read_parquet(os.path.join(a.dump, "s1_parsed.parquet")).set_index("entity_id")
    q = pd.read_parquet(os.path.join(a.dump, "q_parsed.parquet")).set_index("entity_id")
    raw = {}
    for k in (1, 2, 3):
        r = pd.read_parquet(os.path.join(a.dev, f"src{k}.parquet"))
        raw.update(zip(r.entity_id.values, zip(r.business_name.values, r.business_address.values)))
    gt = pd.read_parquet(os.path.join(a.dev, "gt.parquet"))
    pairs = gt_pairs(gt)
    truth = collections.defaultdict(set)
    owner_of = {}
    for s, o in zip(pairs.s1.values, pairs.other.values):
        truth[s].add(o)
        owner_of[o] = s
    meta_thr = a.threshold
    if meta_thr is None:
        import json
        cand = [os.path.join(a.dump, "..", d, "meta.json") for d in os.listdir(os.path.join(a.dump, "..")) if d.startswith("art")]
        meta_thr = 0.625
        for c in cand:
            if os.path.exists(c) and os.path.basename(os.path.dirname(a.dump + "/")).replace("dump", "art") in c:
                meta_thr = json.load(open(c))["threshold"]
    thr = meta_thr
    V = D[D.is_val].copy()
    V["country"] = s1.country.reindex(V.s1_id.values).values
    if a.country:
        V = V[V.country == a.country]
    # validation entity universe: all val S1 (from the crc split), singletons included
    import zlib
    all_val = [s for s in gt.source1_entity_id.values if zlib.crc32(s.encode()) % 100 < 20]
    if a.country:
        ctry_of = dict(zip(s1.index.values, s1.country.values))
        all_val = [s for s in all_val if ctry_of.get(s) == a.country]
    all_val_set = set(all_val)
    # assignment: each q to its best S1 if p2 >= thr (over ALL rows, val and train, like at test time)
    best = D.sort_values("p2", ascending=False).drop_duplicates("q_id")
    best = best[best.p2 >= thr]
    pred = collections.defaultdict(set)
    for s, o in zip(best.s1_id.values, best.q_id.values):
        pred[s].add(o)
    assigned_to = dict(zip(best.q_id.values, best.s1_id.values))
    # candidate sets per (s1, q) presence in the scored pairs
    scored = set(zip(D.s1_id.values, D.q_id.values))
    p2_of = dict(zip(zip(D.s1_id.values, D.q_id.values), D.p2.values))
    missed_block = set()
    missed_s1 = set()
    for name, store in (("missed_blocking.csv", missed_block), ("missed_stage1.csv", missed_s1)):
        p = os.path.join(a.dump, name)
        if os.path.exists(p):
            m = pd.read_csv(p)
            store.update(zip(m.s1.values, m.other.values))
    loss = collections.Counter()
    cnt = collections.Counter()
    ex = collections.defaultdict(list)
    tot = 0.0
    for s in all_val:
        T = truth.get(s, set())
        P = pred.get(s, set())
        sc = 1.0 if (not T and not P) else (0.0 if (not T or not P) else f05(len(P & T), len(P - T), len(T - P)))
        tot += sc
        l = 1.0 - sc
        if l <= 0:
            continue
        if not T:
            loss["singleton_given_match"] += l
            cnt["singleton_given_match"] += 1
            ex["singleton_given_match"].append((s, l, P, T))
            continue
        if not P:
            key = "all_missed"
        else:
            key = "mixed"
        fp = P - T
        fn = T - P
        # attribute the loss proportionally to error records
        n_err = len(fp) + len(fn)
        for o in fp:
            k = "fp_distractor" if o not in owner_of else "fp_belongs_to_other_s1"
            loss[k] += l / n_err
            cnt[k] += 1
            ex[k].append((s, l, o, T))
        for o in fn:
            if (s, o) in missed_block:
                k = "fn_blocking"
            elif (s, o) in missed_s1 and (s, o) not in scored:
                k = "fn_stage1"
            elif o in assigned_to:
                k = "fn_assigned_elsewhere"
            else:
                k = "fn_below_threshold"
            loss[k] += l / n_err
            cnt[k] += 1
            ex[k].append((s, l, o, T))
    n = len(all_val)
    print(f"validation entities {n}  macro F0.5 {tot / n:.5f}  total loss {(n - tot) / n:.5f}  threshold {thr}")
    print(f"{'category':28s} {'loss':>9s} {'share':>7s} {'count':>7s}")
    for k, v in sorted(loss.items(), key=lambda kv: -kv[1]):
        print(f"{k:28s} {v / n:9.5f} {v / max(1e-9, n - tot):7.1%} {cnt[k]:7d}")
    feats = ["p2", "p1", "p1_q_gap", "n_tset", "hn_rel", "hn_logdiff", "num_jacc", "num_q_only", "legal_q_only", "legal_conflict",
             "lex_extra_min", "ex_mod_max", "peer_num_share_p", "anc_p1", "q_addr_empty", "q_native"]
    rng = np.random.RandomState(0)
    for k in ("singleton_given_match", "fp_distractor", "fp_belongs_to_other_s1", "fn_below_threshold", "fn_assigned_elsewhere", "fn_stage1", "fn_blocking"):
        items = ex.get(k, [])
        if not items:
            continue
        print(f"\n===== {k}: {len(items)} cases, {a.examples} examples")
        for it in [items[i] for i in rng.choice(len(items), min(a.examples, len(items)), replace=False)]:
            s = it[0]
            print(f"  S1 {s} {raw.get(s)}   loss {it[1]:.2f}")
            others = it[2] if isinstance(it[2], set) else {it[2]}
            for o in list(others)[:4]:
                p = p2_of.get((s, o))
                row = D[(D.s1_id == s) & (D.q_id == o)]
                fs = " ".join(f"{f}={row[f].values[0]:.2f}" for f in feats if f in row and len(row)) if len(row) else "(not scored)"
                owner = assigned_to.get(o)
                print(f"     {'TRUE ' if o in it[3] else 'FALSE'} {o} {raw.get(o)}  p2={p if p is None else round(float(p), 3)} assigned_to={owner if owner != s else 'self'}")
                if len(row):
                    print(f"          {fs}  extra='{row.extra.values[0]}' miss='{row.miss.values[0]}'")
            for o in sorted(it[3] - others)[:3]:
                print(f"     (true, ok) {o} {raw.get(o)}  p2={round(float(p2_of.get((s, o), -1)), 3)}")


if __name__ == "__main__":
    main()
