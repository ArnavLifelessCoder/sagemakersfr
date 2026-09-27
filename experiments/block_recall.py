"""Blocking recall experiments on the dev subset.

Runs the pipeline's blocking with several settings and an extra exact-key channel
(state + number token) and reports true-pair recall, candidates per query and what is still missed.

    python experiments/block_recall.py [--dev DEV_DIR]
"""
import argparse
import collections
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "code", "business_entity_resolution"))
from src.blocking import iter_candidates  # noqa: E402
from src.data import gt_pairs  # noqa: E402
from src.pipeline import prune  # noqa: E402
from src.prep import extend_translit, learn_translit, parse_frame  # noqa: E402

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def number_key_channel(s1, q, max_key_freq=300, min_len=2):
    """Candidates sharing (country, state, number token) with the S1; numbers shorter than min_len skipped.
    Queries without a state use (country, '', number) against S1 of the whole country only for numbers of length >= 4."""
    idx = collections.defaultdict(list)
    for i, (c, st, nums) in enumerate(zip(s1.country.values, s1.state.values, s1.nums.values)):
        for n in nums.split():
            if len(n) >= min_len:
                idx[(c, st, n)].append(i)
                if len(n) >= 4:
                    idx[(c, "", n)].append(i)
    idx = {k: v for k, v in idx.items() if len(v) <= max_key_freq}
    qi, si = [], []
    for j, (c, st, nums) in enumerate(zip(q.country.values, q.state.values, q.nums.values)):
        seen = set()
        for n in nums.split():
            for i in idx.get((c, st, n), ()) if st else idx.get((c, "", n), ()):
                if i not in seen:
                    seen.add(i)
                    qi.append(j)
                    si.append(i)
    return pd.DataFrame({"qi": np.array(qi, dtype=np.int64), "si": np.array(si, dtype=np.int64)})


def evaluate(name, cand, s1_ids, q_ids, truth_pairs, n_q, raw):
    got = set(zip(s1_ids[cand.si.values], q_ids[cand.qi.values]))
    hit = [(s, o) in got for s, o in truth_pairs]
    rec = np.mean(hit)
    log(f"{name:48s} recall {rec:.5f}  missed {len(hit) - sum(hit):5d}  cand/query {len(cand) / n_q:.2f}")
    return [(s, o) for (s, o), h in zip(truth_pairs, hit) if not h]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default=os.path.join(ROOT, "dev"))
    a = ap.parse_args()
    s1 = pd.read_parquet(os.path.join(a.dev, "src1.parquet"))
    q = pd.concat([pd.read_parquet(os.path.join(a.dev, f"src{k}.parquet")) for k in (2, 3)], ignore_index=True)
    s1["src"] = 1
    q["src"] = q.entity_id.str[1].astype(int)
    pairs = gt_pairs(pd.read_parquet(os.path.join(a.dev, "gt.parquet")))
    name_of = dict(zip(pd.concat([s1, q]).entity_id.values, pd.concat([s1, q]).business_name.values))
    tr = learn_translit([name_of.get(x, "") for x in pairs.s1.values], [name_of.get(x, "") for x in pairs.other.values])
    tr = extend_translit(tr, pd.concat([s1, q]).business_name.values)
    s1 = parse_frame(s1, tr)
    q = parse_frame(q, tr)
    raw = {e: (n, ad) for e, n, ad in zip(pd.concat([s1, q]).entity_id.values, pd.concat([s1, q]).business_name.values, pd.concat([s1, q]).business_address.values)}
    log("parsed", len(s1), len(q))
    s1_ids, q_ids = s1.entity_id.values, q.entity_id.values
    truth_pairs = list(zip(pairs.s1.values, pairs.other.values))
    n_q = len(q)

    def run(k, k_nostate, keep):
        parts = [prune(c, keep) for _, c in iter_candidates(s1, q, k_name=k, k_addr=k, k_name_nostate=k_nostate, threads=-1, log=lambda *x: None)]
        return pd.concat(parts, ignore_index=True).drop_duplicates(["qi", "si"])

    base = run(10, 10, 5)
    missed = evaluate("baseline k10 keep5 nostate10", base, s1_ids, q_ids, truth_pairs, n_q, raw)
    empty = {o for s, o in missed if not raw[o][1].strip()}
    log(f"   missed with empty address: {len(empty)}")
    c20 = run(20, 40, 10)
    evaluate("k20 keep10 nostate40", c20, s1_ids, q_ids, truth_pairs, n_q, raw)
    c20b = run(20, 40, 5)
    evaluate("k20 keep5 nostate40", c20b, s1_ids, q_ids, truth_pairs, n_q, raw)
    nk = number_key_channel(s1, q)
    log(f"number-key channel alone: {len(nk)} pairs ({len(nk) / n_q:.2f}/query)")
    evaluate("number-key channel alone", nk, s1_ids, q_ids, truth_pairs, n_q, raw)
    u = pd.concat([base, nk], ignore_index=True).drop_duplicates(["qi", "si"])
    missed2 = evaluate("baseline + number-key", u, s1_ids, q_ids, truth_pairs, n_q, raw)
    u2 = pd.concat([c20b, nk], ignore_index=True).drop_duplicates(["qi", "si"])
    missed3 = evaluate("k20 keep5 nostate40 + number-key", u2, s1_ids, q_ids, truth_pairs, n_q, raw)
    rng = np.random.RandomState(0)
    print("\nstill missed (k20 + number-key), non-empty address, examples:")
    ne = [(s, o) for s, o in missed3 if raw[o][1].strip()]
    for s, o in [ne[i] for i in rng.choice(len(ne), min(12, len(ne)), replace=False)]:
        print(f"  S1 {raw[s]}\n   Q {raw[o]}")
    print(f"\nmissed with empty address at k20/nostate40: {sum(1 for s, o in missed3 if not raw[o][1].strip())}")
    ea = [(s, o) for s, o in missed3 if not raw[o][1].strip()]
    for s, o in [ea[i] for i in rng.choice(len(ea), min(8, len(ea)), replace=False)]:
        print(f"  S1 {raw[s][0]}   Q {raw[o][0]}")


if __name__ == "__main__":
    main()
