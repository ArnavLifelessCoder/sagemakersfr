"""Transitive rescue: a rejected candidate q of S1 s is accepted if its normalised name (and address, when it has
one) equals that of a record ALREADY accepted for s. Also resolves empty-address records whose name fits
several S1s: they go to the S1 that already holds an accepted record with the same name.
Evaluated on the labelled slice (F0.5 before/after, precision of rescued pairs)."""
import sys, os, collections
import numpy as np, pandas as pd
sys.path.insert(0, r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\code\business_entity_resolution")
sys.stdout.reconfigure(encoding="utf-8")
from src.normalize import parse_name, parse_address

H = os.path.dirname(os.path.abspath(__file__))
SL = os.path.join(H, "eslice")
OUT = os.path.join(H, sys.argv[1] if len(sys.argv) > 1 else "eslice_out_ocr")
THR = 0.65


def f05(P, T):
    if not T and not P:
        return 1.0
    if not T or not P:
        return 0.0
    tp = len(P & T)
    if not tp:
        return 0.0
    pr, rc = tp / len(P), tp / len(T)
    return 1.25 * pr * rc / (0.25 * pr + rc)


gt = pd.read_csv(os.path.join(SL, "gt.tsv"), sep="\t", dtype=str, keep_default_na=False)
truth = {s: set(x for x in m.split(",") if x) for s, m in zip(gt.source1_entity_id, gt.matched_entity_ids)}
owner = {q: s for s, v in truth.items() for q in v}
s1ids = pd.read_csv(os.path.join(SL, "test", "test_source1.tsv"), sep="\t", dtype=str, keep_default_na=False,
                    quoting=3, usecols=["entity_id"]).entity_id.values
ps = pd.read_parquet(os.path.join(OUT, "pair_scores.parquet")).drop_duplicates(["s1", "q"])
best = ps.sort_values("p", ascending=False).drop_duplicates("q")
acc = best[best.p >= THR]
pred = collections.defaultdict(set)
for s, q in zip(acc.s1, acc.q):
    pred[s].add(q)
need = set(ps.q)
rec = {}
for k in (2, 3):
    t = pd.read_csv(os.path.join(SL, "test", f"test_source{k}.tsv"), sep="\t", dtype=str, keep_default_na=False, quoting=3)
    t = t[t.entity_id.isin(need)]
    rec.update(dict(zip(t.entity_id, zip(t.business_name, t.business_address, t.country))))
key = {}
for q, (n, a, c) in rec.items():
    nm = " ".join(sorted(parse_name(n)["core"]))
    pa = parse_address(a, c) if a else None
    key[q] = (nm, (pa["hn"], " ".join(pa["street"])) if pa and not pa["empty"] else None)
# names / (name,address) of accepted records per S1
acc_names = collections.defaultdict(set)
acc_full = collections.defaultdict(set)
for s, qs in pred.items():
    for q in qs:
        nm, ad = key[q]
        if nm:
            acc_names[s].add(nm)
            if ad:
                acc_full[s].add((nm, ad))
# candidates per rejected q: all S1 it was paired with
cand_of = collections.defaultdict(list)
for s, q, p in zip(ps.s1, ps.q, ps.p):
    cand_of[q].append((s, p))
accepted_q = set(acc.q)
resc = []
for q, lst in cand_of.items():
    if q in accepted_q:
        continue
    nm, ad = key[q]
    if not nm:
        continue
    hits = []
    for s, p in lst:
        if p < 0.02:
            continue
        if ad is None:
            if nm in acc_names.get(s, ()):
                hits.append(s)
        elif (nm, ad) in acc_full.get(s, ()):
            hits.append(s)
    if len(hits) == 1:
        resc.append((hits[0], q, ad is None))
lab = np.array([owner.get(q) == s for s, q, _ in resc])
emp = np.array([e for _, _, e in resc])
print(f"rescued {len(resc)}  precision {lab.mean():.4f} | empty-address {emp.sum()} prec {lab[emp].mean():.4f} | "
      f"with address {(~emp).sum()} prec {lab[~emp].mean():.4f}")
base = np.mean([f05(pred.get(s, set()), truth.get(s, set())) for s in s1ids])
new = {s: set(v) for s, v in pred.items()}
for s, q, _ in resc:
    new.setdefault(s, set()).add(q)
after = np.mean([f05(new.get(s, set()), truth.get(s, set())) for s in s1ids])
print(f"macro F0.5 before {base:.5f}  after {after:.5f}  gain {after - base:+.5f}")
