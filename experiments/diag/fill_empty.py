"""'Never leave an S1 empty if its best own candidate has p >= x': labelled evaluation + France test look."""
import sys, os, collections
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
H = os.path.dirname(os.path.abspath(__file__))
T = r"C:\Users\ARNAVG~1\AppData\Local\Temp\claude\C--Users-Arnav-Gawade-pro--OneDrive-Desktop-amazon-ml\89303762-6110-432e-ae87-2d206a5ca21f\scratchpad\sr\student_resource\dataset\test"


def f05(P, Tr):
    if not Tr and not P:
        return 1.0
    if not Tr or not P:
        return 0.0
    tp = len(P & Tr)
    if not tp:
        return 0.0
    pr, rc = tp / len(P), tp / len(Tr)
    return 1.25 * pr * rc / (0.25 * pr + rc)


def decide(best, thr_of, fill):
    """best: argmax pairs (s1,q,p,country). accept p>=thr; then for S1 with nothing accepted, add its top q if p>=fill."""
    acc = best[best.p.values >= best.country.map(thr_of).values]
    pred = collections.defaultdict(set)
    for s, q in zip(acc.s1, acc.q):
        pred[s].add(q)
    if fill is not None:
        top = best.sort_values("p", ascending=False).drop_duplicates("s1")
        for s, q, p in zip(top.s1, top.q, top.p):
            if s not in pred and p >= fill:
                pred[s].add(q)
    return pred


# ---- labelled train-like slice
SL, OUT = os.path.join(H, "eslice"), os.path.join(H, "eslice_out_nb2")
gt = pd.read_csv(os.path.join(SL, "gt.tsv"), sep="\t", dtype=str, keep_default_na=False)
truth = {s: set(x for x in m.split(",") if x) for s, m in zip(gt.source1_entity_id, gt.matched_entity_ids)}
s1 = pd.read_csv(os.path.join(SL, "test", "test_source1.tsv"), sep="\t", dtype=str, keep_default_na=False, quoting=3,
                 usecols=["entity_id", "country"])
ctry = dict(zip(s1.entity_id, s1.country))
ps = pd.read_parquet(os.path.join(OUT, "pair_scores.parquet"))
best = ps.sort_values("p", ascending=False).drop_duplicates("q").copy()
best["country"] = best.s1.map(ctry)
thr = {"US": 0.65, "India": 0.65}
for fill in (None, 0.5, 0.3, 0.2, 0.1):
    pred = decide(best, thr, fill)
    sc = np.mean([f05(pred.get(s, set()), truth.get(s, set())) for s in s1.entity_id])
    print(f"train-like slice: fill={fill}: macro F0.5 {sc:.5f}  empty S1 {1 - len(pred) / len(s1):.4f}")

# ---- France test (v5 seed 102 scores, v6 thresholds)
d = pd.read_parquet(os.path.join(H, "v5s102", "pair_scores.parquet"))
b = d.sort_values("p", ascending=False).drop_duplicates("q")
fr = b[b.country == "France"]
acc = fr[fr.p >= 0.99]
top = fr.sort_values("p", ascending=False).drop_duplicates("s1")
empty_top = top[~top.s1.isin(set(acc.s1))]
print("\nFrance S1 without a p>=0.99 match but with a best own candidate, by p band:",
      np.histogram(empty_top.p, [0, .1, .2, .3, .5, .75, .9, .99])[0].tolist())
samp = empty_top[(empty_top.p >= 0.3) & (empty_top.p < 0.99)].sample(20, random_state=3)
need = set(samp.s1) | set(samp.q)
rec = {}
for k in (1, 2, 3):
    with open(os.path.join(T, f"test_source{k}.tsv"), encoding="utf-8") as f:
        next(f)
        for l in f:
            p = l.rstrip("\n").split("\t")
            if p[0] in need:
                rec[p[0]] = (p[1], p[2][:60])
for s, q, p in zip(samp.s1, samp.q, samp.p):
    print(f"p={p:.2f}  S1 {rec[s]}\n        Q  {rec[q]}")
