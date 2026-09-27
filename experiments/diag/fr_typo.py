"""France pairs with exactly one core word different (same/related house number): typo swap (similar strings) vs
word swap (different word). Density per S1 by band, test France vs labelled reference; plus examples."""
import sys, os, collections, random
import numpy as np, pandas as pd
from rapidfuzz.fuzz import ratio
sys.path.insert(0, r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\code\business_entity_resolution")
sys.stdout.reconfigure(encoding="utf-8")
from two_camp import OLD, T, S202, load, related
BANDS = [(0.3, 0.5), (0.5, 0.75), (0.75, 0.9), (0.9, 0.96), (0.96, 0.99), (0.99, 0.999), (0.999, 1.01)]

def table(sc, info, n1, owner=None, ex=None):
    best = sc.sort_values("p", ascending=False).drop_duplicates("q")
    best = best[best.s1.isin(info.keys()) & best.q.isin(info.keys())]
    cnt, tru = collections.Counter(), collections.Counter()
    for s, q, p in zip(best.s1.values, best.q.values, best.p.values):
        cs, hs = info[s]; cq, hq = info[q]
        if not (len(cs) == len(cq) and len(cs) >= 2 and len(cs - cq) == 1):
            continue
        if hs and hq and not related(hs, hq):
            continue
        a, = cs - cq; b, = cq - cs
        r = ratio(a, b)
        kind = "typo>=80" if r >= 80 else ("near60-80" if r >= 60 else "word<60")
        band = next((x for x in BANDS if x[0] <= p < x[1]), None)
        if band is None:
            continue
        cnt[(kind, band)] += 1
        if owner is not None:
            tru[(kind, band)] += owner.get(q) == s
        if ex is not None and 0.5 <= p < 0.99 and kind == "typo>=80":
            ex.append((s, q, p, a, b))
    return {k: v / n1 for k, v in cnt.items()}, {k: v / n1 for k, v in tru.items()}

if __name__ == "__main__":
    SL = os.path.join(OLD, "eslice")
    gt = pd.read_csv(os.path.join(SL, "gt.tsv"), sep="\t", dtype=str, keep_default_na=False)
    owner = {q: s for s, m in zip(gt.source1_entity_id, gt.matched_entity_ids) for q in m.split(",") if q}
    s1r = pd.read_csv(os.path.join(SL, "test", "test_source1.tsv"), sep="\t", dtype=str, keep_default_na=False,
                      quoting=3, usecols=["entity_id", "country"])
    rc = dict(zip(s1r.entity_id, s1r.country))
    rp = pd.read_parquet(os.path.join(OLD, "eslice_out_ocr", "pair_scores.parquet"))
    rp = rp[rp.p >= 0.3]
    rp = rp.assign(c=rp.s1.map(rc))
    R, RT = collections.Counter(), collections.Counter()
    for c in ("US", "India"):
        sub = rp[rp.c == c]
        info = load(os.path.join(SL, "test"), set(sub.s1) | set(sub.q), c)
        a, t = table(sub, info, 1)
        for k in a: R[k] += a[k]
        for k in t: RT[k] += t[k]
        del info
    n_ref = ((s1r.country == "US") | (s1r.country == "India")).sum()
    del rp
    tp = pd.read_parquet(S202, filters=[("country", "=", "France")], columns=["s1", "q", "p"])
    tp = tp[tp.p >= 0.3]
    info = load(T, set(tp.s1) | set(tp.q), "France")
    ex = []
    t_all, _ = table(tp, info, 259452, ex=ex)
    print(f"{'kind':10s} {'band':14s} {'FR test/S1':>10s} {'ref/S1':>8s} {'ref prec':>8s}")
    for k in sorted(set(t_all) | set(R)):
        r = R.get(k, 0) / n_ref
        print(f"{k[0]:10s} {str(k[1]):14s} {t_all.get(k, 0):10.4f} {r:8.4f} {(RT.get(k, 0) / R[k] if R.get(k) else float('nan')):8.3f}")
    random.seed(0)
    smp = random.sample(ex, min(30, len(ex)))
    ids = {x for e in smp for x in e[:2]}
    rec = {}
    for k in (1, 2, 3):
        with open(os.path.join(T, f"test_source{k}.tsv"), encoding="utf-8") as f:
            next(f)
            for l in f:
                p = l.rstrip("\n").split("\t")
                if p[0] in ids: rec[p[0]] = (p[1], p[2])
    print("\nFrance typo-swap examples in 0.5-0.99:")
    for s, q, p, a, b in smp:
        print(f"p={p:.3f} [{a}->{b}]\n   S1 {rec[s][0]} | {rec[s][1]}\n   q  {rec[q][0]} | {rec[q][1]}")
