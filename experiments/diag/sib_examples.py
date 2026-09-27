import sys, os, collections, random
import pandas as pd
sys.path.insert(0, r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\code\business_entity_resolution")
sys.stdout.reconfigure(encoding="utf-8")
from src.normalize import parse_name, parse_address
from sib_clusters import T, S202, related
tp = pd.read_parquet(S202, filters=[("country", "=", "US")], columns=["s1", "q", "p"])
tp = tp[tp.p >= 0.3]
best = tp.sort_values("p", ascending=False).drop_duplicates("q")
best = best[(best.p >= 0.5) & (best.p < 0.99)]
random.seed(1)
s_pick = set(random.sample(sorted(set(best.s1)), 20000))
sub = tp[tp.s1.isin(s_pick)]
ids = set(sub.s1) | set(sub.q)
rec = {}
for k in (1, 2, 3):
    with open(os.path.join(T, f"test_source{k}.tsv"), encoding="utf-8") as f:
        next(f)
        for l in f:
            p = l.rstrip("\n").split("\t")
            if p[0] in ids:
                rec[p[0]] = (p[1], p[2])
def info(e):
    n, a = rec[e]; pn = parse_name(n)
    return frozenset(pn["core"]), frozenset(pn["legal"]), parse_address(a, "US")["hn"] if a else ""
bys = collections.defaultdict(list)
for s, q, p in zip(sub.s1, sub.q, sub.p): bys[s].append((q, p))
shown = 0
bb = best[best.s1.isin(s_pick)]
for s, q, p in zip(bb.s1, bb.q, bb.p):
    cs, ls, hs = info(s); cq, lq, hq = info(q)
    if not (hs and hq) or related(hs, hq) or cs != cq or not cs or ls == lq: continue
    peers = [(q2, p2) for q2, p2 in bys[s] if q2 != q and info(q2)[0] == cq and info(q2)[2] and related(info(q2)[2], hq) and not related(info(q2)[2], hs)]
    if not peers: continue
    print(f"S1 {s:12s} | {rec[s][0]} | {rec[s][1]}")
    print(f"  q {q:12s} p={p:.3f} | {rec[q][0]} | {rec[q][1]}")
    for q2, p2 in peers: print(f"  peer {q2:9s} p={p2:.3f} | {rec[q2][0]} | {rec[q2][1]}")
    others = [(x, px) for x, px in bys[s] if x != q and x not in {y for y, _ in peers} and px >= 0.5]
    for x, px in others[:4]: print(f"  other {x:8s} p={px:.3f} | {rec[x][0]} | {rec[x][1]}")
    shown += 1
    if shown >= 14: break
