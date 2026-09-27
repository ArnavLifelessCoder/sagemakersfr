import sys, os, collections, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
H = os.path.dirname(os.path.abspath(__file__))
SL, OUT = os.path.join(H, "eslice"), os.path.join(H, "eslice_out_k")
gt = pd.read_csv(os.path.join(SL, "gt.tsv"), sep="\t", dtype=str, keep_default_na=False)
owner = {q: s for s, m in zip(gt.source1_entity_id, gt.matched_entity_ids) for q in m.split(",") if q}
rec = {}
for k in (1, 2, 3):
    t = pd.read_csv(os.path.join(SL, "test", f"test_source{k}.tsv"), sep="\t", dtype=str, keep_default_na=False, quoting=3)
    rec.update(dict(zip(t.entity_id, zip(t.business_name, t.business_address, t.country))))
ps = pd.read_parquet(os.path.join(OUT, "pair_scores.parquet"))
best = ps.sort_values("p", ascending=False).drop_duplicates("q")
bq = dict(zip(best.q, zip(best.s1, best.p)))
pp = dict(zip(zip(ps.s1, ps.q), ps.p))
cat = collections.Counter()
rows = []
for q, s in owner.items():
    if (s, q) not in pp:
        continue  # blocking miss
    bs, bp = bq[q]
    if bs == s and bp >= 0.65:
        continue
    kind = "lost to other S1" if bs != s else "scored < 0.65"
    c = rec[s][2]
    cat[(c, kind)] += 1
    rows.append((c, kind, pp[(s, q)], s, q, bs))
print(cat)
rng = np.random.RandomState(0)
for c in ("US", "India"):
    for kind in ("scored < 0.65", "lost to other S1"):
        sel = [r for r in rows if r[0] == c and r[1] == kind]
        print(f"\n##### {c}: {kind} ({len(sel)})")
        for i in rng.choice(len(sel), min(12, len(sel)), replace=False):
            _, _, p, s, q, bs = sel[i]
            extra = f"  | stolen by: {rec[bs][0]} | {rec[bs][1][:40]}" if bs != s else ""
            print(f"p={p:.2f} S1 {rec[s][0]} | {rec[s][1][:55]}\n       Q  {rec[q][0]} | {rec[q][1][:55]}{extra}")
