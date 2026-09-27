import sys, os, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
src = open("cluster_rule.py", encoding="utf-8").read().split("which = sys.argv[1]")[0]
exec(src)
SL, OUT = os.path.join(H, "eslice"), os.path.join(H, "eslice_out_nb2")
gt = pd.read_csv(os.path.join(SL, "gt.tsv"), sep="\t", dtype=str, keep_default_na=False)
owner = {q: s for s, m in zip(gt.source1_entity_id, gt.matched_entity_ids) for q in m.split(",") if q}
ps = pd.read_parquet(os.path.join(OUT, "pair_scores.parquet")).drop_duplicates(["s1", "q"]).reset_index(drop=True)
hn, ctry = load_hn(set(ps.s1) | set(ps.q), [os.path.join(SL, "test")])
ps["flag"] = flag(ps, hn)
ps["y"] = [owner.get(q) == s for s, q in zip(ps.s1, ps.q)]
ps["c"] = ps.s1.map(ctry)
best = ps.sort_values("p", ascending=False).drop_duplicates("q")
print(best[best.flag].groupby("c").y.agg(["size", "mean"]))
rec = {}
for k in (1, 2, 3):
    t = pd.read_csv(os.path.join(SL, "test", f"test_source{k}.tsv"), sep="\t", dtype=str, keep_default_na=False, quoting=3)
    rec.update(dict(zip(t.entity_id, zip(t.business_name, t.business_address))))
rng = np.random.RandomState(0)
for c in ("US", "India"):
    idx = np.flatnonzero((best.flag & best.y & (best.c == c)).values)
    print(f"\n### {c} flagged TRUE examples (S1 hn vs q hn)")
    for i in rng.choice(idx, min(10, len(idx)), replace=False):
        r = best.iloc[i]
        print(f"  S1 hn={hn[r.s1]!r:8} {rec[r.s1][1][:60]}\n   q hn={hn[r.q]!r:8} {rec[r.q][0][:30]} | {rec[r.q][1][:60]}")
