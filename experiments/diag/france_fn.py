"""France: what does the v4 ensemble reject? p distribution of best candidates + samples by band."""
import sys, os, random, collections
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.stdout.reconfigure(encoding="utf-8")
H = os.path.dirname(os.path.abspath(__file__))
T = r"C:\Users\ARNAVG~1\AppData\Local\Temp\claude\C--Users-Arnav-Gawade-pro--OneDrive-Desktop-amazon-ml\89303762-6110-432e-ae87-2d206a5ca21f\scratchpad\sr\student_resource\dataset\test"
C = sys.argv[1] if len(sys.argv) > 1 else "France"
parts = []
for s in (0, 2, 3):
    d = pq.read_table(os.path.join(H, f"v4s{s}", "pair_scores.parquet"), columns=["s1", "q", "p"],
                      filters=[("country", "=", C)]).to_pandas()
    parts.append(d.set_index(["s1", "q"]).p.rename(f"p{s}"))
e = pd.concat(parts, axis=1).fillna(0)
e["p"] = e.mean(axis=1)
e = e.reset_index()
best = e.sort_values("p", ascending=False).drop_duplicates("q")
bins = [0, .05, .2, .4, .6, .775, .9, 1.01]
print(C, "best-candidate p histogram:", np.histogram(best.p, bins)[0].tolist(), "bins", bins)
rec = {}
need = set()
random.seed(1)
samples = {}
for lo, hi in ((.2, .5), (.5, .775), (.05, .2)):
    b = best[(best.p >= lo) & (best.p < hi)]
    samples[(lo, hi)] = b.sample(min(18, len(b)), random_state=1)
    need |= set(samples[(lo, hi)].s1) | set(samples[(lo, hi)].q)
# accepted pairs of the same S1 for context
acc = best[best.p >= .775]
ctx = acc[acc.s1.isin({s for v in samples.values() for s in v.s1})]
need |= set(ctx.q)
for k in (1, 2, 3):
    with open(os.path.join(T, f"test_source{k}.tsv"), encoding="utf-8") as f:
        next(f)
        for l in f:
            p = l.rstrip("\n").split("\t")
            if p[0] in need:
                rec[p[0]] = (p[1], p[2])
for key, v in samples.items():
    print(f"\n######## rejected band p in [{key[0]}, {key[1]})  (n={((best.p >= key[0]) & (best.p < key[1])).sum()})")
    for s, q, p in zip(v.s1, v.q, v.p):
        kept = [rec[x][0] + " | " + rec[x][1][:45] for x in ctx[ctx.s1 == s].q][:3]
        print(f"p={p:.2f}  S1 {rec[s][0]} | {rec[s][1][:60]}\n        Q  {rec[q][0]} | {rec[q][1][:60]}\n        accepted peers: {kept}")
