"""Inspect India pairs removed by v2 calibration (0.75 <= p < 0.9707) and kept ones just above."""
import sys, random
import pyarrow.parquet as pq, pyarrow.compute as pc, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
HERE = __file__.rsplit("\\", 1)[0]
D = r"C:\Users\ARNAVG~1\AppData\Local\Temp\claude\C--Users-Arnav-Gawade-pro--OneDrive-Desktop-amazon-ml\89303762-6110-432e-ae87-2d206a5ca21f\scratchpad\sr\student_resource\dataset\test"
t = pq.read_table(HERE + r"\v2\pair_scores.parquet")
t = t.filter(pc.equal(t["country"], "India")).to_pandas()
best = t.sort_values("p", ascending=False).drop_duplicates("q")
bands = {"removed .75-.97": best[(best.p >= 0.75) & (best.p < 0.9707)],
         "kept .97-.99": best[(best.p >= 0.9707) & (best.p < 0.99)],
         "kept >=.99": best[best.p >= 0.99]}
samp = {k: v.sample(14, random_state=1) for k, v in bands.items()}
need = set()
for v in samp.values():
    need |= set(v.s1) | set(v.q)
rec = {}
for k in (1, 2, 3):
    with open(rf"{D}\test_source{k}.tsv", encoding="utf-8") as f:
        next(f)
        for l in f:
            p = l.rstrip("\n").split("\t")
            if p[0] in need:
                rec[p[0]] = (p[1], p[2])
for k, v in samp.items():
    print(f"\n######## {k}  (n={len(bands[k])})")
    for s, q, p in zip(v.s1, v.q, v.p):
        print(f"p={p:.3f}\n   S1 {rec[s]}\n   Q  {rec[q]}")
