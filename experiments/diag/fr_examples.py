import os, sys, random
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
S202 = r"C:\Users\ARNAVG~1\AppData\Local\Temp\claude\C--Users-Arnav-Gawade-pro--OneDrive-Desktop-amazon-ml\5cd43ae6-163c-4444-b6bc-8df70e443c7d\scratchpad\s202\pair_scores.parquet"
T = r"C:\Users\ARNAVG~1\AppData\Local\Temp\claude\C--Users-Arnav-Gawade-pro--OneDrive-Desktop-amazon-ml\89303762-6110-432e-ae87-2d206a5ca21f\scratchpad\sr\student_resource\dataset\test"
d = pd.read_parquet(S202, filters=[("country", "=", "France")])
best = d.sort_values("p", ascending=False).drop_duplicates("q")
random.seed(3)
pick = {}
for lo, hi in [(0.01, 0.1), (0.1, 0.5)]:
    b = best[(best.p >= lo) & (best.p < hi)]
    pick[(lo, hi)] = b.sample(22, random_state=5)
ids = set()
for b in pick.values():
    ids |= set(b.s1) | set(b.q)
rec = {}
for k in (1, 2, 3):
    with open(os.path.join(T, f"test_source{k}.tsv"), encoding="utf-8") as f:
        next(f)
        for l in f:
            p = l.rstrip("\n").split("\t")
            if p[0] in ids:
                rec[p[0]] = (p[1], p[2])
for band, b in pick.items():
    print("\n=== band", band)
    for s, q, p in zip(b.s1, b.q, b.p):
        print(f"p={p:.3f}\n   S1 {rec[s][0]} | {rec[s][1]}\n   q  {rec[q][0]} | {rec[q][1]}")
