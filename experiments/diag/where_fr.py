import os, sys, collections
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
S202 = r"C:\Users\ARNAVG~1\AppData\Local\Temp\claude\C--Users-Arnav-Gawade-pro--OneDrive-Desktop-amazon-ml\5cd43ae6-163c-4444-b6bc-8df70e443c7d\scratchpad\s202\pair_scores.parquet"
T = r"C:\Users\ARNAVG~1\AppData\Local\Temp\claude\C--Users-Arnav-Gawade-pro--OneDrive-Desktop-amazon-ml\89303762-6110-432e-ae87-2d206a5ca21f\scratchpad\sr\student_resource\dataset\test"
d = pd.read_parquet(S202)
best = d.sort_values("p", ascending=False).drop_duplicates("q")
n_rec = collections.Counter(); n1 = collections.Counter()
for k in (1, 2, 3):
    with open(os.path.join(T, f"test_source{k}.tsv"), encoding="utf-8") as f:
        next(f)
        for l in f:
            c = l.rstrip("\n").split("\t")[3]
            (n1 if k == 1 else n_rec)[c] += 1
edges = [0, 0.01, 0.1, 0.3, 0.5, 0.75, 0.9, 0.96, 0.99, 0.999, 1.01]
print("records per S1 by best-candidate probability band (records never scored = 'none')")
print(f"{'country':7s} {'none':>6s} " + " ".join(f"{a:>5}-{b:<5}" for a, b in zip(edges[:-1], edges[1:])))
for c in ("US", "India", "France"):
    b = best[best.country == c].p.values
    h, _ = np.histogram(b, bins=edges)
    none = n_rec[c] - len(b)
    print(f"{c:7s} {none/n1[c]:6.3f} " + " ".join(f"{x/n1[c]:11.3f}" for x in h))
