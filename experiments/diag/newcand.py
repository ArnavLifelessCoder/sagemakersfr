import os, sys, random
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
T = r"C:\Users\ARNAVG~1\AppData\Local\Temp\claude\C--Users-Arnav-Gawade-pro--OneDrive-Desktop-amazon-ml\89303762-6110-432e-ae87-2d206a5ca21f\scratchpad\sr\student_resource\dataset\test"
m8 = pd.read_csv("s7/matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
assigned8 = {q for x in m8.iloc[:, 1] for q in x.split(",") if q}
m9 = pd.read_csv("out_v9/matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
p9pairs = [(s, q) for s, x in zip(m9.iloc[:, 0], m9.iloc[:, 1]) for q in x.split(",") if q]
sc8 = pd.read_parquet("s7/pair_scores.parquet", columns=["s1", "q"], filters=[("country", "=", "France")])
cand8 = set(zip(sc8.s1, sc8.q)); del sc8
sc9 = pd.read_parquet("out_v9/pair_scores.parquet"); p9 = dict(zip(zip(sc9.s1, sc9.q), sc9.p)); del sc9
new = [(s, q) for s, q in p9pairs if (s, q) not in cand8 and q not in assigned8]
print("v9 accepted France pairs that were NOT v8 candidates and whose record v8 left unassigned:", len(new), f"= {len(new)/259452:.4f}/S1")
for lo in (0.75, 0.9, 0.99):
    print(f"   with p9 >= {lo}: {sum(1 for x in new if p9[x] >= lo)}")
random.seed(7)
smp = random.sample(new, min(24, len(new)))
ids = {x for p in smp for x in p}
rec = {}
for k in (1, 2, 3):
    with open(os.path.join(T, f"test_source{k}.tsv"), encoding="utf-8") as f:
        next(f)
        for l in f:
            p = l.rstrip("\n").split("\t")
            if p[0] in ids: rec[p[0]] = (p[1], p[2])
for s, q in smp:
    print(f"p9={p9[(s, q)]:.3f}\n   S1 {rec[s][0]} | {rec[s][1]}\n   q  {rec[q][0]} | {rec[q][1]}")
pd.DataFrame(new, columns=["s1", "q"]).assign(p=[p9[x] for x in new]).to_parquet("v9_newcand.parquet")
