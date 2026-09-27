import os, sys, random
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
T = r"C:\Users\ARNAVG~1\AppData\Local\Temp\claude\C--Users-Arnav-Gawade-pro--OneDrive-Desktop-amazon-ml\89303762-6110-432e-ae87-2d206a5ca21f\scratchpad\sr\student_resource\dataset\test"
s1c = pd.read_csv(os.path.join(T, "test_source1.tsv"), sep="\t", dtype=str, keep_default_na=False, quoting=3, usecols=["entity_id", "country"])
fr = set(s1c.entity_id[s1c.country == "France"])
def pairs(p):
    m = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)
    return {(s, q) for s, x in zip(m.iloc[:, 0], m.iloc[:, 1]) if s in fr for q in x.split(",") if q}
a, b = pairs("s7/matching_results.tsv"), pairs("out_v9/matching_results.tsv")
print("France pairs v8", len(a), "v9", len(b), "common", len(a & b), "only v8", len(a - b), "only v9", len(b - a))
sc = pd.read_parquet("out_v9/pair_scores.parquet"); p9 = dict(zip(zip(sc.s1, sc.q), sc.p)); del sc
sc = pd.read_parquet("s7/pair_scores.parquet", filters=[("country", "=", "France")]); p8 = dict(zip(zip(sc.s1, sc.q), sc.p)); del sc
random.seed(4)
ov8, ov9 = random.sample(sorted(a - b), 14), random.sample(sorted(b - a), 14)
ids = {x for p in ov8 + ov9 for x in p}
rec = {}
for k in (1, 2, 3):
    with open(os.path.join(T, f"test_source{k}.tsv"), encoding="utf-8") as f:
        next(f)
        for l in f:
            p = l.rstrip("\n").split("\t")
            if p[0] in ids: rec[p[0]] = (p[1], p[2])
for title, lst in (("ONLY IN v8 (v9 dropped)", ov8), ("ONLY IN v9 (v9 added)", ov9)):
    print("\n==", title)
    for s, q in lst:
        print(f"p8={p8.get((s, q), float('nan')):.3f} p9={p9.get((s, q), float('nan')):.3f}\n   S1 {rec[s][0]} | {rec[s][1]}\n   q  {rec[q][0]} | {rec[q][1]}")
