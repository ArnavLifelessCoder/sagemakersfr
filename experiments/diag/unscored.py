import os, sys, random, collections, re
import pandas as pd
sys.path.insert(0, r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\code\business_entity_resolution")
sys.stdout.reconfigure(encoding="utf-8")
from src.make_dev_subset import state_of
T = r"C:\Users\ARNAVG~1\AppData\Local\Temp\claude\C--Users-Arnav-Gawade-pro--OneDrive-Desktop-amazon-ml\89303762-6110-432e-ae87-2d206a5ca21f\scratchpad\sr\student_resource\dataset\test"
sc = pd.read_parquet("s7/pair_scores.parquet", columns=["q", "country"])
scored = set(sc.q.values); del sc
rows = collections.defaultdict(list); n1 = collections.Counter()
for k in (1, 2, 3):
    with open(os.path.join(T, f"test_source{k}.tsv"), encoding="utf-8") as f:
        next(f)
        for l in f:
            p = l.rstrip("\n").split("\t")
            if k == 1: n1[p[3]] += 1; continue
            if p[0] not in scored: rows[p[3]].append(p)
nonascii = lambda s: any(ord(ch) > 0x2FF for ch in s)
for c in ("India", "US", "France"):
    r = rows[c]
    st = collections.Counter()
    for p in r:
        a = p[2]
        st["empty_addr" if not a.strip() else ("native_addr" if nonascii(a) else "latin_addr")] += 1
        st["native_name" if nonascii(p[1]) else "latin_name"] += 1
        if a.strip(): st["state_found" if state_of(a, c) else "state_missing"] += 1
    print(c, f"unscored/S1 {len(r)/n1[c]:.3f}", {k: round(v / n1[c], 3) for k, v in st.items()})
random.seed(1)
for p in random.sample(rows["India"], 30):
    print("  ", p[0], "|", p[1], "|", p[2])
