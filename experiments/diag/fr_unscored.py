import os, sys, random, collections, re, unicodedata
import pandas as pd
sys.path.insert(0, r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\code\business_entity_resolution")
sys.stdout.reconfigure(encoding="utf-8")
from src.make_dev_subset import state_of
from src.normalize import parse_name
T = r"C:\Users\ARNAVG~1\AppData\Local\Temp\claude\C--Users-Arnav-Gawade-pro--OneDrive-Desktop-amazon-ml\89303762-6110-432e-ae87-2d206a5ca21f\scratchpad\sr\student_resource\dataset\test"
sc = pd.read_parquet("s7/pair_scores.parquet", columns=["q", "country"], filters=[("country", "in", ["France", "US"])])
scored = set(sc.q.values); del sc
s1name = collections.defaultdict(set)   # country -> set of core-name keys
s1_by_key = collections.defaultdict(list)
un = collections.defaultdict(list)
for k in (1, 2, 3):
    with open(os.path.join(T, f"test_source{k}.tsv"), encoding="utf-8") as f:
        next(f)
        for l in f:
            p = l.rstrip("\n").split("\t")
            if p[3] not in ("France", "US"): continue
            key = " ".join(sorted(parse_name(p[1])["core"]))
            if k == 1:
                s1name[p[3]].add(key); s1_by_key[(p[3], key)].append(p)
            elif p[0] not in scored:
                un[p[3]].append((p, key))
for c in ("France", "US"):
    tot = collections.Counter(); hit = collections.Counter()
    for p, key in un[c]:
        g = "empty" if not p[2].strip() else ("state" if state_of(p[2], c) else "nostate")
        tot[g] += 1; hit[g] += key in s1name[c]
    print(c, {g: f"{tot[g]} unscored, {hit[g]} ({hit[g]/max(1,tot[g]):.2f}) with an S1 of identical core name" for g in tot})
random.seed(2)
cand = [(p, key) for p, key in un["France"] if p[2].strip() and not state_of(p[2], "France") and key in s1name["France"]]
for p, key in random.sample(cand, min(15, len(cand))):
    print("  q ", p[1], "|", p[2])
    for s in s1_by_key[("France", key)][:2]: print("     S1", s[1], "|", s[2])
