"""Unassigned S2/S3 records that have an S1 with identical core name AND same house number AND a shared street
word (or, if the S1 has no number, same street words). Count per country (v8 output) + samples."""
import os, sys, random, collections
from multiprocessing import Pool
import pandas as pd
sys.path.insert(0, r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\code\business_entity_resolution")
sys.stdout.reconfigure(encoding="utf-8")
from src.normalize import parse_name, parse_address
T = r"C:\Users\ARNAVG~1\AppData\Local\Temp\claude\C--Users-Arnav-Gawade-pro--OneDrive-Desktop-amazon-ml\89303762-6110-432e-ae87-2d206a5ca21f\scratchpad\sr\student_resource\dataset\test"
TYPES = {"rue", "st", "ave", "rd", "dr", "blvd", "ln", "all", "ch", "imp", "pl", "ct", "cir", "trl", "pkwy", "hwy", "ter",
         "sq", "rte", "way", "road", "street", "avenue", "de", "du", "des", "la", "le", "les", "d", "l"}

def key_of(n):
    return " ".join(sorted(parse_name(n)["core"]))

def keys(rows):
    return [(p, key_of(p[1])) for p in rows]

def addr(rows):
    out = []
    for p in rows:
        a = parse_address(p[2], p[3]) if p[2].strip() else None
        out.append((p[0], (a["hn"], frozenset(a["street"]) - TYPES) if a else ("", frozenset())))
    return out

def chunks(x, n=40000):
    return [x[i:i + n] for i in range(0, len(x), n)]

if __name__ == "__main__":
    m = pd.read_csv("s7/matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
    assigned = {q for x in m.iloc[:, 1] for q in x.split(",") if q}
    s1rows, qrows = [], []
    for k in (1, 2, 3):
        with open(os.path.join(T, f"test_source{k}.tsv"), encoding="utf-8") as f:
            next(f)
            for l in f:
                p = l.rstrip("\n").split("\t")
                if k == 1: s1rows.append(p)
                elif p[0] not in assigned: qrows.append(p)
    with Pool(6) as pool:
        s1k = [x for part in pool.map(keys, chunks(s1rows)) for x in part]
        idx = collections.defaultdict(list)
        for p, key in s1k:
            if key: idx[(p[3], key)].append(p)
        qk = [x for part in pool.map(keys, chunks(qrows)) for x in part]
        qk = [(p, key) for p, key in qk if key and (p[3], key) in idx and p[2].strip()]
        need_s1 = {s[0]: s for p, key in qk for s in idx[(p[3], key)]}
        s1a = dict(x for part in pool.map(addr, chunks(list(need_s1.values()))) for x in part)
        qa = dict(x for part in pool.map(addr, chunks([p for p, _ in qk])) for x in part)
    n1 = collections.Counter(p[3] for p in s1rows)
    hits = collections.defaultdict(list)
    for p, key in qk:
        hq, sq = qa[p[0]]
        cands = []
        for s in idx[(p[3], key)]:
            hs, ss = s1a[s[0]]
            if hq and hs and hq == hs and (sq & ss):
                cands.append(s)
            elif not hq and not hs and len(sq & ss) >= 2:
                cands.append(s)
        if len(cands) == 1:
            hits[p[3]].append((p, cands[0]))
    for c in ("US", "India", "France"):
        print(f"{c:7s} unassigned exact twins (name + number + street): {len(hits[c])} = {len(hits[c]) / n1[c]:.4f}/S1")
    random.seed(0)
    for c in ("France", "US", "India"):
        print("==", c)
        for p, s in random.sample(hits[c], min(8, len(hits[c]))):
            print(f"   q  {p[1]} | {p[2]}\n   S1 {s[1]} | {s[2]}")
    pd.DataFrame([(s[0], p[0], c) for c in hits for p, s in hits[c]], columns=["s1", "q", "country"]).to_parquet("twins.parquet")
