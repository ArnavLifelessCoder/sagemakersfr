"""Label-free blocking-recall proxy: 'easy' true pairs = S2/S3 record with the same sorted core name, the same
house number and an overlapping street token as an S1 (same country). What share are in the candidate set
and what share are finally matched?  usage: easy_recall.py COUNTRY"""
import sys, os, collections
import pandas as pd
sys.path.insert(0, r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\code\business_entity_resolution")
sys.stdout.reconfigure(encoding="utf-8")
from src.normalize import parse_name, parse_address

H = os.path.dirname(os.path.abspath(__file__))
T = r"C:\Users\ARNAVG~1\AppData\Local\Temp\claude\C--Users-Arnav-Gawade-pro--OneDrive-Desktop-amazon-ml\89303762-6110-432e-ae87-2d206a5ca21f\scratchpad\sr\student_resource\dataset\test"
C = sys.argv[1]
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 9
STT = {"rue", "st", "ave", "rd", "dr", "blvd", "ln", "all", "ch", "imp", "pl", "ct", "cir", "trl", "pkwy", "hwy"}


def key(name, addr):
    n = parse_name(name)
    a = parse_address(addr, C)
    if not n["core"] or not a["hn"]:
        return None
    st = frozenset(set(a["street"]) - STT)
    return (" ".join(sorted(n["core"])), a["hn"]), st


s1key = collections.defaultdict(list)
n1 = 0
with open(os.path.join(T, "test_source1.tsv"), encoding="utf-8") as f:
    next(f)
    for l in f:
        p = l.rstrip("\n").split("\t")
        if p[3] != C:
            continue
        n1 += 1
        if n1 > LIMIT:
            break
        k = key(p[1], p[2])
        if k:
            s1key[k[0]].append((p[0], k[1]))
easy = []
for k in (2, 3):
    with open(os.path.join(T, f"test_source{k}.tsv"), encoding="utf-8") as f:
        next(f)
        for l in f:
            p = l.rstrip("\n").split("\t")
            if p[3] != C or not p[2]:
                continue
            kk = key(p[1], p[2])
            if not kk or kk[0] not in s1key:
                continue
            cands = [s for s, st in s1key[kk[0]] if st & kk[1]]
            if len(cands) == 1:
                easy.append((cands[0], p[0]))
ps = pd.read_parquet(os.path.join(H, "v5s102", "pair_scores.parquet"), columns=["s1", "q", "p", "country"],
                     filters=[("country", "=", C)])
cand = set(zip(ps.s1, ps.q))
best = ps.sort_values("p", ascending=False).drop_duplicates("q")
bmap = dict(zip(best.q, zip(best.s1, best.p)))
inc = sum((s, q) in cand for s, q in easy)
won = sum(bmap.get(q, (None, 0))[0] == s for s, q in easy)
acc = sum(bmap.get(q, (None, 0))[0] == s and bmap[q][1] >= 0.75 for s, q in easy)
print(f"{C}: S1 scanned {min(n1, LIMIT)}, easy pairs {len(easy)} | in candidates {inc / len(easy):.4f} | "
      f"best S1 is right {won / len(easy):.4f} | accepted (p>=.75) {acc / len(easy):.4f}")
