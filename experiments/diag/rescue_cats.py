"""Structural categories (rename / initials / empty-address exact name): label rate on a labelled
slice, and how many France test best-candidates in the rejected band fall into them."""
import sys, os, collections
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\code\business_entity_resolution")
sys.stdout.reconfigure(encoding="utf-8")
from src.normalize import parse_name, parse_address
H = os.path.dirname(os.path.abspath(__file__))
T = r"C:\Users\ARNAVG~1\AppData\Local\Temp\claude\C--Users-Arnav-Gawade-pro--OneDrive-Desktop-amazon-ml\89303762-6110-432e-ae87-2d206a5ca21f\scratchpad\sr\student_resource\dataset\test"


def cats(sn, sa, qn, qa, c):
    a, b = parse_name(sn), parse_name(qn)
    A, B = set(a["core"]), set(b["core"])
    pa, pb = parse_address(sa, c), parse_address(qa, c)
    out = []
    same_hn = pa["hn"] and pa["hn"] == pb["hn"]
    street_ov = bool(set(pa["street"]) & set(pb["street"]) - {"rue", "st", "ave", "rd", "dr", "blvd", "ln", "all", "ch", "imp", "pl"})
    if B and not (A & B) and not b["domain"] and same_hn and street_ov:
        ini = "".join(t[0] for t in a["core"])
        out.append("initials" if len(B) == 1 and list(B)[0] == ini else "rename@exact-address")
    if pb["empty"] and B and a["core"] and sorted(a["core"]) == sorted(b["core"]):
        out.append("empty-address exact name")
    return out


# ---- labelled slice (train states TG+AP+OH) with the dev v4 model scores
SL = os.path.join(H, "eslice")
gt = pd.read_csv(os.path.join(SL, "gt.tsv"), sep="\t", dtype=str, keep_default_na=False)
owner = {q: s for s, m in zip(gt.source1_entity_id, gt.matched_entity_ids) for q in m.split(",") if q}
rec = {}
for k in (1, 2, 3):
    d = pd.read_csv(os.path.join(SL, "test", f"test_source{k}.tsv"), sep="\t", dtype=str, keep_default_na=False, quoting=3)
    rec.update(dict(zip(d.entity_id, zip(d.business_name, d.business_address, d.country))))
ps = pd.read_parquet(os.path.join(H, "eslice_out", "pair_scores.parquet"))
best = ps.sort_values("p", ascending=False).drop_duplicates("q")
stat = collections.defaultdict(lambda: [0, 0, []])
for s, q, p in zip(best.s1.values, best.q.values, best.p.values):
    sn, sa, c = rec[s]
    qn, qa, _ = rec[q]
    for k in cats(sn, sa, qn, qa, c):
        v = stat[k]
        v[0] += 1
        v[1] += owner.get(q) == s
        v[2].append(p)
print("LABELLED slice (US-OH, IN-TG/AP): category -> best-candidate pairs, share TRUE, share with p>=0.65")
for k, (n, t, ps_) in stat.items():
    ps_ = np.array(ps_)
    print(f"  {k:26s} n={n:6d}  true={t / n:.4f}  accepted(p>=.65)={np.mean(ps_ >= .65):.3f}")

# ---- France test, v4 3-seed ensemble
parts = []
for sd in (0, 2, 3):
    d = pq.read_table(os.path.join(H, f"v4s{sd}", "pair_scores.parquet"), columns=["s1", "q", "p"],
                      filters=[("country", "=", "France")]).to_pandas()
    parts.append(d.set_index(["s1", "q"]).p.rename(f"p{sd}"))
e = pd.concat(parts, axis=1).fillna(0)
e["p"] = e.mean(axis=1)
e = e.reset_index()
fb = e.sort_values("p", ascending=False).drop_duplicates("q")
rej = fb[(fb.p >= 0.05) & (fb.p < 0.775)]
need = set(rej.s1) | set(rej.q)
frec = {}
for k in (1, 2, 3):
    with open(os.path.join(T, f"test_source{k}.tsv"), encoding="utf-8") as f:
        next(f)
        for l in f:
            p = l.rstrip("\n").split("\t")
            if p[0] in need:
                frec[p[0]] = (p[1], p[2], p[3])
cnt = collections.Counter()
for s, q in zip(rej.s1.values, rej.q.values):
    for k in cats(frec[s][0], frec[s][1], frec[q][0], frec[q][1], "France"):
        cnt[k] += 1
print(f"\nFRANCE test: rejected best candidates (0.05<=p<0.775): {len(rej)}; in categories: {dict(cnt)}"
      f"  (France S1 = 259452)")
