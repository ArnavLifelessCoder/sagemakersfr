import sys, os, collections, random
import pandas as pd
from rapidfuzz.fuzz import ratio
sys.path.insert(0, r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\code\business_entity_resolution")
sys.stdout.reconfigure(encoding="utf-8")
from two_camp import OLD, load, related
SL = os.path.join(OLD, "eslice")
gt = pd.read_csv(os.path.join(SL, "gt.tsv"), sep="\t", dtype=str, keep_default_na=False)
owner = {q: s for s, m in zip(gt.source1_entity_id, gt.matched_entity_ids) for q in m.split(",") if q}
s1r = pd.read_csv(os.path.join(SL, "test", "test_source1.tsv"), sep="\t", dtype=str, keep_default_na=False, quoting=3, usecols=["entity_id", "country"])
rc = dict(zip(s1r.entity_id, s1r.country))
rp = pd.read_parquet(os.path.join(OLD, "eslice_out_ocr", "pair_scores.parquet"))
rp = rp[(rp.p >= 0.3) & (rp.s1.map(rc) == "US")]
best = rp.sort_values("p", ascending=False).drop_duplicates("q")
info = load(os.path.join(SL, "test"), set(best.s1) | set(best.q), "US")
tr, fa = [], []
wc_t, wc_f = collections.Counter(), collections.Counter()
for s, q, p in zip(best.s1, best.q, best.p):
    if s not in info or q not in info: continue
    cs, hs = info[s]; cq, hq = info[q]
    if not (len(cs) == len(cq) and len(cs) >= 2 and len(cs - cq) == 1): continue
    if hs and hq and not related(hs, hq): continue
    a, = cs - cq; b, = cq - cs
    if ratio(a, b) >= 60: continue
    y = owner.get(q) == s
    (tr if y else fa).append((s, q, p, a, b))
    (wc_t if y else wc_f)[b] += 1
print("US ref word<60 swaps: true", len(tr), "false", len(fa))
print("most common NEW words, true :", wc_t.most_common(25))
print("most common NEW words, false:", wc_f.most_common(25))
random.seed(0)
smp = random.sample(tr, 12) + random.sample(fa, min(12, len(fa)))
ids = {x for e in smp for x in e[:2]}
rec = {}
for k in (1, 2, 3):
    with open(os.path.join(SL, "test", f"test_source{k}.tsv"), encoding="utf-8") as f:
        next(f)
        for l in f:
            p_ = l.rstrip("\n").split("\t")
            if p_[0] in ids: rec[p_[0]] = (p_[1], p_[2])
for i, (s, q, p, a, b) in enumerate(smp):
    print(("TRUE " if i < 12 else "FALSE") + f" p={p:.3f} [{a}->{b}]  S1 {rec[s][0]} | {rec[s][1]}\n        q  {rec[q][0]} | {rec[q][1]}")
