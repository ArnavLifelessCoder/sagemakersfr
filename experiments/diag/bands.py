import sys, os, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__))
bins = [0.5, 0.65, 0.775, 0.85, 0.9, 0.95, 0.97, 0.99, 1.01]
# reference: labelled train-like slice (dev v4 model + neighbour fix)
gt = pd.read_csv(os.path.join(H, "eslice", "gt.tsv"), sep="\t", dtype=str, keep_default_na=False)
owner = {q: s for s, m in zip(gt.source1_entity_id, gt.matched_entity_ids) for q in m.split(",") if q}
s1 = pd.read_csv(os.path.join(H, "eslice", "test", "test_source1.tsv"), sep="\t", dtype=str, keep_default_na=False, quoting=3, usecols=["entity_id", "country"])
ctry = dict(zip(s1.entity_id, s1.country))
ref = pd.read_parquet(os.path.join(H, "eslice_out_nb2", "pair_scores.parquet"))
rb = ref.sort_values("p", ascending=False).drop_duplicates("q")
rb["c"] = rb.s1.map(ctry)
rb["y"] = [owner.get(q) == s for s, q in zip(rb.s1, rb.q)]
print("REFERENCE (train-like, labelled): per band -> best-candidates per S1 and TRUE share")
for c in ("US", "India"):
    d = rb[rb.c == c]; n = (s1.country == c).sum()
    cnt = np.histogram(d.p, bins)[0] / n
    tru = [d.y[(d.p >= a) & (d.p < b)].mean() for a, b in zip(bins[:-1], bins[1:])]
    print(f"  {c:6s} per-S1 " + " ".join(f"{x:.3f}" for x in cnt) + "\n         true   " + " ".join(f"{x:.3f}" for x in tru))
t = pd.read_parquet(os.path.join(H, "v5s102", "pair_scores.parquet"))
tb = t.sort_values("p", ascending=False).drop_duplicates("q")
ns = {"France": 259452, "India": 809986, "US": 663106}
print("TEST v5 seed102: per band -> best-candidates per S1")
for c in ("US", "India", "France"):
    d = tb[tb.country == c]
    print(f"  {c:6s} per-S1 " + " ".join(f"{x:.3f}" for x in np.histogram(d.p, bins)[0] / ns[c]))
print("bands:", bins)
