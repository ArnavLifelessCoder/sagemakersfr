import sys, os, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
sys.argv = ["x", "eval", "eslice", "eslice_out_nb2", "0.65"]
exec(open("nearnum_rule.py", encoding="utf-8").read().split("mode = sys.argv[1]")[0])
SL, OUT = os.path.join(H, "eslice"), os.path.join(H, "eslice_out_nb2")
gt = pd.read_csv(os.path.join(SL, "gt.tsv"), sep="\t", dtype=str, keep_default_na=False)
owner = {q: s for s, m in zip(gt.source1_entity_id, gt.matched_entity_ids) for q in m.split(",") if q}
rec = load_rec([os.path.join(SL, "test")])
ps = pd.read_parquet(os.path.join(OUT, "pair_scores.parquet"))
best = ps.sort_values("p", ascending=False).drop_duplicates("q")
us = best[best.s1.map(lambda s: rec[s][2]) == "US"]
acc = us[us.p >= 0.65].reset_index(drop=True)
fl = flag_pairs(acc, rec, 0.97)
lab = np.array([owner.get(q) == s for s, q in zip(acc.s1, acc.q)])
print("US flagged", fl.sum(), "true share", round(lab[fl].mean(), 4))
print("p distribution of flagged TRUE:", np.percentile(acc.p[fl & lab], [10, 50, 90]).round(3))
rng = np.random.RandomState(0)
for tag, m in (("TRUE", fl & lab), ("FALSE", fl & ~lab)):
    idx = np.flatnonzero(m)
    print(f"\n### flagged {tag} examples")
    for i in rng.choice(idx, min(12, len(idx)), replace=False):
        s, q = acc.s1[i], acc.q[i]
        print(f"p={acc.p[i]:.2f}  S1 {rec[s][0]} | {rec[s][1][:50]}\n        Q  {rec[q][0]} | {rec[q][1][:50]}")
# how many US true pairs lie in the p band 0.775-0.96 and what share is true (train-like data)
band = us[(us.p >= 0.775) & (us.p < 0.96)]
bl = np.array([owner.get(q) == s for s, q in zip(band.s1, band.q)])
print(f"\nUS best candidates with 0.775 <= p < 0.96: {len(band)}, TRUE share {bl.mean():.4f}")
