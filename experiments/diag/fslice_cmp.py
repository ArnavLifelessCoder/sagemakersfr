"""v2 vs v3(dev) on the France PDL slice, same S1 and same S2/S3 records."""
import sys, pandas as pd, numpy as np
sys.stdout.reconfigure(encoding="utf-8")
H = __file__.rsplit("\\", 1)[0]
rec = {}
ids = {}
for k in (1, 2, 3):
    df = pd.read_csv(rf"{H}\fslice\test\test_source{k}.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=3)
    rec.update(dict(zip(df.entity_id, zip(df.business_name, df.business_address))))
    ids[k] = set(df.entity_id)
s1set, qset = ids[1], ids[2] | ids[3]
v3 = pd.read_parquet(rf"{H}\fslice_out\pair_scores.parquet")
v2 = pd.read_parquet(rf"{H}\v2\pair_scores.parquet", filters=[("country", "=", "France")])
v2 = v2[v2.s1.isin(s1set) & v2.q.isin(qset)]


def assigned(d, thr):
    b = d.sort_values("p", ascending=False).drop_duplicates("q")
    return b[b.p >= thr]


a2, a3 = assigned(v2, 0.75), assigned(v3, 0.75)
n = len(s1set)
print(f"S1 {n}: v2 matches/S1 {len(a2) / n:.3f} singletons {1 - a2.s1.nunique() / n:.3f} | "
      f"v3dev matches/S1 {len(a3) / n:.3f} singletons {1 - a3.s1.nunique() / n:.3f}")
k2 = set(zip(a2.s1, a2.q))
k3 = set(zip(a3.s1, a3.q))
only2, only3 = list(k2 - k3), list(k3 - k2)
print(f"both {len(k2 & k3)}, only v2 {len(only2)}, only v3 {len(only3)}")
rng = np.random.RandomState(0)
p3 = dict(zip(zip(v3.s1, v3.q), v3.p))
p2 = dict(zip(zip(v2.s1, v2.q), v2.p))
for tag, lst in (("ONLY v2 accepts", only2), ("ONLY v3 accepts", only3)):
    print(f"\n##### {tag}")
    for i in rng.choice(len(lst), min(22, len(lst)), replace=False):
        s, q = lst[i]
        print(f"v2 {p2.get((s, q), float('nan')):.3f} v3 {p3.get((s, q), float('nan')):.3f} | S1 {rec[s]} | Q {rec[q]}")
