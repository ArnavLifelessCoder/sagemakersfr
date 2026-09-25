import sys, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
H = __file__.rsplit("\\", 1)[0]
V2 = H + r"\v2\pair_scores.parquet"
names = ["Chennai Cosmetics Private Limited", "Kamdhenu Plasto Private Limited", "Channel Collective Pvt Ltd",
         "Cresco Technologies Private Limited", "Rk Infratel (India) Co"]
rec = {}
for k in (1, 2, 3):
    df = pd.read_csv(rf"{H}\tslice\test\test_source{k}.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=3)
    rec.update(dict(zip(df.entity_id, zip(df.business_name, df.business_address))))
s1 = pd.read_csv(rf"{H}\tslice\test\test_source1.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=3)
ids = s1[s1.business_name.isin(names)].entity_id.tolist()
v3 = pd.read_parquet(rf"{H}\tslice_out\pair_scores.parquet")
v2 = pd.read_parquet(V2, filters=[("s1", "in", ids)])
for s in ids:
    print("\nS1", rec[s])
    a = v3[v3.s1 == s].merge(v2[v2.s1 == s][["q", "p"]], on="q", how="outer", suffixes=("_v3dev", "_v2"))
    for _, r in a.sort_values("p_v2", ascending=False).iterrows():
        print(f"   v2 {r.p_v2:.3f}  v3dev {r.p_v3dev:.3f}  {rec.get(r.q)}")
