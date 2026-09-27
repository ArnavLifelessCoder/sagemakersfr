"""Which features push obviously-true pairs down? TreeSHAP (pred_contrib) on the dumped stage-2 features."""
import sys, os, glob, json
import numpy as np, pandas as pd, lightgbm as lgb
sys.stdout.reconfigure(encoding="utf-8")
H = os.path.dirname(os.path.abspath(__file__))
SL, OUT, ART = os.path.join(H, "eslice"), os.path.join(H, "eslice_out_f"), os.path.join(H, "art_v4e")
meta = json.load(open(os.path.join(ART, "meta.json")))
feats = meta["features2"]
bst = lgb.Booster(model_file=os.path.join(ART, "stage2.txt"))
F = pd.concat([pd.read_parquet(f) for f in glob.glob(os.path.join(OUT, "features_*.parquet"))], ignore_index=True)
gt = pd.read_csv(os.path.join(SL, "gt.tsv"), sep="\t", dtype=str, keep_default_na=False)
owner = {q: s for s, m in zip(gt.source1_entity_id, gt.matched_entity_ids) for q in m.split(",") if q}
F["y"] = [owner.get(q) == s for s, q in zip(F.s1, F.q)]
F["p"] = bst.predict(F[feats].values.astype(np.float32))
rec = {}
for k in (1, 2, 3):
    t = pd.read_csv(os.path.join(SL, "test", f"test_source{k}.tsv"), sep="\t", dtype=str, keep_default_na=False, quoting=3)
    rec.update(dict(zip(t.entity_id, zip(t.business_name, t.business_address))))
fn = F[F.y & (F.p < 0.65)]
tp = F[F.y & (F.p >= 0.65)].sample(min(20000, int((F.y & (F.p >= 0.65)).sum())), random_state=0)
print("true pairs scored < 0.65:", len(fn), " | true accepted sample:", len(tp))
cfn = bst.predict(fn[feats].values.astype(np.float32), pred_contrib=True)[:, :-1]
ctp = bst.predict(tp[feats].values.astype(np.float32), pred_contrib=True)[:, :-1]
diff = pd.DataFrame({"feature": feats, "mean_contrib_FN": cfn.mean(0), "mean_contrib_TP": ctp.mean(0)})
diff["gap"] = diff.mean_contrib_FN - diff.mean_contrib_TP
print("\nfeatures that pull rejected TRUE pairs down the most (vs accepted true pairs):")
print(diff.sort_values("gap").head(15).to_string(index=False))
print("\nfeature values: rejected-true median vs accepted-true median")
for f in diff.sort_values("gap").feature.head(10):
    print(f"  {f:20s} FN {fn[f].median():8.3f}   TP {tp[f].median():8.3f}")
# the obvious-typo examples
names = ["Salemo Interiors", "Northem Express Ligand Co", "Southem  Integrated Immfx", "MQ lnsurance  Inc."]
for n in names:
    ids = [k for k, v in rec.items() if v[0] == n]
    rows = F[F.q.isin(ids) & F.y]
    for i, r in rows.iterrows():
        c = bst.predict(r[feats].values.astype(np.float32).reshape(1, -1), pred_contrib=True)[0, :-1]
        top = sorted(zip(c, feats), key=lambda t: t[0])[:6]
        print(f"\n{n!r} p={r.p:.3f} extra={r.extra!r} miss={r.miss!r}")
        print("   most negative:", [(f, round(v, 2), round(float(r[f]), 3)) for v, f in top])
