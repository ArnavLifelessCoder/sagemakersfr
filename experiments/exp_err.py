import os, sys, pickle, collections
sys.path.insert(0, r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\code\business_entity_resolution")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, lightgbm as lgb
from src import model as M

DEV = os.path.join(os.path.dirname(__file__), "dev")
s1, q, _, pairs = pickle.load(open(f"{DEV}/parsed.pkl", "rb"))
c, f, extra, miss, y, is_b, lex, feats = pickle.load(open(f"{DEV}/feat.pkl", "rb"))
bst = lgb.Booster(model_file=f"{DEV}/lgb.txt")
pb = is_b[c.si.values]
M.add_lex(f, extra, miss, lex)  # (valid-style lexicon for all rows; fine for inspection)
p = bst.predict(f[feats])
s1_ids, q_ids = s1.entity_id.values, q.entity_id.values
aq, as_ = M.assign(c.qi.values, c.si.values, p, 0.7)
truth = dict(zip(pairs.other, pairs.s1))
b_set = set(s1_ids[is_b])
pred_s = s1_ids[as_]
pred_q = q_ids[aq]
fp = [(s, x) for s, x in zip(pred_s, pred_q) if s in b_set and truth.get(x) != s]
matched_q = set(pred_q)
pq = dict(zip(pred_q, pred_s))
fn = [(s, x) for x, s in truth.items() if s in b_set and pq.get(x) != s]
print("FP", len(fp), "FN", len(fn))
S = s1.set_index("entity_id")
Q = q.set_index("entity_id")
cat = collections.Counter()
for s, x in fp:
    t = truth.get(x)
    cat["fp_unmatched_q" if t is None else "fp_q_belongs_other"] += 1
print(cat)
def show(tag, lst, n=30):
    rng = np.random.RandomState(0)
    for k in rng.choice(len(lst), min(n, len(lst)), replace=False):
        s, x = lst[k]
        t = truth.get(x)
        print(tag, "S1:", S.loc[s, ["business_name", "business_address"]].tolist())
        print("      Q:", Q.loc[x, ["business_name", "business_address"]].tolist(), "true:", t if t is None else S.loc[t, "business_name"] if t in S.index else t)
show("FP", fp)
show("FN", fn)
