import pickle, sys, collections
import numpy as np, lightgbm as lgb
sys.path.insert(0, r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\code\business_entity_resolution")
sys.path.insert(0, __file__.rsplit("\\", 1)[0])
from src import model as M
from group_rule import sibling_flags

OLD = r"C:\Users\ARNAVG~1\AppData\Local\Temp\claude\C--Users-Arnav-Gawade-pro--OneDrive-Desktop-amazon-ml\89303762-6110-432e-ae87-2d206a5ca21f\scratchpad\dev"
s1, q, _, pairs = pickle.load(open(OLD + r"\parsed.pkl", "rb"))
c, f, extra, miss, y, is_b, lex, feats = pickle.load(open(OLD + r"\feat.pkl", "rb"))
bst = lgb.Booster(model_file=OLD + r"\lgb.txt")
M.add_lex(f, extra, miss, lex)
p = bst.predict(f[feats])
aq, as_ = M.assign(c.qi.values, c.si.values, p, 0.7)
truth = dict(zip(pairs.other.values, pairs.s1.values))
sid, qid = s1.entity_id.values, q.entity_id.values
clusters = collections.defaultdict(list)
for a, b in zip(aq, as_):
    clusters[b].append(a)
tp_fl = fp_fl = tp = fp = 0
for si, qs in clusters.items():
    fl = sibling_flags(s1.nums.values[si], [(qi, q.nums.values[qi]) for qi in qs])
    for qi in qs:
        ok = truth.get(qid[qi]) == sid[si]
        tp += ok
        fp += not ok
        if qi in fl:
            tp_fl += ok
            fp_fl += not ok
print(f"dev assigned: true {tp}, false {fp}; sibling-flagged true {tp_fl} ({tp_fl / tp:.4%}), false {fp_fl} ({fp_fl / max(1, fp):.2%})")
