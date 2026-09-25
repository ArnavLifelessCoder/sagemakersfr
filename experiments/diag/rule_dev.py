"""Validate 'replaced number' rule on dev (train slice with labels)."""
import pickle, sys, collections
import numpy as np, lightgbm as lgb
sys.path.insert(0, r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\code\business_entity_resolution")
sys.stdout.reconfigure(encoding="utf-8")
from src import model as M

OLD = r"C:\Users\ARNAVG~1\AppData\Local\Temp\claude\C--Users-Arnav-Gawade-pro--OneDrive-Desktop-amazon-ml\89303762-6110-432e-ae87-2d206a5ca21f\scratchpad\dev"
s1, q, _, pairs = pickle.load(open(OLD + r"\parsed.pkl", "rb"))
c, f, extra, miss, y, is_b, lex, feats = pickle.load(open(OLD + r"\feat.pkl", "rb"))
bst = lgb.Booster(model_file=OLD + r"\lgb.txt")
M.add_lex(f, extra, miss, lex)
p = bst.predict(f[feats])
aq, as_ = M.assign(c.qi.values, c.si.values, p, 0.7)
key = set(zip(aq, as_))
sel = np.array([(a, b) in key for a, b in zip(c.qi.values, c.si.values)])
print("assigned pairs", sel.sum(), "true", y[sel].sum(), "false", (y[sel] == 0).sum())


def covered(x, others):
    for o in others:
        if x == o or (len(x) >= 2 and len(o) >= 2 and (x.startswith(o) or o.startswith(x) or x.endswith(o) or o.endswith(x))):
            return True
    return False


def replaced(sn, qn):
    a, b = sn.split(), qn.split()
    if not a or not b:
        return 0
    s_missing = [x for x in a if not covered(x, b)]
    q_new = [x for x in b if not covered(x, a)]
    return int(bool(s_missing) and bool(q_new))


S_n, Q_n = s1.nums.values, q.nums.values
ctry = s1.country.values
r = np.array([replaced(S_n[si], Q_n[qi]) for qi, si in zip(c.qi.values, c.si.values)])
for cc in ("US", "India"):
    m = sel & (ctry[c.si.values] == cc)
    tp, fp = m & (y == 1), m & (y == 0)
    print(f"{cc:6s} assigned true {tp.sum():7d} rule-flag {r[tp].mean():.4f} | assigned false {fp.sum():5d} rule-flag {r[fp].mean():.4f}")
    # flagged true pairs examples
    idx = np.flatnonzero(tp & (r == 1))[:8]
    for i in idx:
        print("   TRUE flagged:", S_n[c.si.values[i]], "|", Q_n[c.qi.values[i]])
