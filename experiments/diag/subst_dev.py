"""On dev (train labels): content-word substitution + changed number -> how often true vs false?"""
import pickle, sys, collections
import numpy as np
sys.path.insert(0, r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\code\business_entity_resolution")
sys.stdout.reconfigure(encoding="utf-8")
OLD = r"C:\Users\ARNAVG~1\AppData\Local\Temp\claude\C--Users-Arnav-Gawade-pro--OneDrive-Desktop-amazon-ml\89303762-6110-432e-ae87-2d206a5ca21f\scratchpad\dev"
s1, q, _, pairs = pickle.load(open(OLD + r"\parsed.pkl", "rb"))
c, f, extra, miss, y, is_b, lex, feats = pickle.load(open(OLD + r"\feat.pkl", "rb"))
ctry = s1.country.values[c.si.values]
sub = (f.ex_n_vocab.values >= 1) & (f.mi_n_vocab.values >= 1)
numchg = (f.num_s_only.values >= 1) & (f.num_q_only.values >= 1)
close = f.n_tset.values >= 50
for cc in ("India", "US"):
    m = ctry == cc
    for name, mask in [("substitution", sub), ("subst & number changed", sub & numchg),
                       ("number changed only", numchg & ~sub)]:
        mm = m & mask & close
        print(f"{cc:6s} {name:26s} pairs {mm.sum():7d}  true share {y[mm].mean():.3f}")
# examples of TRUE substitution pairs in India
idx = np.flatnonzero((ctry == "India") & sub & (y == 1))
rng = np.random.RandomState(0)
print("\nTRUE India pairs flagged as substitution:")
for i in rng.choice(idx, 20, replace=False):
    print("  S1:", s1.name_core.values[c.si.values[i]], "| Q:", q.name_core.values[c.qi.values[i]],
          "| extra:", extra[i], "| missing:", miss[i])
idx = np.flatnonzero((ctry == "India") & sub & (y == 0) & close)
print("\nFALSE India pairs flagged as substitution:")
for i in rng.choice(idx, min(12, len(idx)), replace=False):
    print("  S1:", s1.name_core.values[c.si.values[i]], "| Q:", q.name_core.values[c.qi.values[i]],
          "| extra:", extra[i], "| missing:", miss[i])
