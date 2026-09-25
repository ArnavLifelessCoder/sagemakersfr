import pickle, sys, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\code\business_entity_resolution")
from src import model as M
c, f, extra, miss, y, is_b, lex, feats = pickle.load(open("dev/feat.pkl", "rb"))
bst = lgb.Booster(model_file="dev/lgb.txt")
M.add_lex(f, extra, miss, lex)
p = bst.predict(f[feats])
sub = f.substitution.values == 1
hi = p >= 0.7
print("all true pairs: substitution rate %.4f" % sub[y == 1].mean())
print("predicted matches (p>=0.7): n=%d  substitution among TRUE %.4f  among FALSE %.4f" % (hi.sum(), sub[hi & (y == 1)].mean(), sub[hi & (y == 0)].mean()))
print("pairs p>=0.7 with substitution: %d true, %d false" % ((hi & sub & (y == 1)).sum(), (hi & sub & (y == 0)).sum()))
for thr in (1e-4, 3e-4, 1e-3):
    pass
