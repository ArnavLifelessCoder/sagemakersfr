import os, sys, time, pickle
sys.path.insert(0, r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\code\business_entity_resolution")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from src.prep import parse_frame, learn_translit
from src.blocking import generate_candidates
from src.data import gt_pairs

DEV = os.path.join(os.path.dirname(__file__), "dev")


def main():
    t = time.time()
    s1 = pd.read_parquet(f"{DEV}/src1.parquet")
    q = pd.concat([pd.read_parquet(f"{DEV}/src2.parquet"), pd.read_parquet(f"{DEV}/src3.parquet")], ignore_index=True)
    gt = pd.read_parquet(f"{DEV}/gt.parquet")
    pairs = gt_pairs(gt)
    name_of = dict(zip(s1.entity_id, s1.business_name))
    qname = dict(zip(q.entity_id, q.business_name))
    tr = learn_translit([name_of[a] for a in pairs.s1], [qname.get(b, "") for b in pairs.other])
    print("translit entries", len(tr), list(tr.items())[:15])
    s1 = parse_frame(s1, tr, workers=6)
    q = parse_frame(q, tr, workers=6)
    print(flush=True); print("parsed", time.time() - t)
    c = generate_candidates(s1, q, k_name=20, k_addr=20, k_name_nostate=20, threads=8)
    print("cands", len(c), "per q %.2f" % (len(c) / len(q)), time.time() - t)
    c["s1"] = s1.entity_id.values[c.si.values]
    c["q"] = q.entity_id.values[c.qi.values]
    pickle.dump((s1, q, c, pairs, tr), open(f"{DEV}/block.pkl", "wb"))
    truth = dict(zip(pairs.other, pairs.s1))
    c["y"] = (c.q.map(truth) == c.s1).astype(np.int8)
    npos = len(pairs)
    # rank within channel
    c["rn"] = c.groupby("qi").name_cos.rank(ascending=False, method="first")
    c["ra"] = c.groupby("qi").addr_cos.rank(ascending=False, method="first")
    for k in (1, 3, 5, 10, 20):
        sel = c[((c.rn <= k) & (c.name_cos > 0)) | ((c.ra <= k) & (c.addr_cos > 0))]
        print(f"k={k:2d} recall {sel.y.sum() / npos:.5f}  pairs/q {len(sel) / len(q):.2f}")
    print("name-only recall", c[(c.rn <= 20) & (c.name_cos > 0)].y.sum() / npos,
          "addr-only recall", c[(c.ra <= 20) & (c.addr_cos > 0)].y.sum() / npos)
    found = set(c.loc[c.y == 1, "q"])
    miss = pairs[~pairs.other.isin(found)].sample(25, random_state=0)
    qi = q.set_index("entity_id")
    si = s1.set_index("entity_id")
    for a, b in zip(miss.s1, miss.other):
        print("MISS", si.loc[a, ["business_name", "business_address"]].tolist(), "||",
              qi.loc[b, ["business_name", "business_address"]].tolist())


if __name__ == "__main__":
    main()
