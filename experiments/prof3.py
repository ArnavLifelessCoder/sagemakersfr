import sys, collections
sys.path.insert(0, r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\code\business_entity_resolution")
import pandas as pd
from src.make_dev_subset import state_of
from src.data import gt_pairs
if __name__ == "__main__":
    s1 = pd.read_parquet("dev/src1.parquet"); q = pd.concat([pd.read_parquet("dev/src2.parquet"), pd.read_parquet("dev/src3.parquet")])
    st = {e: state_of(a, c) for e, a, c in zip(s1.entity_id, s1.business_address, s1.country)}
    qs = {e: (state_of(a, c) if a else "EMPTY") for e, a, c in zip(q.entity_id, q.business_address, q.country)}
    qc = dict(zip(q.entity_id, q.country)); sc = dict(zip(s1.entity_id, s1.country))
    p = gt_pairs(pd.read_parquet("dev/gt.parquet"))
    c = collections.Counter()
    for a, b in zip(p.s1, p.other):
        x, y = st[a], qs[b]
        c["country_mismatch" if sc[a] != qc[b] else ("s1_nostate" if not x else ("q_empty" if y == "EMPTY" else ("q_nostate" if not y else ("same" if x == y else "diff"))))] += 1
    print(c)
    print("unmatched q with no state:", sum(1 for e, v in qs.items() if v == ""), "of", len(qs))
