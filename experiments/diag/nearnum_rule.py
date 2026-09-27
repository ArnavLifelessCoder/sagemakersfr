"""'Near-number sibling' rule: accepted q whose house number conflicts with the S1's while another
confident record of the same S1 carries exactly the S1's number.
mode 'eval' : labelled slice -> precision of the rule + F0.5 before/after dropping flagged pairs
mode 'test' : count flagged pairs per country in a test pair_scores file"""
import sys, os, collections
import numpy as np, pandas as pd
sys.path.insert(0, r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\code\business_entity_resolution")
sys.stdout.reconfigure(encoding="utf-8")
from src.normalize import parse_address, parse_name

H = os.path.dirname(os.path.abspath(__file__))


def related(a, b):
    return a == b or a.startswith(b) or b.startswith(a) or a.endswith(b) or b.endswith(a)


def flag_pairs(assigned, rec, anchor_p):
    """assigned: DataFrame s1,q,p of accepted pairs. rec: id -> (name, address, country)."""
    hn = {}
    for x in set(assigned.s1) | set(assigned.q):
        n, a, c = rec[x]
        hn[x] = parse_address(a, c)["hn"] if a else ""
    flags = np.zeros(len(assigned), bool)
    for s, g in assigned.groupby("s1"):
        shn = hn[s]
        if not shn:
            continue
        anchors = {q for q, p in zip(g.q, g.p) if p >= anchor_p and hn[q] == shn}
        if not anchors:
            continue
        for i, q in zip(g.index, g.q):
            qh = hn[q]
            if q in anchors or not qh or related(qh, shn):
                continue
            flags[assigned.index.get_loc(i)] = True
    return flags


def load_rec(dirs):
    rec = {}
    for d in dirs:
        for k in (1, 2, 3):
            t = pd.read_csv(os.path.join(d, f"test_source{k}.tsv"), sep="\t", dtype=str, keep_default_na=False, quoting=3)
            rec.update(dict(zip(t.entity_id, zip(t.business_name, t.business_address, t.country))))
    return rec


def f05(P, T):
    if not T and not P:
        return 1.0
    if not T or not P:
        return 0.0
    tp = len(P & T)
    if not tp:
        return 0.0
    pr, rc = tp / len(P), tp / len(T)
    return 1.25 * pr * rc / (0.25 * pr + rc)


mode = sys.argv[1]
if mode == "eval":
    SL, OUT, thr = os.path.join(H, sys.argv[2]), os.path.join(H, sys.argv[3]), float(sys.argv[4])
    gt = pd.read_csv(os.path.join(SL, "gt.tsv"), sep="\t", dtype=str, keep_default_na=False)
    truth = {s: set(x for x in m.split(",") if x) for s, m in zip(gt.source1_entity_id, gt.matched_entity_ids)}
    owner = {q: s for s, v in truth.items() for q in v}
    rec = load_rec([os.path.join(SL, "test")])
    ps = pd.read_parquet(os.path.join(OUT, "pair_scores.parquet"))
    best = ps.sort_values("p", ascending=False).drop_duplicates("q")
    acc = best[best.p >= thr].reset_index(drop=True)
    s1ids = list(rec_k for rec_k in rec if rec_k.startswith("S1"))
    for ap in (0.9, 0.97):
        fl = flag_pairs(acc, rec, ap)
        lab = np.array([owner.get(q) == s for s, q in zip(acc.s1, acc.q)])
        print(f"anchor p>={ap}: flagged {fl.sum()} of {len(acc)} accepted ({fl.mean():.4f}); "
              f"flagged TRUE share {lab[fl].mean():.4f}")
        for name, keep in (("before", np.ones(len(acc), bool)), ("after", ~fl)):
            pred = collections.defaultdict(set)
            for s, q in zip(acc.s1[keep], acc.q[keep]):
                pred[s].add(q)
            print(f"   {name}: macro F0.5 {np.mean([f05(pred.get(s, set()), truth.get(s, set())) for s in s1ids]):.5f}")
else:
    RUN, TDIR = sys.argv[2], sys.argv[3]
    ps = pd.read_parquet(os.path.join(H, RUN, "pair_scores.parquet"))
    mr = collections.defaultdict(set)
    with open(os.path.join(H, RUN, "matching_results.tsv"), encoding="utf-8") as f:
        next(f)
        for l in f:
            s, _, r = l.rstrip("\n").partition("\t")
            for x in r.split(","):
                if x:
                    mr[s].add(x)
    keyset = {(s, q) for s, v in mr.items() for q in v}
    acc = ps[[k in keyset for k in zip(ps.s1, ps.q)]].drop_duplicates(["s1", "q"]).reset_index(drop=True)
    rec = load_rec([TDIR])
    fl = flag_pairs(acc, rec, 0.97)
    acc["flag"] = fl
    print(acc.groupby("country").flag.agg(["sum", "mean"]))
    acc[acc.flag][["s1", "q"]].to_parquet(os.path.join(H, RUN, "nearnum_flags.parquet"), index=False)
