"""Cluster rule: a candidate q of S1 s is a 'sibling-cluster member' if
  - q's house identifier differs from s's (and is not a truncation / zero-pad / prefix of it),
  - at least one OTHER candidate of s (any score) carries the same identifier as q,
  - and s's own identifier is confirmed by a confident candidate (p >= 0.97).
Evaluates label precision on the labelled train-like slice and counts on the test scores."""
import sys, os, collections
import numpy as np, pandas as pd
sys.path.insert(0, r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\code\business_entity_resolution")
sys.stdout.reconfigure(encoding="utf-8")
from src.normalize import parse_address

H = os.path.dirname(os.path.abspath(__file__))
T = r"C:\Users\ARNAVG~1\AppData\Local\Temp\claude\C--Users-Arnav-Gawade-pro--OneDrive-Desktop-amazon-ml\89303762-6110-432e-ae87-2d206a5ca21f\scratchpad\sr\student_resource\dataset\test"


def related(a, b):
    return a == b or a.startswith(b) or b.startswith(a) or a.endswith(b) or b.endswith(a)


def load_hn(ids, dirs):
    out, ctry = {}, {}
    for d in dirs:
        for k in (1, 2, 3):
            with open(os.path.join(d, f"test_source{k}.tsv"), encoding="utf-8") as f:
                next(f)
                for l in f:
                    p = l.rstrip("\n").split("\t")
                    if p[0] in ids:
                        out[p[0]] = parse_address(p[2], p[3])["hn"] if p[2] else ""
                        ctry[p[0]] = p[3]
    return out, ctry


def flag(ps, hn):
    """ps: all scored pairs (s1, q, p). returns boolean per row of ps."""
    fl = np.zeros(len(ps), bool)
    pos = {i: k for k, i in enumerate(ps.index)}
    for s, g in ps.groupby("s1"):
        shn = hn.get(s, "")
        if not shn:
            continue
        qh = [hn.get(q, "") for q in g.q]
        if not any(h == shn and p >= 0.97 for h, p in zip(qh, g.p)):
            continue
        cnt = collections.Counter(h for h in qh if h and not related(h, shn))
        for i, h in zip(g.index, qh):
            if h and not related(h, shn) and cnt[h] >= 2:
                fl[pos[i]] = True
    return fl


which = sys.argv[1]
if which == "eval":
    SL, OUT = os.path.join(H, "eslice"), os.path.join(H, "eslice_out_nb2")
    gt = pd.read_csv(os.path.join(SL, "gt.tsv"), sep="\t", dtype=str, keep_default_na=False)
    owner = {q: s for s, m in zip(gt.source1_entity_id, gt.matched_entity_ids) for q in m.split(",") if q}
    ps = pd.read_parquet(os.path.join(OUT, "pair_scores.parquet")).drop_duplicates(["s1", "q"]).reset_index(drop=True)
    hn, ctry = load_hn(set(ps.s1) | set(ps.q), [os.path.join(SL, "test")])
    fl = flag(ps, hn)
    ps["flag"] = fl
    ps["y"] = [owner.get(q) == s for s, q in zip(ps.s1, ps.q)]
    best = ps.sort_values("p", ascending=False).drop_duplicates("q")
    for lo in (0.5, 0.75, 0.9, 0.97):
        m = (best.p >= lo) & best.flag
        print(f"train-like: flagged best-candidates with p>={lo}: {m.sum():6d}, TRUE share {best.y[m].mean():.3f}")
else:
    ps = pd.read_parquet(os.path.join(H, "v5s102", "pair_scores.parquet")).drop_duplicates(["s1", "q"]).reset_index(drop=True)
    ps = ps[ps.country == which].reset_index(drop=True)
    hn, _ = load_hn(set(ps.s1) | set(ps.q), [T])
    ps["flag"] = flag(ps, hn)
    best = ps.sort_values("p", ascending=False).drop_duplicates("q")
    n = {"France": 259452, "India": 809986, "US": 663106}[which]
    for lo, hi in ((0.5, 0.775), (0.775, 0.9), (0.9, 0.96), (0.96, 0.99), (0.99, 1.01)):
        m = (best.p >= lo) & (best.p < hi)
        print(f"TEST {which}: band [{lo},{hi}) best-candidates {m.sum():7d} ({m.sum() / n:.3f}/S1), "
              f"flagged {(m & best.flag).sum():6d} ({(m & best.flag).sum() / max(1, m.sum()):.3f})")
    best[best.flag][["s1", "q", "p"]].to_parquet(os.path.join(H, f"cluster_flags_{which}.parquet"), index=False)
