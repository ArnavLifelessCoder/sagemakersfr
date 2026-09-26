"""Global threshold vs per-S1 expected-F0.5 decoding, on a labelled slice."""
import sys, os, collections
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
H = os.path.dirname(os.path.abspath(__file__))
SL, OUT = os.path.join(H, sys.argv[1]), os.path.join(H, sys.argv[2])
gt = pd.read_csv(os.path.join(SL, "gt.tsv"), sep="\t", dtype=str, keep_default_na=False)
truth = {s: set(x for x in m.split(",") if x) for s, m in zip(gt.source1_entity_id, gt.matched_entity_ids)}
s1 = pd.read_csv(os.path.join(SL, "test", "test_source1.tsv"), sep="\t", dtype=str, keep_default_na=False, quoting=3,
                 usecols=["entity_id", "country"])
ps = pd.read_parquet(os.path.join(OUT, "pair_scores.parquet"))
best = ps.sort_values("p", ascending=False).drop_duplicates("q")
groups = {s: (g.q.values, g.p.values) for s, g in best.groupby("s1")}


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


def score(pred):
    return np.mean([f05(pred.get(s, set()), truth.get(s, set())) for s in s1.entity_id.values])


def by_threshold(t):
    return {s: set(q[p >= t]) for s, (q, p) in groups.items()}


def expected_f(p, k, r, n_sim=0):
    """E[F0.5] of predicting the top-k (p sorted desc); r = expected true matches outside the candidates."""
    if k == 0:
        return float(np.prod(1 - p)) * np.exp(-r)
    tp = p[:k].sum()
    tot = p.sum() + r
    return 1.25 * tp / (0.25 * tot + k)


def by_expected_f(r, shrink=1.0):
    out = {}
    for s, (q, p) in groups.items():
        o = np.argsort(-p)
        q, pp = q[o], np.clip(p[o] * shrink, 0, 1)
        pp = pp[pp > 1e-4]
        if len(pp) == 0:
            continue
        ef = [expected_f(pp, k, r) for k in range(len(pp) + 1)]
        k = int(np.argmax(ef))
        out[s] = set(q[:k])
    return out


for t in (0.5, 0.65, 0.75, 0.8):
    print(f"threshold {t}: macro F0.5 {score(by_threshold(t)):.5f}")
for r in (0.0, 0.05, 0.15):
    for shrink in (1.0, 0.9):
        print(f"expected-F decoding r={r} shrink={shrink}: macro F0.5 {score(by_expected_f(r, shrink)):.5f}")
