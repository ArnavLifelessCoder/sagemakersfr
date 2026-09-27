"""Stratified label-shift correction (final decision layer).

    python -m src.stratify --ref-scores REF/pair_scores.parquet --ref-dir REF_SLICE (test/ + gt.tsv)
                           --scores TEST/pair_scores.parquet --test-dir DATASET/test --out OUT
                           [--min-prec 0.75]

Idea: test holds sibling distractors that a single-pair model cannot tell from noisy true records, so the
model's probabilities are too optimistic in some regions of feature space. Every candidate pair (its S2/S3
record's best S1) is put into a cell = probability band x house-number relation (same/related, missing,
changed) x exact core name (yes/no) x legal-form change (yes/no). On a LABELLED reference slice scored by the
SAME model we know, per S1, how many true pairs each cell holds. On test, the same cell usually holds that many
true pairs plus the sibling excess, so its precision is estimated as
    ref_true_per_S1(cell) / test_pairs_per_S1(cell)
and the cell is accepted only if that clears --min-prec (F0.5 break-even ~0.71-0.78). Cells with too little
reference data back off to the probability band alone. Countries without reference data (France) use the
pooled reference.
"""
import argparse
import collections
import os

import numpy as np
import pandas as pd

from .data import write_id_lists
from .normalize import parse_address, parse_name

BANDS = [0.3, 0.5, 0.65, 0.775, 0.85, 0.9, 0.95, 0.97, 0.99, 1.01]


def _related(a, b):
    return a == b or a.startswith(b) or b.startswith(a) or a.endswith(b) or b.endswith(a)


def read_records(test_dir, ids):
    rec = {}
    for k in (1, 2, 3):
        with open(os.path.join(test_dir, f"test_source{k}.tsv"), encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\n").split("\t")
                if p[0] in ids:
                    rec[p[0]] = (p[1], p[2], p[3])
    return rec


def cells(best, rec):
    """Add cell columns to a frame of best pairs (s1, q, p)."""
    cache = {}

    def parsed(i):
        r = cache.get(i)
        if r is None:
            n, a, c = rec[i]
            pn = parse_name(n)
            pa = parse_address(a, c) if a else {"hn": "", "empty": 1}
            r = (tuple(sorted(pn["core"])), set(pn["legal"]), pa["hn"], pa["empty"])
            cache[i] = r
        return r

    hn_g, nm_g, lg_g = [], [], []
    for s, q in zip(best.s1.values, best.q.values):
        cs, ls, hs, _ = parsed(s)
        cq, lq, hq, eq = parsed(q)
        if not hq or not hs:
            hn_g.append("missing")
        elif _related(hq, hs):
            hn_g.append("same")
        else:
            hn_g.append("changed")
        nm_g.append(int(cs == cq and len(cs) > 0))
        lg_g.append(int(ls != lq))
    out = best.copy()
    out["band"] = np.digitize(out.p.values, BANDS)
    out["hn_g"] = hn_g
    out["nm_g"] = nm_g
    out["lg_g"] = lg_g
    return out


def best_pairs(scores, min_p):
    d = pd.read_parquet(scores).drop_duplicates(["s1", "q"])
    best = d.sort_values("p", ascending=False).drop_duplicates("q")
    return best[best.p >= min_p].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-scores", required=True)
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-prec", type=float, default=0.75)
    ap.add_argument("--min-ref", type=int, default=30)
    ap.add_argument("--eval-gt", default=None, help="gt.tsv of the target (only for offline validation)")
    a = ap.parse_args()
    lo = BANDS[0]

    # ---- reference: labelled
    gt = pd.read_csv(os.path.join(a.ref_dir, "gt.tsv"), sep="\t", dtype=str, keep_default_na=False)
    owner = {q: s for s, m in zip(gt.source1_entity_id, gt.matched_entity_ids) for q in m.split(",") if q}
    rb = best_pairs(a.ref_scores, lo)
    rrec = read_records(os.path.join(a.ref_dir, "test"), set(rb.s1) | set(rb.q))
    rb = cells(rb, rrec)
    rb["country"] = [rrec[s][2] for s in rb.s1]
    rb["y"] = [owner.get(q) == s for s, q in zip(rb.s1, rb.q)]
    rs1 = pd.read_csv(os.path.join(a.ref_dir, "test", "test_source1.tsv"), sep="\t", dtype=str,
                      keep_default_na=False, quoting=3, usecols=["entity_id", "country"])
    n_ref = rs1.country.value_counts().to_dict()
    n_ref["ALL"] = len(rs1)
    keys = ["band", "hn_g", "nm_g", "lg_g"]

    def ref_table(sub, n):
        g = sub.groupby(keys).agg(cnt=("y", "size"), tru=("y", "sum")).reset_index()
        g["tru_ps"] = g.tru / n
        gb = sub.groupby("band").agg(cnt_b=("y", "size"), tru_b=("y", "sum")).reset_index()
        gb["tru_ps_b"] = gb.tru_b / n
        return g, gb

    ref = {c: ref_table(rb[rb.country == c], n_ref[c]) for c in n_ref if c != "ALL"}
    ref["ALL"] = ref_table(rb, n_ref["ALL"])

    # ---- target
    tb = best_pairs(a.scores, lo)
    trec = read_records(a.test_dir, set(tb.s1) | set(tb.q))
    tb = cells(tb, trec)
    if "country" not in tb:
        tb["country"] = [trec[s][2] for s in tb.s1]
    ts1 = pd.read_csv(os.path.join(a.test_dir, "test_source1.tsv"), sep="\t", dtype=str,
                      keep_default_na=False, quoting=3, usecols=["entity_id", "country"])
    n_t = ts1.country.value_counts().to_dict()
    keep = np.zeros(len(tb), bool)
    report = []
    for c in sorted(n_t):
        m = (tb.country == c).values
        sub = tb[m]
        g, gb = ref.get(c, ref["ALL"])
        src = c if c in ref else "ALL"
        tc = sub.groupby(keys).size().rename("tcnt").reset_index()
        tc["t_ps"] = tc.tcnt / n_t[c]
        tc = tc.merge(g, on=keys, how="left").merge(gb[["band", "tru_ps_b", "cnt_b"]], on="band", how="left")
        tcb = sub.groupby("band").size().rename("tcnt_b").reset_index()
        tcb["t_ps_b"] = tcb.tcnt_b / n_t[c]
        tc = tc.merge(tcb, on="band", how="left")
        fine = tc.cnt.fillna(0) >= a.min_ref
        prec_fine = (tc.tru_ps / tc.t_ps).clip(upper=1.0)
        prec_band = (tc.tru_ps_b / tc.t_ps_b).clip(upper=1.0)
        tc["prec"] = np.where(fine, prec_fine, prec_band).astype(float)
        tc["prec"] = tc.prec.fillna(0.0)
        tc.loc[tc.band == len(BANDS) - 1, "prec"] = np.maximum(tc.loc[tc.band == len(BANDS) - 1, "prec"], 1.0)
        tc["accept"] = tc.prec >= a.min_prec
        acc_cells = set(map(tuple, tc.loc[tc.accept, keys].values))
        keep[np.flatnonzero(m)] = [tuple(r) in acc_cells for r in sub[keys].values]
        report.append((c, src, tc))
    final = tb[keep]
    match = final.groupby("s1", sort=False).q.apply(list).to_dict()
    os.makedirs(a.out, exist_ok=True)
    write_id_lists(os.path.join(a.out, "matching_results.tsv"), ts1.entity_id.values, match, "matched_entity_ids")
    for c, src, tc in report:
        acc = tc[tc.accept]
        rej = tc[~tc.accept]
        print(f"{c} (ref {src}): accepted cells {len(acc)}, pairs/S1 accepted {acc.t_ps.sum():.3f}, "
              f"rejected {rej.t_ps.sum():.3f}")
        show = tc[(tc.band >= 3) & (tc.band <= 8)].sort_values(["band", "hn_g", "nm_g", "lg_g"])
        print(show[keys + ["t_ps", "tru_ps", "prec", "accept"]].round(4).head(40).to_string(index=False))
    stats = pd.DataFrame({"country": ts1.country, "n": [len(match.get(x, ())) for x in ts1.entity_id]})
    print(stats.groupby("country").n.agg(matches_per_S1="mean", singletons=lambda x: (x == 0).mean()).round(4))
    if a.eval_gt:
        g2 = pd.read_csv(a.eval_gt, sep="\t", dtype=str, keep_default_na=False)
        truth = {s: set(x for x in m.split(",") if x) for s, m in zip(g2.source1_entity_id, g2.matched_entity_ids)}

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

        sc = np.mean([f05(set(match.get(s, ())), truth.get(s, set())) for s in ts1.entity_id])
        base = collections.defaultdict(set)
        for s, q in zip(tb.s1[tb.p >= 0.65], tb.q[tb.p >= 0.65]):
            base[s].add(q)
        b = np.mean([f05(base.get(s, set()), truth.get(s, set())) for s in ts1.entity_id])
        print(f"EVAL: threshold 0.65 F0.5 {b:.5f} | stratified F0.5 {sc:.5f}")


if __name__ == "__main__":
    main()
