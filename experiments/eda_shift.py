"""Train -> test shift diagnostic on matching state slices.

For every S2/S3 record of a few states we retrieve its nearest S1 records (name char-3gram TF-IDF and
address-token TF-IDF, exactly like the pipeline's blocking) and look at how similar the best S1 is.
Train records are labelled (matched / unmatched), test records are not. Comparing the distributions
tells us what the ~1.1 extra records per S1 in the test set are:
  * "siblings" : a near-identical S1 exists (high name cosine) but the address / name differs slightly
  * "orphans"  : no similar S1 at all (their S1 was dropped from the test set)

    python experiments/eda_shift.py [states,comma,separated]
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "code", "business_entity_resolution"))
from src.blocking import iter_candidates  # noqa: E402
from src.data import COLS, gt_pairs, read_tsv  # noqa: E402
from src.make_dev_subset import state_of  # noqa: E402
from src.prep import extend_translit, learn_translit, parse_frame  # noqa: E402

DATA = os.path.join(ROOT, "data", "student_resource", "dataset")
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def load_slice(split, states):
    parts = []
    for k in (1, 2, 3):
        df = read_tsv(os.path.join(DATA, split, f"{split}_source{k}.tsv"))[COLS]
        st = np.array([state_of(a, c) if a else "" for a, c in zip(df.business_address.values, df.country.values)])
        keep = np.isin(st, list(states))
        df = df[keep].copy()
        df["src"] = k
        parts.append(df)
        log(split, "source", k, "kept", int(keep.sum()), "of", len(keep))
    return pd.concat(parts, ignore_index=True)


def best_per_query(s1, q, translit):
    s1 = parse_frame(s1, translit)
    q = parse_frame(q, translit)
    rows = []
    for _, cand in iter_candidates(s1, q, k_name=10, k_addr=10, k_name_nostate=10, threads=-1, log=log):
        rows.append(cand)
    c = pd.concat(rows, ignore_index=True).drop_duplicates(["qi", "si"])
    c["comb"] = c.name_cos + c.addr_cos
    idx_n = c.groupby("qi").name_cos.idxmax()
    idx_c = c.groupby("qi").comb.idxmax()
    bn = c.loc[idx_n, ["qi", "si", "name_cos", "addr_cos"]].rename(columns={"si": "si_n", "name_cos": "n_best", "addr_cos": "a_at_nbest"})
    bc = c.loc[idx_c, ["qi", "si", "name_cos", "addr_cos"]].rename(columns={"si": "si_c", "name_cos": "n_at_cbest", "addr_cos": "a_at_cbest"})
    b = bn.merge(bc, on="qi")
    b["a_best"] = c.groupby("qi").addr_cos.max().reindex(b.qi.values).values
    b["s1_hn_eq"] = (s1.hn.values[b.si_c.values] == q.hn.values[b.qi.values]) & (q.hn.values[b.qi.values] != "")
    b["q_id"] = q.entity_id.values[b.qi.values]
    b["s1_id_c"] = s1.entity_id.values[b.si_c.values]
    b["country"] = q.country.values[b.qi.values]
    b["src"] = q.src.values[b.qi.values]
    full = pd.DataFrame({"q_id": q.entity_id.values, "country": q.country.values, "src": q.src.values})
    b = full.merge(b.drop(columns=["country", "src"]), on="q_id", how="left")
    for col in ("n_best", "a_best", "n_at_cbest", "a_at_cbest", "a_at_nbest"):
        b[col] = b[col].fillna(0.0)
    b["s1_hn_eq"] = b.s1_hn_eq.fillna(False).astype(bool)
    return b, s1, q


def describe(b, label):
    bins = [0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01]
    h = pd.cut(b.n_best, bins, right=False).value_counts(normalize=True).sort_index()
    log(f"{label}: n={len(b)}  best-name-cos histogram: " + " ".join(f"{str(i):>12s}:{v:.3f}" for i, v in h.items()))
    strong = b.n_best >= 0.8
    log(f"{label}: share with a near-identical S1 name (cos>=0.8) {strong.mean():.3f}; of those, S1 house number equal {b.s1_hn_eq[strong].mean():.3f}; addr cos>=0.6 {(b.a_at_cbest[strong] >= 0.6).mean():.3f}")
    log(f"{label}: share with NO similar S1 (name cos<0.4 and addr cos<0.4): {((b.n_best < 0.4) & (b.a_best < 0.4)).mean():.4f}")


def main():
    states = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else {"US-VT", "US-ME", "US-NM", "US-MT", "US-DE", "US-ND",
                                                                   "IN-KL", "IN-PB", "IN-OD", "FR-PDL"}
    out = {}
    tr = load_slice("train", states)
    gt = read_tsv(os.path.join(DATA, "train", "train_ground_truth.tsv"))
    pairs = gt_pairs(gt)
    name_of = dict(zip(tr.entity_id.values, tr.business_name.values))
    translit = learn_translit([name_of.get(x, "") for x in pairs.s1.values], [name_of.get(x, "") for x in pairs.other.values])
    log("translit entries (from slice pairs)", len(translit))
    truth = dict(zip(pairs.other.values, pairs.s1.values))
    s1 = tr[tr.src == 1].reset_index(drop=True)
    q = tr[tr.src != 1].reset_index(drop=True)
    b, s1p, qp = best_per_query(s1, q, extend_translit(translit, tr.business_name.values))
    b["matched"] = b.q_id.map(truth).notna()
    b["top1_is_true"] = b.s1_id_c.values == b.q_id.map(truth).values
    n_s1 = s1.country.value_counts()
    log("TRAIN slice records per S1 by country:", (q.country.value_counts() / n_s1).round(3).to_dict())
    log("TRAIN matched per S1:", (q[q.entity_id.isin(truth)].country.value_counts() / n_s1).round(3).to_dict(),
        "unmatched per S1:", (q[~q.entity_id.isin(truth)].country.value_counts() / n_s1).round(3).to_dict())
    for c in sorted(b.country.unique()):
        m = b.country == c
        describe(b[m & b.matched], f"TRAIN {c} matched")
        describe(b[m & ~b.matched], f"TRAIN {c} unmatched")
        log(f"TRAIN {c}: top-1 (name+addr) is the true S1 for {b.top1_is_true[m & b.matched].mean():.4f} of matched records")
    out["train"] = b
    del tr
    te = load_slice("test", states)
    s1 = te[te.src == 1].reset_index(drop=True)
    q = te[te.src != 1].reset_index(drop=True)
    b2, s1p2, qp2 = best_per_query(s1, q, extend_translit(translit, te.business_name.values))
    n_s1 = s1.country.value_counts()
    log("TEST slice records per S1 by country:", (q.country.value_counts() / n_s1).round(3).to_dict())
    for c in sorted(b2.country.unique()):
        describe(b2[b2.country == c], f"TEST {c} all")
    # mixture decomposition: test = matched-like + unmatched-like + orphan-like, using the 'no similar S1' and 'strong name' bins
    for c in sorted(b2.country.unique()):
        if c not in set(b.country.unique()):
            continue
        tm, tu, tt = b[(b.country == c) & b.matched], b[(b.country == c) & ~b.matched], b2[b2.country == c]
        f = lambda d: np.array([((d.n_best >= 0.8) & d.s1_hn_eq).mean(), ((d.n_best >= 0.8) & ~d.s1_hn_eq).mean(),
                                ((d.n_best < 0.8) & (d.n_best >= 0.4)).mean(), (d.n_best < 0.4).mean()])
        log(f"{c} bins [strong&hn_eq, strong&hn_diff, mid, weak]: train-matched {np.round(f(tm), 3)} train-unmatched {np.round(f(tu), 3)} TEST {np.round(f(tt), 3)}")
    # examples from test: sibling-like and orphan-like
    rec = dict(zip(te.entity_id.values, zip(te.business_name.values, te.business_address.values)))
    rng = np.random.RandomState(0)
    for label, mask in (("SIBLING-LIKE (name cos>=0.8, S1 house number differs)", (b2.n_best >= 0.8) & ~b2.s1_hn_eq),
                        ("ORPHAN-LIKE (no similar S1)", (b2.n_best < 0.4) & (b2.a_best < 0.4))):
        idx = np.flatnonzero(mask.values)
        print(f"\n===== TEST {label}: {len(idx)} records")
        for i in rng.choice(idx, min(14, len(idx)), replace=False):
            r = b2.iloc[i]
            print(f"  Q  {rec[r.q_id]}\n  S1 {rec.get(r.s1_id_c)}   name_cos {r.n_at_cbest:.2f} addr_cos {r.a_at_cbest:.2f}\n")
    b.to_parquet(os.path.join(HERE, "eda_shift_train.parquet"), index=False)
    b2.to_parquet(os.path.join(HERE, "eda_shift_test.parquet"), index=False)
    log("done")


if __name__ == "__main__":
    main()
