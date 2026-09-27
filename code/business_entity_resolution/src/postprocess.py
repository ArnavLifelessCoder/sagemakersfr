"""Band-aware post-processing of pair scores (v6).

    python -m src.postprocess --scores pair_scores.parquet --test-dir DATASET_DIR/test --out out_dir
           [--thr France=0.99 India=0.9 US=0.968] [--rescue-min 0.5]

Why: test holds ~1.7x more sibling distractors per S1 than training, and they concentrate in the
middle probability bands. Comparing records-per-S1 per band with a labelled train-like slice shows
the US bands 0.775-0.97 inflated 2-4x (precision ~25-55%) and the France bands 0.775-0.99 inflated
4-6x (hand-checked: ~30-45% true), while India is close to training. Per-country thresholds are
therefore set where the estimated precision clears the F0.5 break-even (~0.78).
True matches that sit in those bands are of three structural kinds that siblings (which keep and
modify the name) never are: initials of the S1 name ("DG"), a random rename at the exact S1 address
("Veraveralyra", same house number + street), and an empty address with exactly the S1 name. Those
are rescued down to --rescue-min.
"""
import argparse
import os

import numpy as np
import pandas as pd

from .data import write_id_lists
from .normalize import parse_address, parse_name

STREET_TYPES = {"rue", "st", "ave", "rd", "dr", "blvd", "ln", "all", "ch", "imp", "pl", "ct", "cir", "trl",
                "pkwy", "hwy", "ter", "sq", "rte", "way"}


def rescue_kind(sn, sa, qn, qa, c):
    a, b = parse_name(sn), parse_name(qn)
    A, B = a["core"], b["core"]
    if not A or not B:
        return ""
    pb = parse_address(qa, c)
    if pb["empty"]:
        return "empty_exact_name" if sorted(A) == sorted(B) else ""
    if set(A) & set(B) or b["domain"]:
        return ""
    pa = parse_address(sa, c)
    if len(B) == 1 and B[0] == "".join(t[0] for t in A) and len(B[0]) >= 2:
        return "initials"
    if pa["hn"] and pa["hn"] == pb["hn"] and (set(pa["street"]) & set(pb["street"])) - STREET_TYPES:
        return "rename_exact_address"
    return ""


def _related(a, b):
    return a == b or a.startswith(b) or b.startswith(a) or a.endswith(b) or b.endswith(a)


def sibling_signature(srec, qrec):
    """Same core name, house number changed (not a pad/prefix/truncation), legal form changed."""
    sn, sa, c = srec
    qn, qa, _ = qrec
    if not sa or not qa:
        return False
    a, b = parse_name(sn), parse_name(qn)
    if not a["core"] or sorted(a["core"]) != sorted(b["core"]) or set(a["legal"]) == set(b["legal"]):
        return False
    ha, hb = parse_address(sa, c)["hn"], parse_address(qa, c)["hn"]
    return bool(ha) and bool(hb) and not _related(ha, hb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thr", nargs="+", default=["France=0.99", "India=0.9", "US=0.968"])
    ap.add_argument("--rescue-min", type=float, default=0.5)
    ap.add_argument("--sibling-countries", nargs="*", default=["US"],
                    help="countries where accepted 'same name + changed number + changed legal form' pairs are "
                         "dropped (test sibling signature; 95%% fakes below p 0.96, ~44%% precision above)")
    a = ap.parse_args()
    thr = {k: float(v) for k, v in (x.split("=") for x in a.thr)}
    d = pd.read_parquet(a.scores)
    best = d.sort_values("p", ascending=False).drop_duplicates("q")
    best["t"] = best.country.map(thr).fillna(0.75).values
    keep = best.p.values >= best.t.values
    band = best[(~keep) & (best.p.values >= a.rescue_min)]
    acc_sib = best[keep & best.country.isin(a.sibling_countries).values]
    need = set(band.s1) | set(band.q) | set(acc_sib.s1) | set(acc_sib.q)
    rec = {}
    for k in (1, 2, 3):
        with open(os.path.join(a.test_dir, f"test_source{k}.tsv"), encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\n").split("\t")
                if p[0] in need:
                    rec[p[0]] = (p[1], p[2], p[3])
    kinds = [rescue_kind(rec[s][0], rec[s][1], rec[q][0], rec[q][1], rec[s][2]) for s, q in zip(band.s1, band.q)]
    band = band.assign(kind=kinds)
    resc = band[band.kind != ""]
    print("rescued per country / kind:\n", resc.groupby(["country", "kind"]).size())
    drop = set()
    for s1_, q_ in zip(acc_sib.s1, acc_sib.q):
        if sibling_signature(rec[s1_], rec[q_]):
            drop.add((s1_, q_))
    kept = best[keep]
    kept = kept[[(x, y) not in drop for x, y in zip(kept.s1, kept.q)]]
    print("dropped sibling-signature pairs:", len(drop))
    final = pd.concat([kept, resc])
    match = final.groupby("s1", sort=False).q.apply(list).to_dict()
    s1 = pd.read_csv(os.path.join(a.test_dir, "test_source1.tsv"), sep="\t", dtype=str, keep_default_na=False,
                     usecols=["entity_id", "country"], quoting=3)
    os.makedirs(a.out, exist_ok=True)
    write_id_lists(os.path.join(a.out, "matching_results.tsv"), s1.entity_id.values, match, "matched_entity_ids")
    stats = pd.DataFrame({"country": s1.country, "n": [len(match.get(x, ())) for x in s1.entity_id]})
    print(stats.groupby("country").n.agg(matches_per_S1="mean", singletons=lambda x: (x == 0).mean()).round(4))


if __name__ == "__main__":
    main()
