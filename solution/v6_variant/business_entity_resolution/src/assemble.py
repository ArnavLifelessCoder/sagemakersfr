"""Assemble matching_results.tsv variants from per-country score checkpoints (scores_<country>.parquet).

    python -m src.assemble --scores-dir OUT_DIR [--override France=OTHER_DIR] --test-dir DATASET_DIR/test
                           --out OUT --meta ART_DIR/meta.json
                           [--decision expected|threshold] [--odds-div 1.9] [--odds-div-country France=3]
                           [--thr France=0.99 India=0.9 US=0.968] [--calibrate shape|none] [--tol 0.02]

Every variant reads the same checkpoints, so a variant costs minutes, not a re-scoring run.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from .data import write_id_lists
from .model import assign_expected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", required=True)
    ap.add_argument("--override", nargs="*", default=[], help="Country=DIR: take that country's checkpoint from DIR")
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--decision", choices=["expected", "threshold"], default="expected")
    ap.add_argument("--odds-div", type=float, default=1.0)
    ap.add_argument("--odds-div-country", nargs="*", default=[])
    ap.add_argument("--thr", nargs="*", default=[], help="fixed per-country thresholds (threshold decision)")
    ap.add_argument("--calibrate", choices=["shape", "none"], default="shape")
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--p-unfound", type=float, default=0.02)
    ap.add_argument("--band-ref", default=None, help="training dump dir: records per S1 per probability band on its "
                    "validation entities is the train-like reference; test bands inflated by r are rescaled p -> p / r")
    ap.add_argument("--band-gt", default=None, help="gt.parquet of the dump's entities (for the validation universe)")
    a = ap.parse_args()
    meta = json.load(open(a.meta))
    base_thr = meta["threshold"]
    target = meta.get("target_mps")
    over = dict(x.split("=") for x in a.override)
    div_c = {k: float(v) for k, v in (x.split("=") for x in a.odds_div_country)}
    thr_c = {k: float(v) for k, v in (x.split("=") for x in a.thr)}
    s1 = pd.read_csv(os.path.join(a.test_dir, "test_source1.tsv"), sep="\t", dtype=str, keep_default_na=False,
                     usecols=["entity_id", "country"], quoting=3)
    n_s1 = s1.country.value_counts().to_dict()
    match = {}
    lines = []
    BANDS = np.array([0.3, 0.5, 0.6, 0.7, 0.775, 0.85, 0.9, 0.95, 0.97, 0.99, 1.0001])
    ref = None
    if a.band_ref:
        import zlib
        D = pd.read_parquet(os.path.join(a.band_ref, "pairs.parquet"), columns=["s1_id", "q_id", "p2"])
        s1p = pd.read_parquet(os.path.join(a.band_ref, "s1_parsed.parquet"), columns=["entity_id", "country"])
        ctry_of = dict(zip(s1p.entity_id.values, s1p.country.values))
        gt = pd.read_parquet(a.band_gt)
        val = [x for x in gt.source1_entity_id.values if zlib.crc32(x.encode()) % 100 < 20]
        vset = set(val)
        b = D.sort_values("p2", ascending=False).drop_duplicates("q_id")
        b = b[b.s1_id.isin(vset)]
        b["country"] = b.s1_id.map(ctry_of)
        n_val = pd.Series([ctry_of[x] for x in val]).value_counts().to_dict()
        ref = {}
        for cc, g in b.groupby("country"):
            h, _ = np.histogram(g.p2.values, bins=BANDS)
            ref[cc] = h / n_val[cc]
        ref["*"] = np.mean([ref[k] for k in ref], axis=0)
        print("reference records/S1 per band:", {k: np.round(v, 3).tolist() for k, v in ref.items()})
    for c in sorted(n_s1):
        d = pd.read_parquet(os.path.join(over.get(c, a.scores_dir), f"scores_{c}.parquet"))
        best = d.sort_values("p", ascending=False).drop_duplicates("q").reset_index(drop=True)
        n = n_s1[c]
        if ref is not None:
            h, _ = np.histogram(best.p.values, bins=BANDS)
            test_b = h / n
            r_c = ref.get(c, ref["*"])
            infl = np.maximum(1.0, test_b / np.maximum(r_c, 1e-9))
            infl[-1] = 1.0  # the top band (>= 0.99) is trusted
            idx = np.clip(np.searchsorted(BANDS, best.p.values, side="right") - 1, 0, len(infl) - 1)
            p_new = np.where(best.p.values >= BANDS[0], best.p.values / infl[idx], best.p.values)
            print(f"  {c}: band inflation {np.round(infl, 2).tolist()}")
            best["p"] = p_new.astype(np.float32)
        k_target = target * n if target else None
        if a.decision == "threshold":
            t = thr_c.get(c, base_thr)
            above = int((best.p.values >= t).sum())
            if a.calibrate == "shape" and k_target and above > k_target * (1 + a.tol):
                t = float(max(t, np.sort(best.p.values)[::-1][int(round(k_target)) - 1]))
            kept = best[best.p.values >= t]
            info = f"threshold {t:.4f}"
        else:
            s_code, _ = pd.factorize(best.s1.values)
            pv = best.p.values.astype(np.float64)
            div = div_c.get(c, a.odds_div)
            kept_idx = assign_expected(np.arange(len(best)), s_code, pv, p_unfound=a.p_unfound, odds_div=div)[0]
            if a.calibrate == "shape" and k_target and len(kept_idx) > k_target * (1 + a.tol):
                lo, hi = div, 8.0
                for _ in range(7):
                    mid = (lo + hi) / 2
                    km = assign_expected(np.arange(len(best)), s_code, pv, p_unfound=a.p_unfound, odds_div=mid)[0]
                    if len(km) > k_target * (1 + a.tol):
                        lo = mid
                    else:
                        hi, kept_idx = mid, km
                div = hi
            kept = best.iloc[np.sort(kept_idx)]
            info = f"expected-F odds/{div:.3f}"
        for s, q in zip(kept.s1.values, kept.q.values):
            match.setdefault(s, []).append(q)
        single = 1 - kept.s1.nunique() / n
        lines.append(f"  {c:7s} {info:24s} matches/S1 {len(kept) / n:.3f}  singletons {single:.4f}  (target {target:.3f})")
    os.makedirs(a.out, exist_ok=True)
    write_id_lists(os.path.join(a.out, "matching_results.tsv"), s1.entity_id.values, match, "matched_entity_ids")
    print("\n".join(lines))
    print("wrote", os.path.join(a.out, "matching_results.tsv"))


if __name__ == "__main__":
    main()
