"""Blend cross-encoder scores with the LightGBM probabilities.

Validation (choose the weight):
    python -m src.blend_ce val --dump DUMP_DIR --ce ce_val.parquet --gt GT_PARQUET [--val-pct 20]
Test (write blended per-country checkpoints for src.assemble):
    python -m src.blend_ce test --scores-dir OUT_DIR --ce ce_test.parquet --w 0.5 --out BLEND_DIR

Blend in logit space: logit(p) = (1 - w) logit(p_lgbm) + w logit(p_ce).
"""
import argparse
import os
import zlib

import numpy as np
import pandas as pd

from .data import gt_pairs
from .model import assign_expected, fbeta_macro, to_mapping


def logit(p):
    p = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def blend(p, c, w):
    z = (1 - w) * logit(p) + w * logit(c)
    return 1 / (1 + np.exp(-z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["val", "test"])
    ap.add_argument("--dump")
    ap.add_argument("--ce", required=True)
    ap.add_argument("--gt")
    ap.add_argument("--val-pct", type=int, default=20)
    ap.add_argument("--scores-dir")
    ap.add_argument("--w", type=float, default=0.5)
    ap.add_argument("--out")
    a = ap.parse_args()
    ce = pd.read_parquet(a.ce)
    if a.mode == "val":
        D = pd.read_parquet(os.path.join(a.dump, "pairs.parquet"), columns=["s1_id", "q_id", "p2", "y", "is_val"])
        D = D.merge(ce, on=["s1_id", "q_id"], how="left")
        gt = pd.read_parquet(a.gt)
        pairs = gt_pairs(gt)
        truth = to_mapping(pairs.s1.values, pairs.other.values)
        val = [s for s in gt.source1_entity_id.values if zlib.crc32(s.encode()) % 100 < a.val_pct]
        print(f"pairs {len(D)}, with CE score {D.ce.notna().mean():.3f}")
        D["ce"] = D.ce.fillna(D.p2)
        cov = D.ce.notna()
        for w in (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0):
            p = blend(D.p2.values, D.ce.values, w)
            best = pd.DataFrame({"s1": D.s1_id.values, "q": D.q_id.values, "p": p}).sort_values("p", ascending=False).drop_duplicates("q")
            best = best[best.s1.isin(set(val))]
            sc, _ = pd.factorize(best.s1.values)
            aq, as_ = assign_expected(np.arange(len(best)), sc, best.p.values.astype(float), p_unfound=0.02)
            kept = best.iloc[np.sort(aq)]
            mp = to_mapping(kept.s1.values, kept.q.values)
            f1 = fbeta_macro(mp, truth, val)
            f19 = fbeta_macro(mp, truth, val, fp_weight=1.9)
            ll = -np.mean(D.y.values * np.log(np.clip(p, 1e-6, 1)) + (1 - D.y.values) * np.log(np.clip(1 - p, 1e-6, 1)))
            print(f"w={w:.2f}: F0.5 {f1:.5f}  test-like {f19:.5f}  logloss {ll:.4f}")
    else:
        os.makedirs(a.out, exist_ok=True)
        for c in ("France", "India", "US"):
            d = pd.read_parquet(os.path.join(a.scores_dir, f"scores_{c}.parquet"))
            d = d.merge(ce.rename(columns={"s1_id": "s1", "q_id": "q"}), on=["s1", "q"], how="left")
            m = d.ce.notna().values
            p = d.p.values.astype(np.float64)
            p[m] = blend(p[m], d.ce.values[m], a.w)
            d["p"] = p.astype(np.float32)
            d.drop(columns=["ce"]).to_parquet(os.path.join(a.out, f"scores_{c}.parquet"), index=False)
            print(c, len(d), f"blended {m.mean():.3f}")


if __name__ == "__main__":
    main()
