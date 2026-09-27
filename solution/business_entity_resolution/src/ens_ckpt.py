"""Average per-country score checkpoints of several runs into a new checkpoint dir (for src.assemble / blend_ce).

    python -m src.ens_ckpt --runs DIR1 DIR2 [--weights 1 1] --out ENS_DIR

A pair scored by only some runs keeps the mean over the runs that scored it (a run's blocking difference is
mostly extra recall, so no penalty is applied).
"""
import argparse
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    w = np.array(a.weights or [1.0] * len(a.runs), dtype=float)
    os.makedirs(a.out, exist_ok=True)
    for c in ("France", "India", "US"):
        parts = []
        for k, r in enumerate(a.runs):
            d = pd.read_parquet(os.path.join(r, f"scores_{c}.parquet"), columns=["s1", "q", "p"])
            d["wp"] = d.p.values * w[k]
            d["w"] = w[k]
            parts.append(d[["s1", "q", "wp", "w"]])
        allp = pd.concat(parts, ignore_index=True).groupby(["s1", "q"], sort=False)[["wp", "w"]].sum().reset_index()
        allp["p"] = (allp.wp / allp.w).astype(np.float32)
        allp["country"] = c
        allp[["s1", "q", "p", "country"]].to_parquet(os.path.join(a.out, f"scores_{c}.parquet"), index=False)
        n_all = len(allp)
        n_each = [len(p) for p in parts]
        print(f"{c}: runs {n_each} -> union {n_all}")


if __name__ == "__main__":
    main()
