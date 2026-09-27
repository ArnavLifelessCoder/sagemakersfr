"""Export text pairs for a neural cross-encoder.

Training pairs come from a training dump (pipeline train --dump DIR: every stage-1 survivor with its label);
test pairs come from a predict run's pair_scores.parquet (every scored pair). Each row holds the two texts
"name | address | country" and, for training, the label and the validation flag.

    python -m src.export_pairs train --dump DUMP_DIR --data-dir DATASET_DIR --out pairs_train.parquet [--max-neg-per-pos 2]
    python -m src.export_pairs test  --scores OUT_DIR/pair_scores.parquet --data-dir DATASET_DIR --out pairs_test.parquet
"""
import argparse
import os

import numpy as np
import pandas as pd

from .data import read_sources


def _texts(src):
    t = (src.business_name.fillna("") + " | " + src.business_address.fillna("") + " | " + src.country.fillna(""))
    return dict(zip(src.entity_id.values, t.values))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "test"])
    ap.add_argument("--dump", default=None)
    ap.add_argument("--scores", default=None)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-neg-per-pos", type=float, default=3.0, help="train: cap on negatives per positive (hardest kept)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.mode == "train":
        D = pd.read_parquet(os.path.join(a.dump, "pairs.parquet"), columns=["s1_id", "q_id", "y", "is_val", "p2", "p1"])
        src = read_sources(a.data_dir, "train")
    else:
        D = pd.read_parquet(a.scores)
        D = D.rename(columns={"s1": "s1_id", "q": "q_id"})
        src = read_sources(a.data_dir, "test")
    tx = _texts(src)
    del src
    D["text_a"] = D.s1_id.map(tx)
    D["text_b"] = D.q_id.map(tx)
    if a.mode == "train" and a.max_neg_per_pos:
        pos = D[D.y == 1]
        neg = D[D.y == 0].sort_values("p2", ascending=False)  # hardest negatives first
        n_keep = int(len(pos) * a.max_neg_per_pos)
        neg = neg.iloc[:n_keep]
        D = pd.concat([pos, neg], ignore_index=True).sample(frac=1.0, random_state=a.seed).reset_index(drop=True)
    D.to_parquet(a.out, index=False)
    print("wrote", a.out, len(D), "rows", D.columns.tolist())


if __name__ == "__main__":
    main()
