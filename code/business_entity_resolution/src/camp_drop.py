"""Drop sibling sub-clusters ('camp B', FINDINGS 5.16) from an existing matching_results.tsv, nothing else.

    python -m src.camp_drop --scores pair_scores.parquet --matching matching_results.tsv --test-dir DATASET/test
           --out out_dir [--camp US=0.99 France=0.99]

Test sibling entities are generated like real ones: a neighbouring business (house number +-1..20 or one name word
swapped) with 2-4 noisy records of its own. Under one S1, camp A agrees with S1 and camp B is >= 2 records agreeing
with each other on a house number (or name word) that differs from S1. In training this structure is rare and mostly
an outdated S1 address (true); in test it is ~24x over-represented in the US below p 0.99 (est. precision 2-9%).
"""
import argparse
import os

import pandas as pd

from .data import write_id_lists
from .postprocess import drop_camps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--matching", required=True)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--camp", nargs="+", default=["US=0.99", "France=0.99"])
    ap.add_argument("--camp-kinds", nargs="*", default=["num_same_core", "num_swap_core", "num_other_core",
                                                         "name_swap"])
    a = ap.parse_args()
    d = pd.read_parquet(a.scores)
    best = d.sort_values("p", ascending=False).drop_duplicates("q")
    del d
    m = pd.read_csv(a.matching, sep="\t", dtype=str, keep_default_na=False)
    pairs = pd.DataFrame([(s, q) for s, x in zip(m.iloc[:, 0], m.iloc[:, 1]) for q in x.split(",") if q],
                         columns=["s1", "q"])
    final = pairs.merge(best, on=["s1", "q"], how="left")
    miss = final.p.isna().sum()
    print(f"pairs in matching file: {len(final)}  (without a score, kept as is: {miss})")
    s1c = pd.read_csv(os.path.join(a.test_dir, "test_source1.tsv"), sep="\t", dtype=str, keep_default_na=False,
                      usecols=["entity_id", "country"], quoting=3)
    final["country"] = final.s1.map(dict(zip(s1c.entity_id, s1c.country)))
    final["p"] = final.p.fillna(1.0)
    kept = drop_camps(final, best[best.p.values >= 0.5], a.test_dir, dict(x.split("=") for x in a.camp),
                      set(a.camp_kinds))
    match = kept.groupby("s1", sort=False).q.apply(list).to_dict()
    os.makedirs(a.out, exist_ok=True)
    write_id_lists(os.path.join(a.out, "matching_results.tsv"), s1c.entity_id.values, match, "matched_entity_ids")
    stats = pd.DataFrame({"country": s1c.country, "n": [len(match.get(x, ())) for x in s1c.entity_id]})
    print(stats.groupby("country").n.agg(matches_per_S1="mean", singletons=lambda x: (x == 0).mean()).round(4))


if __name__ == "__main__":
    main()
