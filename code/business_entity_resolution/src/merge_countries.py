"""Take the matches of some countries from one submission and the rest from another.

    python -m src.merge_countries --base v8/matching_results.tsv --override v9/matching_results.tsv
           --countries France --test-dir DATASET/test --out out_dir
"""
import argparse
import os

import pandas as pd

from .data import write_id_lists


def read_matches(path):
    m = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return {s: [q for q in x.split(",") if q] for s, x in zip(m.iloc[:, 0], m.iloc[:, 1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--override", required=True)
    ap.add_argument("--countries", nargs="+", required=True)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s1 = pd.read_csv(os.path.join(a.test_dir, "test_source1.tsv"), sep="\t", dtype=str, keep_default_na=False,
                     usecols=["entity_id", "country"], quoting=3)
    ctry = dict(zip(s1.entity_id, s1.country))
    base, over = read_matches(a.base), read_matches(a.override)
    take = set(a.countries)
    n_over = sum(1 for s in over if ctry.get(s) in take and over[s])
    assert n_over > 0, "override file has no matches in the requested countries"
    match = {s: (over.get(s, []) if ctry[s] in take else base.get(s, [])) for s in s1.entity_id}
    # a record may only be claimed once: drop override records already claimed by a base-country S1
    claimed = {q for s, l in match.items() if ctry[s] not in take for q in l}
    dup = 0
    for s, l in match.items():
        if ctry[s] in take:
            k = [q for q in l if q not in claimed]
            dup += len(l) - len(k)
            match[s] = k
    os.makedirs(a.out, exist_ok=True)
    write_id_lists(os.path.join(a.out, "matching_results.tsv"), s1.entity_id.values,
                   {s: l for s, l in match.items() if l}, "matched_entity_ids")
    st = pd.DataFrame({"country": s1.country, "n": [len(match[s]) for s in s1.entity_id]})
    print(f"countries from override: {sorted(take)}; records dropped as already claimed: {dup}")
    print(st.groupby("country").n.agg(matches_per_S1="mean", singletons=lambda x: (x == 0).mean()).round(4))


if __name__ == "__main__":
    main()
