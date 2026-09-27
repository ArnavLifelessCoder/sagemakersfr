"""Create a small regional slice of the TRAIN data for fast local iteration.

Keeps every record (all 3 sources) located in a handful of states, all ground-truth
matches of the selected S1 entities, and a proportional sample of empty-address
unmatched records, so distractor density is realistic. Streams the files, so it
runs in little memory.

    python -m src.make_dev_subset --zip ../../6ab10eb3b23ba_student_resource.zip --out DEV_DIR
"""
import argparse
import os
import random

import pandas as pd

from .data import COLS, _open
from .normalize import STATE_MAP, STATE_ANY, fold, is_native

DEFAULT_STATES = {"US-VT", "US-ME", "US-NM", "US-MT", "US-DE", "US-ND", "IN-KL", "IN-PB", "IN-OD"}


def state_of(addr, country):
    for c in reversed(addr.split(",")):
        c = c.strip()
        if not c:
            continue
        key = c if is_native(c) else fold(c).strip(" .")
        st = STATE_MAP.get((country, key)) or (STATE_ANY.get(key) if len(key) > 3 else None)
        if st:
            return st
    return ""


def rows(zip_path, member):
    with _open(zip_path, member) as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 4:
                yield p[:4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--states", default=",".join(sorted(DEFAULT_STATES)))
    a = ap.parse_args()
    states = set(a.states.split(","))
    os.makedirs(a.out, exist_ok=True)

    s1_rows, n_s1 = [], 0
    for p in rows(a.zip, "train_source1.tsv"):
        n_s1 += 1
        if state_of(p[2], p[3]) in states:
            s1_rows.append(p)
    s1_sel = {p[0] for p in s1_rows}
    frac = len(s1_sel) / n_s1
    print("S1 selected", len(s1_sel), "frac %.4f" % frac)

    gt_rows, matched_all, keep_other = [], set(), set()
    for p in rows_gt(a.zip):
        ids = [x for x in p[1].split(",") if x]
        matched_all.update(ids)
        if p[0] in s1_sel:
            gt_rows.append(p)
            keep_other.update(ids)

    rnd = random.Random(0)
    for k in (2, 3):
        out = []
        for p in rows(a.zip, f"train_source{k}.tsv"):
            eid = p[0]
            if eid in keep_other:
                out.append(p)
            elif eid not in matched_all:
                if p[2].strip() == "":
                    if rnd.random() < frac:
                        out.append(p)
                elif state_of(p[2], p[3]) in states:
                    out.append(p)
        pd.DataFrame(out, columns=COLS).to_parquet(os.path.join(a.out, f"src{k}.parquet"), index=False)
        print("source", k, len(out))
    pd.DataFrame(s1_rows, columns=COLS).to_parquet(os.path.join(a.out, "src1.parquet"), index=False)
    g = pd.DataFrame(gt_rows, columns=["source1_entity_id", "matched_entity_ids"])
    g.to_parquet(os.path.join(a.out, "gt.parquet"), index=False)
    print("gt rows", len(g), "singletons", (g.matched_entity_ids == "").sum())


def rows_gt(zip_path):
    with _open(zip_path, "train_ground_truth.tsv") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            yield (p[0], p[1] if len(p) > 1 else "")


if __name__ == "__main__":
    main()
