"""Accepted pairs by (name relation x address agreement): test (v8 output, per country) vs TRUE pairs of the labelled
reference (gt of eslice, US+India). Excess of test accepted over reference true = estimated false matches."""
import os, sys, collections
from multiprocessing import Pool
import pandas as pd
sys.path.insert(0, r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\code\business_entity_resolution")
sys.stdout.reconfigure(encoding="utf-8")
from src.normalize import parse_name, parse_address
T = r"C:\Users\ARNAVG~1\AppData\Local\Temp\claude\C--Users-Arnav-Gawade-pro--OneDrive-Desktop-amazon-ml\89303762-6110-432e-ae87-2d206a5ca21f\scratchpad\sr\student_resource\dataset\test"
OLD = r"C:\Users\ARNAVG~1\AppData\Local\Temp\claude\C--Users-Arnav-Gawade-pro--OneDrive-Desktop-amazon-ml\61cd2e14-f061-440d-8f3f-a2caed65f90a\scratchpad"
TYPES = {"rue", "st", "ave", "rd", "dr", "blvd", "ln", "all", "ch", "imp", "pl", "ct", "cir", "trl", "pkwy", "hwy", "ter",
         "sq", "rte", "way", "road", "street", "avenue", "de", "du", "des", "la", "le", "les", "d", "l"}


def related(a, b):
    return a == b or a.startswith(b) or b.startswith(a) or a.endswith(b) or b.endswith(a)


def parse(rows):
    out = []
    for e, n, a, c in rows:
        pa = parse_address(a, c) if a.strip() else None
        out.append((e, frozenset(parse_name(n)["core"]), (pa["hn"], frozenset(pa["street"]) - TYPES) if pa else None))
    return out


def load(d, ids):
    rows = []
    for k in (1, 2, 3):
        with open(os.path.join(d, f"test_source{k}.tsv"), encoding="utf-8") as f:
            next(f)
            for l in f:
                p = l.rstrip("\n").split("\t")
                if p[0] in ids:
                    rows.append((p[0], p[1], p[2], p[3]))
    with Pool(6) as pool:
        res = pool.map(parse, [rows[i:i + 40000] for i in range(0, len(rows), 40000)])
    return {e: (c, a) for part in res for e, c, a in part}


def cell(i_s, i_q):
    cs, as_ = i_s
    cq, aq = i_q
    name = "same_name" if cs == cq and cs else ("overlap" if cs & cq else "no_overlap")
    if aq is None:
        addr = "q_empty"
    elif as_ is None:
        addr = "s_empty"
    else:
        (hs, ss), (hq, sq) = as_, aq
        num = "num_same" if hs and hq and related(hs, hq) else ("num_diff" if hs and hq else "num_missing")
        addr = num + ("+street" if ss & sq else "-street")
    return name, addr


def table(pairs, info, n1):
    cnt = collections.Counter()
    for s, q in pairs:
        if s in info and q in info:
            cnt[cell(info[s], info[q])] += 1
    return {k: v / n1 for k, v in cnt.items()}


if __name__ == "__main__":
    SL = os.path.join(OLD, "eslice")
    gt = pd.read_csv(os.path.join(SL, "gt.tsv"), sep="\t", dtype=str, keep_default_na=False)
    rpairs = [(s, q) for s, m in zip(gt.source1_entity_id, gt.matched_entity_ids) for q in m.split(",") if q]
    rinfo = load(os.path.join(SL, "test"), {x for p in rpairs for x in p})
    ref = table(rpairs, rinfo, len(gt))
    m = pd.read_csv("s7/matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
    s1c = pd.read_csv(os.path.join(T, "test_source1.tsv"), sep="\t", dtype=str, keep_default_na=False, quoting=3,
                      usecols=["entity_id", "country"])
    ctry = dict(zip(s1c.entity_id, s1c.country))
    n1 = collections.Counter(s1c.country)
    out = {}
    for c in ("France", "US", "India"):
        tp = [(s, q) for s, x in zip(m.iloc[:, 0], m.iloc[:, 1]) if ctry[s] == c for q in x.split(",") if q]
        info = load(T, {x for p in tp for x in p})
        out[c] = table(tp, info, n1[c])
        del info
    keys = sorted(set(ref) | {k for c in out for k in out[c]})
    print(f"{'name':11s} {'address':20s} {'refTRUE':>8s} {'US acc':>8s} {'IN acc':>8s} {'FR acc':>8s}   (per S1)")
    for k in keys:
        print(f"{k[0]:11s} {k[1]:20s} {ref.get(k, 0):8.4f} {out['US'].get(k, 0):8.4f} {out['India'].get(k, 0):8.4f} "
              f"{out['France'].get(k, 0):8.4f}")
