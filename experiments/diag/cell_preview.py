"""Per-S1 density of (house-number relation x same-name) cells among accepted-ish pairs (p>=0.5):
TEST (v5 seed-102 scores) vs labelled train-like reference. Excess on test = sibling contamination."""
import sys, os, collections
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\code\business_entity_resolution")
sys.stdout.reconfigure(encoding="utf-8")
from src.normalize import parse_name, parse_address

OLD = r"C:\Users\ARNAVG~1\AppData\Local\Temp\claude\C--Users-Arnav-Gawade-pro--OneDrive-Desktop-amazon-ml\61cd2e14-f061-440d-8f3f-a2caed65f90a\scratchpad"
T = r"C:\Users\ARNAVG~1\AppData\Local\Temp\claude\C--Users-Arnav-Gawade-pro--OneDrive-Desktop-amazon-ml\89303762-6110-432e-ae87-2d206a5ca21f\scratchpad\sr\student_resource\dataset\test"


def related(a, b):
    return a == b or a.startswith(b) or b.startswith(a) or a.endswith(b) or b.endswith(a)


def parse_chunk(rows):
    return [(e, frozenset(parse_name(n)["core"]), parse_address(a, c)["hn"] if a else "") for e, n, a, c in rows]


def load(dirpath, ids):
    rows = []
    for k in (1, 2, 3):
        with open(os.path.join(dirpath, f"test_source{k}.tsv"), encoding="utf-8") as f:
            next(f)
            for l in f:
                p = l.rstrip("\n").split("\t")
                if p[0] in ids:
                    rows.append((p[0], p[1], p[2], p[3]))
    chunks = [rows[i:i + 50000] for i in range(0, len(rows), 50000)]
    with Pool(6) as pool:
        res = pool.map(parse_chunk, chunks)
    return {e: (c, h) for part in res for e, c, h in part}


def cell_table(best, info, n_s1, ylab=None):
    out = collections.Counter()
    tru = collections.Counter()
    for i, (s, q) in enumerate(zip(best.s1.values, best.q.values)):
        cs, hs = info[s]
        cq, hq = info[q]
        hg = "missing" if not hq or not hs else ("same" if related(hq, hs) else "changed")
        nm = "same_name" if cs == cq and cs else "diff_name"
        band = "p>=0.99" if best.p.values[i] >= 0.99 else "0.5-0.99"
        k = (band, hg, nm)
        out[k] += 1
        if ylab is not None:
            tru[k] += ylab[i]
    return {k: v / n_s1 for k, v in out.items()}, {k: v / n_s1 for k, v in tru.items()}


if __name__ == "__main__":
    # reference (labelled)
    SL = os.path.join(OLD, "eslice")
    gt = pd.read_csv(os.path.join(SL, "gt.tsv"), sep="\t", dtype=str, keep_default_na=False)
    owner = {q: s for s, m in zip(gt.source1_entity_id, gt.matched_entity_ids) for q in m.split(",") if q}
    rp = pd.read_parquet(os.path.join(OLD, "eslice_out_ocr", "pair_scores.parquet"))
    rb = rp.sort_values("p", ascending=False).drop_duplicates("q")
    rb = rb[rb.p >= 0.5]
    rinfo = load(os.path.join(SL, "test"), set(rb.s1) | set(rb.q))
    s1r = pd.read_csv(os.path.join(SL, "test", "test_source1.tsv"), sep="\t", dtype=str, keep_default_na=False, quoting=3,
                      usecols=["entity_id", "country"])
    rc = dict(zip(s1r.entity_id, s1r.country))
    rb = rb.assign(c=rb.s1.map(rc))
    ref = {}
    for c in ("US", "India"):
        b = rb[rb.c == c]
        y = np.array([owner.get(q) == s for s, q in zip(b.s1, b.q)])
        ref[c] = cell_table(b, rinfo, (s1r.country == c).sum(), y)
    # test
    tp = pd.read_parquet(os.path.join(OLD, "v5s102", "pair_scores.parquet"))
    tb = tp.sort_values("p", ascending=False).drop_duplicates("q")
    tb = tb[tb.p >= 0.5]
    tinfo = load(T, set(tb.s1) | set(tb.q))
    ns = {"France": 259452, "India": 809986, "US": 663106}
    print(f"{'country':7s} {'cell':38s} {'test/S1':>8s} {'ref/S1':>8s} {'ref true/S1':>11s} {'est prec':>8s}")
    for c in ("US", "India", "France"):
        t, _ = cell_table(tb[tb.country == c], tinfo, ns[c])
        r_all, r_true = ref.get(c, ref["US"])
        for k in sorted(t):
            est = min(1.0, r_true.get(k, 0) / t[k]) if t[k] else float("nan")
            print(f"{c:7s} {str(k):38s} {t[k]:8.4f} {r_all.get(k, 0):8.4f} {r_true.get(k, 0):11.4f} {est:8.3f}")
