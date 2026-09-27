"""US 'changed number + same core name' pairs: split by probability band, legal-form change and |number delta|,
test (v5 seed-102) vs labelled train-like reference -> estimated test precision per sub-cell."""
import sys, os, collections
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\code\business_entity_resolution")
sys.stdout.reconfigure(encoding="utf-8")
from src.normalize import parse_name, parse_address

OLD = r"C:\Users\ARNAVG~1\AppData\Local\Temp\claude\C--Users-Arnav-Gawade-pro--OneDrive-Desktop-amazon-ml\61cd2e14-f061-440d-8f3f-a2caed65f90a\scratchpad"
T = r"C:\Users\ARNAVG~1\AppData\Local\Temp\claude\C--Users-Arnav-Gawade-pro--OneDrive-Desktop-amazon-ml\89303762-6110-432e-ae87-2d206a5ca21f\scratchpad\sr\student_resource\dataset\test"
BANDS = [(0.5, 0.9), (0.9, 0.96), (0.96, 0.99), (0.99, 0.999), (0.999, 1.01)]


def related(a, b):
    return a == b or a.startswith(b) or b.startswith(a) or a.endswith(b) or b.endswith(a)


def parse_chunk(rows):
    out = []
    for e, n, a, c in rows:
        pn = parse_name(n)
        out.append((e, frozenset(pn["core"]), frozenset(pn["legal"]), parse_address(a, c)["hn"] if a else ""))
    return out


def load(dirpath, ids):
    rows = []
    for k in (1, 2, 3):
        with open(os.path.join(dirpath, f"test_source{k}.tsv"), encoding="utf-8") as f:
            next(f)
            for l in f:
                p = l.rstrip("\n").split("\t")
                if p[0] in ids and p[3] == "US":
                    rows.append((p[0], p[1], p[2], p[3]))
    chunks = [rows[i:i + 50000] for i in range(0, len(rows), 50000)]
    with Pool(6) as pool:
        res = pool.map(parse_chunk, chunks)
    return {e: (c, l, h) for part in res for e, c, l, h in part}


def table(best, info, n_s1, y=None):
    cnt, tru = collections.Counter(), collections.Counter()
    for i, (s, q, p) in enumerate(zip(best.s1.values, best.q.values, best.p.values)):
        if s not in info or q not in info:
            continue
        cs, ls, hs = info[s]
        cq, lq, hq = info[q]
        if not (hs and hq) or related(hs, hq) or cs != cq or not cs:
            continue
        band = next(b for b in BANDS if b[0] <= p < b[1])
        try:
            d = abs(int(hs) - int(hq))
        except ValueError:
            d = 10 ** 9
        k = (band, "legal_changed" if ls != lq else "legal_same", "delta<=20" if d <= 20 else "delta>20")
        cnt[k] += 1
        if y is not None:
            tru[k] += y[i]
    return {k: v / n_s1 for k, v in cnt.items()}, {k: v / n_s1 for k, v in tru.items()}


if __name__ == "__main__":
    SL = os.path.join(OLD, "eslice")
    gt = pd.read_csv(os.path.join(SL, "gt.tsv"), sep="\t", dtype=str, keep_default_na=False)
    owner = {q: s for s, m in zip(gt.source1_entity_id, gt.matched_entity_ids) for q in m.split(",") if q}
    rp = pd.read_parquet(os.path.join(OLD, "eslice_out_ocr", "pair_scores.parquet"))
    rb = rp.sort_values("p", ascending=False).drop_duplicates("q")
    rb = rb[rb.p >= 0.5]
    rinfo = load(os.path.join(SL, "test"), set(rb.s1) | set(rb.q))
    s1r = pd.read_csv(os.path.join(SL, "test", "test_source1.tsv"), sep="\t", dtype=str, keep_default_na=False,
                      quoting=3, usecols=["entity_id", "country"])
    y = np.array([owner.get(q) == s for s, q in zip(rb.s1, rb.q)])
    r_all, r_tru = table(rb, rinfo, (s1r.country == "US").sum(), y)
    tp = pd.read_parquet(os.path.join(OLD, "v5s102", "pair_scores.parquet"), filters=[("country", "=", "US")])
    tb = tp.sort_values("p", ascending=False).drop_duplicates("q")
    tb = tb[tb.p >= 0.5]
    tinfo = load(T, set(tb.s1) | set(tb.q))
    t_all, _ = table(tb, tinfo, 663106)
    print(f"{'band':14s} {'legal':14s} {'delta':10s} {'test/S1':>8s} {'ref/S1':>8s} {'refTrue/S1':>10s} {'est prec':>8s}")
    for k in sorted(set(t_all) | set(r_all)):
        t = t_all.get(k, 0)
        est = min(1.0, r_tru.get(k, 0) / t) if t else float("nan")
        print(f"{str(k[0]):14s} {k[1]:14s} {k[2]:10s} {t:8.4f} {r_all.get(k, 0):8.4f} {r_tru.get(k, 0):10.4f} {est:8.3f}")
