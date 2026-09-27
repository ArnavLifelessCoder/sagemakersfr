"""Do test sibling distractors come as their OWN small clusters (tree/sub-cluster structure)?
For accepted-ish US pairs with the sibling signature (same core name, house number changed, legal form changed),
count 'peers': other candidates of the same S1 that share q's new house number and core name.
Test (seed-202 scores) vs labelled train-like reference (eslice). Per-S1 densities + reference precision."""
import sys, os, collections
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\code\business_entity_resolution")
sys.stdout.reconfigure(encoding="utf-8")
from src.normalize import parse_name, parse_address

OLD = r"C:\Users\ARNAVG~1\AppData\Local\Temp\claude\C--Users-Arnav-Gawade-pro--OneDrive-Desktop-amazon-ml\61cd2e14-f061-440d-8f3f-a2caed65f90a\scratchpad"
T = r"C:\Users\ARNAVG~1\AppData\Local\Temp\claude\C--Users-Arnav-Gawade-pro--OneDrive-Desktop-amazon-ml\89303762-6110-432e-ae87-2d206a5ca21f\scratchpad\sr\student_resource\dataset\test"
S202 = r"C:\Users\ARNAVG~1\AppData\Local\Temp\claude\C--Users-Arnav-Gawade-pro--OneDrive-Desktop-amazon-ml\5cd43ae6-163c-4444-b6bc-8df70e443c7d\scratchpad\s202\pair_scores.parquet"
PEER_MIN = 0.3


def related(a, b):
    return a == b or a.startswith(b) or b.startswith(a) or a.endswith(b) or b.endswith(a)


def parse_chunk(rows):
    out = []
    for e, n, a, c in rows:
        pn = parse_name(n)
        out.append((e, frozenset(pn["core"]), frozenset(pn["legal"]), parse_address(a, c)["hn"] if a else "",
                    e.split("-")[0]))
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
    with Pool(4) as pool:
        res = pool.map(parse_chunk, chunks)
    return {e: v for part in res for e, *v in part}


def analyse(sc, info, n_s1, owner=None):
    sc = sc[sc.s1.isin(info.keys()) & sc.q.isin(info.keys())]
    best = sc.sort_values("p", ascending=False).drop_duplicates("q")
    best = best[best.p >= 0.5]
    peers = sc[sc.p >= PEER_MIN]
    by_s = collections.defaultdict(list)
    for s, q, p in zip(peers.s1.values, peers.q.values, peers.p.values):
        by_s[s].append((q, p))
    cnt, tru = collections.Counter(), collections.Counter()
    srcmix = collections.Counter()
    for s, q, p in zip(best.s1.values, best.q.values, best.p.values):
        cs, ls, hs, _ = info[s]
        cq, lq, hq, srq = info[q]
        if not (hs and hq) or related(hs, hq) or cs != cq or not cs or ls == lq:
            continue
        npeer, nsame_legal, other_src = 0, 0, 0
        for q2, p2 in by_s[s]:
            if q2 == q:
                continue
            c2, l2, h2, sr2 = info[q2]
            if c2 == cq and h2 and related(h2, hq) and not related(h2, hs):
                npeer += 1
                nsame_legal += (l2 == lq)
                other_src += (sr2 != srq)
        band = "0.5-0.9" if p < 0.9 else ("0.9-0.99" if p < 0.99 else ">=0.99")
        k = (band, "peers=0" if npeer == 0 else ("peers=1" if npeer == 1 else "peers>=2"),
             "" if npeer == 0 else ("peer_same_legal" if nsame_legal else "peer_diff_legal"))
        cnt[k] += 1
        if npeer:
            srcmix[("other_src" if other_src else "same_src_only")] += 1
        if owner is not None:
            tru[k] += owner.get(q) == s
    return {k: v / n_s1 for k, v in cnt.items()}, {k: v / n_s1 for k, v in tru.items()}, srcmix


if __name__ == "__main__":
    SL = os.path.join(OLD, "eslice")
    gt = pd.read_csv(os.path.join(SL, "gt.tsv"), sep="\t", dtype=str, keep_default_na=False)
    owner = {q: s for s, m in zip(gt.source1_entity_id, gt.matched_entity_ids) for q in m.split(",") if q}
    s1r = pd.read_csv(os.path.join(SL, "test", "test_source1.tsv"), sep="\t", dtype=str, keep_default_na=False,
                      quoting=3, usecols=["entity_id", "country"])
    rp = pd.read_parquet(os.path.join(OLD, "eslice_out_ocr", "pair_scores.parquet"))
    rp = rp[rp.p >= PEER_MIN]
    rinfo = load(os.path.join(SL, "test"), set(rp.s1) | set(rp.q))
    r_all, r_tru, r_mix = analyse(rp, rinfo, (s1r.country == "US").sum(), owner)
    del rp
    tp = pd.read_parquet(S202, filters=[("country", "=", "US")], columns=["s1", "q", "p"])
    tp = tp[tp.p >= PEER_MIN]
    tinfo = load(T, set(tp.s1) | set(tp.q))
    t_all, _, t_mix = analyse(tp, tinfo, 663106)
    print(f"{'band':9s} {'peers':9s} {'peer legal':16s} {'test/S1':>8s} {'ref/S1':>8s} {'refTrue/S1':>10s} "
          f"{'ref prec':>8s} {'est test prec':>13s}")
    for k in sorted(set(t_all) | set(r_all)):
        t, r, rt = t_all.get(k, 0), r_all.get(k, 0), r_tru.get(k, 0)
        print(f"{k[0]:9s} {k[1]:9s} {k[2]:16s} {t:8.4f} {r:8.4f} {rt:10.4f} {(rt / r if r else float('nan')):8.3f} "
              f"{(min(1, rt / t) if t else float('nan')):13.3f}")
    print("peer source mix  test:", dict(t_mix), " ref:", dict(r_mix))
