"""'Two camps' under one S1: camp A agrees with S1, camp B = >=2 records agreeing with EACH OTHER on something that
differs from S1 (house number, or one name word). Test sibling entities come as such sub-clusters.
Density per S1 test vs labelled reference, reference precision, estimated test precision = refTrue / test."""
import sys, os, collections
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\code\business_entity_resolution")
sys.stdout.reconfigure(encoding="utf-8")
from src.normalize import parse_name, parse_address

OLD = r"C:\Users\ARNAVG~1\AppData\Local\Temp\claude\C--Users-Arnav-Gawade-pro--OneDrive-Desktop-amazon-ml\61cd2e14-f061-440d-8f3f-a2caed65f90a\scratchpad"
T = r"C:\Users\ARNAVG~1\AppData\Local\Temp\claude\C--Users-Arnav-Gawade-pro--OneDrive-Desktop-amazon-ml\89303762-6110-432e-ae87-2d206a5ca21f\scratchpad\sr\student_resource\dataset\test"
S202 = r"C:\Users\ARNAVG~1\AppData\Local\Temp\claude\C--Users-Arnav-Gawade-pro--OneDrive-Desktop-amazon-ml\5cd43ae6-163c-4444-b6bc-8df70e443c7d\scratchpad\s202\pair_scores.parquet"
PMIN = 0.5
BANDS = [(0.5, 0.9), (0.9, 0.99), (0.99, 0.999), (0.999, 1.01)]


def related(a, b):
    return a == b or a.startswith(b) or b.startswith(a) or a.endswith(b) or b.endswith(a)


def parse_chunk(rows):
    return [(e, frozenset(parse_name(n)["core"]), parse_address(a, c)["hn"] if a else "") for e, n, a, c in rows]


def load(dirpath, ids, country):
    rows = []
    for k in (1, 2, 3):
        with open(os.path.join(dirpath, f"test_source{k}.tsv"), encoding="utf-8") as f:
            next(f)
            for l in f:
                p = l.rstrip("\n").split("\t")
                if p[3] == country and p[0] in ids:
                    rows.append((p[0], p[1], p[2], p[3]))
    chunks = [rows[i:i + 50000] for i in range(0, len(rows), 50000)]
    with Pool(6) as pool:
        res = pool.map(parse_chunk, chunks)
    return {e: (c, h) for part in res for e, c, h in part}


def one_word_swap(a, b):
    return len(a) == len(b) and len(a) >= 2 and len(a - b) == 1


def camps(sc, info, n_s1, owner=None):
    """sc: s1, q, p (all candidates, p >= PMIN). Returns per-S1 densities of camp-B pairs by (kind, band)."""
    sc = sc[sc.s1.isin(info.keys()) & sc.q.isin(info.keys())]
    best = sc.sort_values("p", ascending=False).drop_duplicates("q")
    grp = collections.defaultdict(list)
    for s, q, p in zip(best.s1.values, best.q.values, best.p.values):
        grp[s].append((q, p))
    cnt, tru = collections.Counter(), collections.Counter()
    for s, lst in grp.items():
        cs, hs = info[s]
        recs = [(q, p) + info[q] for q, p in lst]
        # camp A: strong record agreeing with S1 on name and number
        a_num = any(p >= 0.9 and cq == cs and hq and hs and related(hq, hs) for q, p, cq, hq in recs)
        a_name = any(p >= 0.9 and cq == cs for q, p, cq, hq in recs)
        for q, p, cq, hq in recs:
            kind = None
            if hs and hq and not related(hq, hs) and a_num:
                peers = sum(1 for q2, p2, c2, h2 in recs if q2 != q and h2 and related(h2, hq) and not related(h2, hs)
                            and (c2 == cq or c2 == cs))
                if peers:
                    kind = "num_" + ("same_core" if cq == cs else ("swap_core" if one_word_swap(cq, cs) else "other_core"))
            elif cq != cs and a_name and (not hq or not hs or related(hq, hs)):
                peers = sum(1 for q2, p2, c2, h2 in recs if q2 != q and c2 == cq)
                if one_word_swap(cq, cs):
                    kind = "name_swap" + ("_camp" if peers else "_single")
                elif cq > cs and len(cq - cs) == 1:
                    kind = "name_add" + ("_camp" if peers else "_single")
                elif cq < cs and len(cs - cq) == 1:
                    kind = "name_drop" + ("_camp" if peers else "_single")
            if kind is None or not kind.startswith("name"):
                continue
            band = next(b for b in BANDS if b[0] <= p < b[1])
            k = (kind, band)
            cnt[k] += 1
            if owner is not None:
                tru[k] += owner.get(q) == s
    return {k: v / n_s1 for k, v in cnt.items()}, {k: v / n_s1 for k, v in tru.items()}


if __name__ == "__main__":
    SL = os.path.join(OLD, "eslice")
    gt = pd.read_csv(os.path.join(SL, "gt.tsv"), sep="\t", dtype=str, keep_default_na=False)
    owner = {q: s for s, m in zip(gt.source1_entity_id, gt.matched_entity_ids) for q in m.split(",") if q}
    s1r = pd.read_csv(os.path.join(SL, "test", "test_source1.tsv"), sep="\t", dtype=str, keep_default_na=False,
                      quoting=3, usecols=["entity_id", "country"])
    rc = dict(zip(s1r.entity_id, s1r.country))
    rp = pd.read_parquet(os.path.join(OLD, "eslice_out_ocr", "pair_scores.parquet"))
    rp = rp[rp.p >= PMIN]
    rp = rp.assign(c=rp.s1.map(rc))
    ref = {}
    for c in ("US", "India"):
        sub = rp[rp.c == c]
        info = load(os.path.join(SL, "test"), set(sub.s1) | set(sub.q), c)
        ref[c] = camps(sub, info, (s1r.country == c).sum(), owner)
    del rp
    ns = {"France": 259452, "India": 809986, "US": 663106}
    for c in ("France", "US"):
        tp = pd.read_parquet(S202, filters=[("country", "=", c)], columns=["s1", "q", "p"])
        tp = tp[tp.p >= PMIN]
        info = load(T, set(tp.s1) | set(tp.q), c)
        t_all, _ = camps(tp, info, ns[c])
        del tp, info
        # France: no labelled reference -> pooled US+India reference rates
        if c in ref:
            r_all, r_tru = ref[c]
        else:
            r_all = {k: (ref["US"][0].get(k, 0) * 663106 + ref["India"][0].get(k, 0) * 809986) / 1473092
                     for k in set(ref["US"][0]) | set(ref["India"][0])}
            r_tru = {k: (ref["US"][1].get(k, 0) * 663106 + ref["India"][1].get(k, 0) * 809986) / 1473092
                     for k in set(ref["US"][1]) | set(ref["India"][1])}
        print(f"\n== {c}")
        print(f"{'kind':16s} {'band':14s} {'test/S1':>8s} {'ref/S1':>8s} {'refTrue/S1':>10s} {'ref prec':>8s} {'est test prec':>13s}")
        for k in sorted(set(t_all) | set(r_all)):
            t, r, rt = t_all.get(k, 0), r_all.get(k, 0), r_tru.get(k, 0)
            print(f"{k[0]:16s} {str(k[1]):14s} {t:8.4f} {r:8.4f} {rt:10.4f} {(rt / r if r else float('nan')):8.3f} "
                  f"{(min(1, rt / t) if t else float('nan')):13.3f}", flush=True)
