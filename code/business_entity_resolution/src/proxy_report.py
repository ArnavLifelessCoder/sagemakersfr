"""Label-free quality proxies for a matching_results.tsv on the TEST set (streams, low memory).

    python -m src.proxy_report --data-dir DATASET_DIR --matching output/matching_results.tsv

Reports per country: matches per S1 and singleton share (training prior: ~3.46 and 5.6% in every
country), and the share of accepted matches whose S2/S3 name carries a *modifier* word absent from
the S1 name, where modifier = token >= 20x more frequent in S2/S3 names than in S1 names of that
country (e.g. eastgate, holding, groupe, bakery). Those are almost always sibling distractors, so the
share is a lower-bound estimate of the false-merge rate.
"""
import argparse
import collections
import os

from .normalize import parse_name
from .prep import extend_translit


def rows(path):
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 4:
                yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--matching", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--ratio", type=float, default=20.0)
    a = ap.parse_args()
    d = os.path.join(a.data_dir, a.split)
    files = [os.path.join(d, f"{a.split}_source{k}.tsv") for k in (1, 2, 3)]
    # pass 0: transliteration fallback vocabulary (Latin names + unknown native tokens)
    tr = extend_translit({}, (p[1] for f in files for p in rows(f)))
    # matched ids
    match = {}
    for p in rows_match(a.matching):
        match[p[0]] = p[1]
    needed = {x for v in match.values() for x in v}
    # pass 1: document frequencies + tokens of needed records only
    df = collections.defaultdict(collections.Counter)
    n = collections.Counter()
    toks, country = {}, {}
    for k, f in enumerate(files, 1):
        for e, name, _, c in rows(f):
            t = frozenset(parse_name(name, tr)["core"])
            key = (c, 1 if k == 1 else 23)
            n[key] += 1
            df[key].update(t)
            if k == 1:
                country[e] = c
                toks[e] = t
            elif e in needed:
                toks[e] = t
    mod = {}
    for c in {k[0] for k in n}:
        s1df, qdf = df[(c, 1)], df[(c, 23)]
        n1, nq = n[(c, 1)], n[(c, 23)]
        mod[c] = {t for t, v in qdf.items() if v >= 50 and (v / nq) / ((s1df.get(t, 0) + 1) / n1) >= a.ratio}
    st = collections.defaultdict(collections.Counter)
    for s, ids in match.items():
        c = country[s]
        st[c]["s1"] += 1
        st[c]["single"] += not ids
        st[c]["m"] += len(ids)
        for x in ids:
            st[c]["mod"] += bool((toks[x] - toks[s]) & mod[c])
    print(f"{'country':8s} {'S1':>9s} {'match/S1':>9s} {'single%':>8s} {'modifier-FP%':>13s}   (prior 3.46 / 5.6%)")
    for c, v in sorted(st.items()):
        print(f"{c:8s} {v['s1']:9d} {v['m'] / v['s1']:9.3f} {100 * v['single'] / v['s1']:8.2f} "
              f"{100 * v['mod'] / max(1, v['m']):12.2f}%")
    for c in sorted(mod):
        print(c, "modifier words:", sorted(mod[c])[:40], "..." if len(mod[c]) > 40 else "")


def rows_match(path):
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            s, _, rest = line.rstrip("\n").partition("\t")
            yield s, [x for x in rest.split(",") if x]


if __name__ == "__main__":
    main()
