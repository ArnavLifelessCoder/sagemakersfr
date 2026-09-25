import os, sys, random, collections
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.join(os.path.dirname(__file__), "sr", "student_resource", "dataset", "test")
O = os.path.join(os.path.dirname(__file__), "kout", "output")


def rows(p):
    with open(p, encoding="utf-8") as f:
        next(f)
        for l in f:
            yield l.rstrip("\n").split("\t")


s1 = {}
for p in rows(f"{D}/test_source1.tsv"):
    s1[p[0]] = (p[1], p[2], p[3])
match = {p[0]: [x for x in p[1].split(",") if x] for p in rows(f"{O}/matching_results.tsv")}
cand_n = {p[0]: len([x for x in p[1].split(",") if x]) for p in rows(f"{O}/candidate_pairs.tsv")}

qc = collections.Counter()
qid_country = {}
for k in (2, 3):
    for p in rows(f"{D}/test_source{k}.tsv"):
        qc[p[3]] += 1
st = collections.defaultdict(lambda: collections.Counter())
for s, (n, a, c) in s1.items():
    m = match[s]
    st[c]["s1"] += 1
    st[c]["single"] += (len(m) == 0)
    st[c]["matches"] += len(m)
    st[c]["s2"] += sum(x.startswith("S2") for x in m)
    st[c]["s3"] += sum(x.startswith("S3") for x in m)
    st[c]["cands"] += cand_n[s]
    st[c]["big"] += (len(m) >= 8)
print(f"{'country':8s} {'S1':>9s} {'single%':>8s} {'match/S1':>9s} {'cand/S1':>8s} {'S2/S3 matched % of q':>22s} {'>=8 matches':>11s}")
for c, v in sorted(st.items()):
    print(f"{c:8s} {v['s1']:9d} {100*v['single']/v['s1']:8.2f} {v['matches']/v['s1']:9.3f} {v['cands']/v['s1']:8.3f} "
          f"{100*v['matches']/qc[c]:21.1f}% {v['big']:11d}")
print("train reference: single 5.6%, ~3.46 matches/S1, ~74% of S2/S3 matched")

# France spot-check
random.seed(1)
fr = [s for s, (n, a, c) in s1.items() if c == "France"]
samp_m = random.sample([s for s in fr if match[s]], 12)
samp_e = random.sample([s for s in fr if not match[s]], 6)
need = {x for s in samp_m for x in match[s]}
rec = {}
for k in (2, 3):
    for p in rows(f"{D}/test_source{k}.tsv"):
        if p[0] in need:
            rec[p[0]] = (p[1], p[2])
for s in samp_m:
    print("\nS1", s1[s][:2])
    for x in match[s]:
        print("   ", x[:2], rec.get(x))
print("\n--- France S1 predicted as singletons:")
for s in samp_e:
    print("   ", s1[s][:2], "cands:", cand_n[s])
