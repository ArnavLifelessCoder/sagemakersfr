import zipfile, io, re, collections, random, pickle, sys, os
sys.stdout.reconfigure(encoding="utf-8")
Z = r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\6ab10eb3b23ba_student_resource.zip"
OUT = os.path.join(os.path.dirname(__file__), "vt.pkl")
z = zipfile.ZipFile(Z)


def lines(n):
    f = io.TextIOWrapper(z.open("student_resource/dataset/" + n), encoding="utf-8")
    next(f)
    return f


if not os.path.exists(OUT):
    pat = re.compile(r"(\bVT\b|Vermont|VERMONT)")
    recs = {}
    for n in ["train/train_source1.tsv", "train/train_source2.tsv", "train/train_source3.tsv"]:
        for l in lines(n):
            p = l.rstrip("\n").split("\t")
            if len(p) > 2 and pat.search(p[2]):
                recs[p[0]] = p[1:3]
    owner = {}
    for l in lines("train/train_ground_truth.tsv"):
        a, b = l.rstrip("\n").split("\t")
        for x in b.split(","):
            if x:
                owner[x] = a
    # also grab the full S1 records owning VT S2/S3 records (their S1 may not mention VT)
    need = {owner[k] for k in recs if k in owner}
    for l in lines("train/train_source1.tsv"):
        k = l.split("\t", 1)[0]
        if k in need and k not in recs:
            recs[k] = l.rstrip("\n").split("\t")[1:3]
    pickle.dump((recs, owner), open(OUT, "wb"))
recs, owner = pickle.load(open(OUT, "rb"))

s1 = [k for k in recs if k.startswith("S1")]
un = [k for k in recs if not k.startswith("S1") and k not in owner]
m = [k for k in recs if not k.startswith("S1") and k in owner]
print("VT S1", len(s1), "matched s2s3", len(m), "unmatched", len(un))

random.seed(0)


def key(a):
    return set(re.findall(r"\w+", a.lower()))


idx = collections.defaultdict(set)
for k in s1:
    for t in key(recs[k][1]) | key(recs[k][0]):
        idx[t].add(k)
for u in random.sample(un, 30):
    toks = key(recs[u][1]) | key(recs[u][0])
    sc = collections.Counter()
    for t in toks:
        if len(idx[t]) < 50:
            for k in idx[t]:
                sc[k] += 1
    print("UNM", u, recs[u])
    for k, c in sc.most_common(2):
        print("     ~", k, c, recs[k])
