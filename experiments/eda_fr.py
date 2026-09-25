import zipfile, io, re, collections, sys, os, pickle
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\code\business_entity_resolution")
from src.normalize import parse_name, parse_address
Z = r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\6ab10eb3b23ba_student_resource.zip"
OUT = os.path.join(os.path.dirname(__file__), "fr_pornic.pkl")
if not os.path.exists(OUT):
    z = zipfile.ZipFile(Z)
    recs = {}
    for n in ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"]:
        f = io.TextIOWrapper(z.open("student_resource/dataset/test/" + n), encoding="utf-8")
        next(f)
        for l in f:
            p = l.rstrip("\n").split("\t")
            if len(p) > 3 and p[3] == "France" and re.search(r"pornic", p[2], re.I):
                recs[p[0]] = p[1:3]
    pickle.dump(recs, open(OUT, "wb"))
recs = pickle.load(open(OUT, "rb"))
s1 = [k for k in recs if k.startswith("S1")]
print("Pornic records", len(recs), "S1", len(s1))
# group by (house number, first street token) -> show clusters with S1
grp = collections.defaultdict(list)
for k, (n, a) in recs.items():
    pa = parse_address(a, "France")
    st = pa["street"]
    key = st[1] if len(st) > 1 else (st[0] if st else "")
    grp[key].append((k, n, a, pa["hn"]))
shown = 0
for key, v in grp.items():
    if 3 <= len(v) <= 12 and any(x[0].startswith("S1") for x in v):
        print("=== street", key)
        for x in sorted(v, key=lambda t: t[0]):
            print("   ", x[0][:3], "|", x[1], "|", x[2])
        shown += 1
    if shown >= 14:
        break
# name token frequencies in France (legal forms etc.)
cnt = collections.Counter()
for k, (n, a) in recs.items():
    cnt.update(re.findall(r"[a-zà-ÿ]+", n.lower()))
print(cnt.most_common(80))
