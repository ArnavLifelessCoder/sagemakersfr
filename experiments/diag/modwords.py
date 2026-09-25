"""Token document-frequency in S1 vs S2/S3, train vs test: find test-only 'modifier' words."""
import io, sys, zipfile, collections, re, math, pickle, os
sys.path.insert(0, r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\code\business_entity_resolution")
sys.stdout.reconfigure(encoding="utf-8")
from src.normalize import fold

Z = r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\6ab10eb3b23ba_student_resource.zip"
HERE = os.path.dirname(os.path.abspath(__file__))
z = zipfile.ZipFile(Z)
TOK = re.compile(r"[a-z]+")


def rows(member):
    f = io.TextIOWrapper(z.open("student_resource/dataset/" + member), encoding="utf-8")
    next(f)
    for l in f:
        yield l.rstrip("\n").split("\t")


cache = os.path.join(HERE, "df.pkl")
if os.path.exists(cache):
    df, n = pickle.load(open(cache, "rb"))
else:
    df = collections.defaultdict(collections.Counter)  # (split, src, country) -> token df
    n = collections.Counter()
    for split in ("train", "test"):
        for k in (1, 2, 3):
            for p in rows(f"{split}/{split}_source{k}.tsv"):
                key = (split, 1 if k == 1 else 23, p[3])
                n[key] += 1
                df[key].update(set(TOK.findall(fold(p[1]))))
    pickle.dump((dict(df), n), open(cache, "wb"))

WORDS = ["motors", "stores", "agencies", "jewellers", "general", "bakery", "provision", "sweets", "pharmacy",
         "medicals", "automobiles", "traders", "hardware", "garments", "steel", "electronics", "furniture",
         "restaurant", "textiles", "eastgate", "southside", "downtown", "holdings", "group", "international",
         "developpement", "groupe", "holding", "trading", "technologies", "foundation"]
for c in ("India", "US", "France"):
    print(f"\n=== {c}: document frequency per 10k names")
    print(f"{'word':14s} {'trainS1':>8s} {'trainQ':>8s} {'testS1':>8s} {'testQ':>8s}")
    for w in WORDS:
        vals = []
        for key in (("train", 1, c), ("train", 23, c), ("test", 1, c), ("test", 23, c)):
            vals.append(1e4 * df.get(key, {}).get(w, 0) / max(1, n[key]))
        if max(vals) > 0.5:
            print(f"{w:14s} " + " ".join(f"{v:8.1f}" for v in vals))

# systematic search: tokens with high test-Q rate but ~absent from test S1 and train
print("\n=== tokens frequent in test S2/S3 but rare in test S1 (ratio), per country")
for c in ("India", "US", "France"):
    tq, ts, trq = df[("test", 23, c)], df[("test", 1, c)], df.get(("train", 23, c), {})
    nq, ns, ntrq = n[("test", 23, c)], n[("test", 1, c)], max(1, n[("train", 23, c)])
    out = []
    for w, v in tq.items():
        if v < 500:
            continue
        rq, rs = v / nq, (ts.get(w, 0) + 1) / ns
        out.append((rq / rs, w, 1e4 * rq, 1e4 * rs, 1e4 * trq.get(w, 0) / ntrq))
    out.sort(reverse=True)
    print(c, [(w, round(r, 1), round(a, 1), round(b, 1), round(t, 1)) for r, w, a, b, t in out[:30]])
