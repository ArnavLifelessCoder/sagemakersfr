"""TRAIN: for every true (S1, S2/S3) pair, is the detected state of the S2/S3 record the same as the S1's?
Streams the zip; memory ~1 GB."""
import io, sys, zipfile, collections, pickle, os
sys.path.insert(0, r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\code\business_entity_resolution")
sys.stdout.reconfigure(encoding="utf-8")
from src.make_dev_subset import state_of

Z = r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\6ab10eb3b23ba_student_resource.zip"
z = zipfile.ZipFile(Z)


def rows(m):
    f = io.TextIOWrapper(z.open("student_resource/dataset/" + m), encoding="utf-8")
    next(f)
    for l in f:
        yield l.rstrip("\n").split("\t")


codes = {}
def code(s):
    return codes.setdefault(s, len(codes))

s1st = {}
for p in rows("train/train_source1.tsv"):
    s1st[p[0]] = code(state_of(p[2].replace('"', ''), p[3]) if p[2] else "")
own = {}
for p in rows("train/train_ground_truth.tsv"):
    if len(p) > 1 and p[1]:
        c = s1st[p[0]]
        for x in p[1].split(","):
            own[x] = c
inv = {v: k for k, v in codes.items()}
tab = collections.Counter()
for k in (2, 3):
    for p in rows(f"train/train_source{k}.tsv"):
        c = own.get(p[0])
        if c is None:
            continue
        qs = state_of(p[2].replace('"', ''), p[3]) if p[2] else "EMPTY"
        tab[(inv[c], qs)] += 1
pickle.dump(tab, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "state_shift.pkl"), "wb"))
per = collections.defaultdict(collections.Counter)
for (a, b), n in tab.items():
    per[a]["total"] += n
    per[a]["same" if a == b else "empty" if b == "EMPTY" else "nostate" if b == "" else "other"] += n
tot = collections.Counter()
for a, v in per.items():
    tot.update(v)
print("ALL:", {k: f"{v / tot['total']:.4f}" for k, v in tot.items()})
print("\nstates with the most true pairs landing in ANOTHER state:")
rows_ = sorted(per.items(), key=lambda kv: -kv[1]["other"])[:25]
for a, v in rows_:
    top = sorted(((n, b) for (x, b), n in tab.items() if x == a and b not in (a, "EMPTY", "")), reverse=True)[:4]
    print(f"  {a:8s} total {v['total']:8d}  other {v['other'] / v['total']:.4f}  nostate {v['nostate'] / v['total']:.4f}  -> {top}")
