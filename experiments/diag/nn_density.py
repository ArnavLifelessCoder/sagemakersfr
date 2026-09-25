"""Near-duplicate neighbour density per S1: train vs test, same states."""
import io, re, sys, zipfile, collections, random, pickle, os
sys.path.insert(0, r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\code\business_entity_resolution")
sys.stdout.reconfigure(encoding="utf-8")
from src.normalize import parse_name, parse_address

Z = r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\6ab10eb3b23ba_student_resource.zip"
HERE = os.path.dirname(os.path.abspath(__file__))
PAT = re.compile(r"(Kerala|Keralam|\bKL\b|കേരളം|\bVT\b|Vermont|VERMONT)")
z = zipfile.ZipFile(Z)


def rows(member):
    f = io.TextIOWrapper(z.open("student_resource/dataset/" + member), encoding="utf-8")
    next(f)
    for l in f:
        yield l.rstrip("\n").split("\t")


cache = os.path.join(HERE, "nn.pkl")
if os.path.exists(cache):
    data = pickle.load(open(cache, "rb"))
else:
    data = {}
    for split in ("train", "test"):
        recs = {}
        for k in (1, 2, 3):
            for p in rows(f"{split}/{split}_source{k}.tsv"):
                if len(p) > 3 and PAT.search(p[2]):
                    recs[p[0]] = (p[1], p[2], p[3])
        data[split] = recs
    owner = {}
    for p in rows("train/train_ground_truth.tsv"):
        for x in (p[1].split(",") if len(p) > 1 else []):
            if x:
                owner[x] = p[0]
    data["owner"] = owner
    pickle.dump(data, open(cache, "wb"))

owner = data["owner"]
for split in ("train", "test"):
    recs = data[split]
    parsed = {}
    for e, (n, a, c) in recs.items():
        pn = parse_name(n)
        pa = parse_address(a, c)
        parsed[e] = (set(pn["core"]), pn["core"][:1], c, set(pa["nums"]), pn["native"])
    s1 = [e for e in recs if e.startswith("S1")]
    idx = collections.defaultdict(list)
    for e, v in parsed.items():
        if not e.startswith("S1") and v[1]:
            idx[(v[2], v[1][0])].append(e)
    for country in ("US", "India"):
        per = []
        per_true = []
        examples = []
        for s in s1:
            toks, first, c, nums, _ = parsed[s]
            if c != country or not first:
                continue
            nn = []
            for e in idx[(c, first[0])]:
                t2 = parsed[e][0]
                if len(toks & t2) / max(1, len(toks | t2)) >= 0.5:
                    nn.append(e)
            per.append(len(nn))
            if split == "train":
                per_true.append(sum(owner.get(e) == s for e in nn))
                if len(examples) < 6:
                    bad = [e for e in nn if owner.get(e) != s]
                    if bad:
                        examples.append((recs[s][:2], [recs[e][:2] for e in bad[:3]]))
        if not per:
            continue
        msg = f"{split:5s} {country:6s} S1={len(per):6d}  near-dup neighbours/S1 = {sum(per) / len(per):.3f}"
        if per_true:
            msg += f"  (true {sum(per_true) / len(per):.3f}, non-match {(sum(per) - sum(per_true)) / len(per):.3f})"
        print(msg)
        for s, bad in examples[:4]:
            print("      S1", s, "\n        non-match neighbours:", bad)
