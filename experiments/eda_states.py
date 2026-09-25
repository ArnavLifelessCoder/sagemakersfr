import zipfile, io, collections, sys, itertools, json, os
sys.stdout.reconfigure(encoding="utf-8")
Z = r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\6ab10eb3b23ba_student_resource.zip"
z = zipfile.ZipFile(Z)


def lines(n, lim):
    f = io.TextIOWrapper(z.open("student_resource/dataset/" + n), encoding="utf-8")
    next(f)
    return itertools.islice(f, lim)


res = {}
for n in ["train/train_source1.tsv", "train/train_source2.tsv", "train/train_source3.tsv",
          "test/test_source1.tsv", "test/test_source2.tsv", "test/test_source3.tsv"]:
    comp = collections.defaultdict(collections.Counter)
    for l in lines(n, 400000):
        p = l.rstrip("\n").split("\t")
        if len(p) < 4 or not p[2]:
            continue
        for c in p[2].split(","):
            c = c.strip()
            if c and not any(ch.isdigit() for ch in c) and len(c) <= 30:
                comp[p[3]][c] += 1
    for cty, cnt in comp.items():
        res.setdefault(cty, collections.Counter()).update(cnt)
out = {c: v.most_common(400) for c, v in res.items()}
json.dump(out, open(os.path.join(os.path.dirname(__file__), "components.json"), "w", encoding="utf-8"), ensure_ascii=False)
for c, v in out.items():
    print("=====", c)
    print(" | ".join(f"{k}:{n}" for k, n in v[:150]))
