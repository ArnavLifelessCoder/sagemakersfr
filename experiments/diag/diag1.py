"""Train vs test structural comparison + shape of our v1 test predictions."""
import collections, io, sys, zipfile
sys.stdout.reconfigure(encoding="utf-8")
Z = r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\6ab10eb3b23ba_student_resource.zip"
OLD = r"C:\Users\ARNAVG~1\AppData\Local\Temp\claude\C--Users-Arnav-Gawade-pro--OneDrive-Desktop-amazon-ml\89303762-6110-432e-ae87-2d206a5ca21f\scratchpad"
z = zipfile.ZipFile(Z)


def rows(member):
    f = io.TextIOWrapper(z.open("student_resource/dataset/" + member), encoding="utf-8")
    next(f)
    for l in f:
        yield l.rstrip("\n").split("\t")


def rows_file(path):
    with open(path, encoding="utf-8") as f:
        next(f)
        for l in f:
            yield l.rstrip("\n").split("\t")


# ---- counts per split/country/source
cnt = collections.Counter()
s1c = {}
for split in ("train", "test"):
    for k in (1, 2, 3):
        for p in rows(f"{split}/{split}_source{k}.tsv"):
            cnt[(split, p[3], k)] += 1
            if k == 1:
                s1c[p[0]] = p[3]
print("records per S1 (S2+S3 / S1):")
for split in ("train", "test"):
    for c in ("US", "India", "France"):
        n1 = cnt[(split, c, 1)]
        if n1:
            print(f"  {split:5s} {c:7s} S1={n1:8d}  S2/S1={cnt[(split, c, 2)] / n1:.3f}  S3/S1={cnt[(split, c, 3)] / n1:.3f}")


def shape(mapping, label):
    per = collections.defaultdict(collections.Counter)
    for s, ids in mapping:
        n2 = sum(x.startswith("S2") for x in ids)
        n3 = sum(x.startswith("S3") for x in ids)
        per[s1c[s]][(min(n2, 5), min(n3, 5))] += 1
    for c, d in per.items():
        tot = sum(d.values())
        m2 = sum(k[0] * v for k, v in d.items()) / tot
        m3 = sum(k[1] * v for k, v in d.items()) / tot
        top = ", ".join(f"{k}:{100 * v / tot:.1f}%" for k, v in d.most_common(8))
        print(f"  {label:12s} {c:7s} mean S2 {m2:.3f} mean S3 {m3:.3f} | (nS2,nS3) {top}")


gt = ((p[0], [x for x in p[1].split(",") if x]) if len(p) > 1 else (p[0], [])
      for p in rows("train/train_ground_truth.tsv"))
shape(gt, "train truth")
pred = ((p[0], [x for x in p[1].split(",") if x]) for p in rows_file(OLD + r"\kout\output\matching_results.tsv"))
shape(pred, "test v1 pred")
