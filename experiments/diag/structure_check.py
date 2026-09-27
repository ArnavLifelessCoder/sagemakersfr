"""Do row order or entity-id values carry information about matched vs distractor records (TRAIN)?"""
import io, sys, zipfile, collections
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
Z = r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\6ab10eb3b23ba_student_resource.zip"
z = zipfile.ZipFile(Z)


def rows(m):
    f = io.TextIOWrapper(z.open("student_resource/dataset/" + m), encoding="utf-8")
    next(f)
    for l in f:
        yield l.rstrip("\n").split("\t")


owner = {}
gt_order = {}
for i, p in enumerate(rows("train/train_ground_truth.tsv")):
    gt_order[p[0]] = i
    for x in (p[1].split(",") if len(p) > 1 else []):
        if x:
            owner[x] = p[0]
s1_pos = {}
for i, p in enumerate(rows("train/train_source1.tsv")):
    s1_pos[p[0]] = i
n1 = len(s1_pos)
for k in (2, 3):
    pos, matched, idnum, own_pos = [], [], [], []
    for i, p in enumerate(rows(f"train/train_source{k}.tsv")):
        pos.append(i)
        o = owner.get(p[0])
        matched.append(o is not None)
        idnum.append(int(p[0].split("-")[1]))
        own_pos.append(s1_pos[o] / n1 if o else -1)
    pos, matched, idnum, own_pos = map(np.array, (pos, matched, idnum, own_pos))
    n = len(pos)
    dec = (pos * 10 // n)
    print(f"source {k}: {n} rows, matched share {matched.mean():.4f}")
    print("   matched share by row-position decile:", [round(matched[dec == d].mean(), 4) for d in range(10)])
    q = np.quantile(idnum, np.linspace(0, 1, 11))
    idd = np.clip(np.searchsorted(q, idnum, side="right") - 1, 0, 9)
    print("   matched share by id-value decile:   ", [round(matched[idd == d].mean(), 4) for d in range(10)])
    # does a matched record's row position track its S1's row position?
    m = matched
    r = np.corrcoef(pos[m] / n, own_pos[m])[0, 1]
    print(f"   corr(row position, owner S1 row position) = {r:.4f}")
    # are consecutive rows often the same entity?
    ow = [owner.get(x) for x in []]
