"""Labelled evaluation slice from TRAIN in test layout: eslice/test/test_source{k}.tsv + eslice/gt.tsv.
S1 in the chosen states; all their true matches (wherever they are); unmatched S2/S3 in those states;
a proportional sample of empty-address unmatched records."""
import io, os, sys, zipfile, random
sys.path.insert(0, r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\code\business_entity_resolution")
from src.make_dev_subset import state_of

Z = r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\6ab10eb3b23ba_student_resource.zip"
STATES = set(sys.argv[1].split(","))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), sys.argv[2])
os.makedirs(os.path.join(OUT, "test"), exist_ok=True)
z = zipfile.ZipFile(Z)


def lines(m):
    f = io.TextIOWrapper(z.open("student_resource/dataset/" + m), encoding="utf-8")
    head = next(f)
    return head, f


head, f = lines("train/train_source1.tsv")
sel, n1 = set(), 0
with open(os.path.join(OUT, "test", "test_source1.tsv"), "w", encoding="utf-8", newline="\n") as g:
    g.write(head)
    for l in f:
        n1 += 1
        p = l.rstrip("\n").split("\t")
        if p[2] and state_of(p[2].replace('"', ''), p[3]) in STATES:
            sel.add(p[0])
            g.write(l)
frac = len(sel) / n1
_, f = lines("train/train_ground_truth.tsv")
keep, matched = set(), set()
with open(os.path.join(OUT, "gt.tsv"), "w", encoding="utf-8", newline="\n") as g:
    g.write("source1_entity_id\tmatched_entity_ids\n")
    for l in f:
        p = l.rstrip("\n").split("\t")
        ids = [x for x in (p[1].split(",") if len(p) > 1 else []) if x]
        matched.update(ids)
        if p[0] in sel:
            keep.update(ids)
            g.write(l if l.endswith("\n") else l + "\n")
rnd = random.Random(0)
for k in (2, 3):
    head, f = lines(f"train/train_source{k}.tsv")
    n = 0
    with open(os.path.join(OUT, "test", f"test_source{k}.tsv"), "w", encoding="utf-8", newline="\n") as g:
        g.write(head)
        for l in f:
            p = l.rstrip("\n").split("\t")
            e = p[0]
            if e in keep or (e not in matched and ((p[2] and state_of(p[2].replace('"', ''), p[3]) in STATES)
                                                   or (not p[2] and rnd.random() < frac))):
                g.write(l)
                n += 1
    print("source", k, n)
print("S1", len(sel), "true pairs", len(keep))
