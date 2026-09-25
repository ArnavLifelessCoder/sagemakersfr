"""Does the transliteration table generalise? coverage on held-out train vs test native tokens."""
import io, sys, zipfile, zlib, collections
sys.path.insert(0, r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\code\business_entity_resolution")
sys.stdout.reconfigure(encoding="utf-8")
from src.normalize import is_native
from src.prep import learn_translit

Z = r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\6ab10eb3b23ba_student_resource.zip"
z = zipfile.ZipFile(Z)


def rows(member):
    f = io.TextIOWrapper(z.open("student_resource/dataset/" + member), encoding="utf-8")
    next(f)
    for l in f:
        yield l.rstrip("\n").split("\t")


# native-name records of train S2/S3 + their owners
owner = {}
for p in rows("train/train_ground_truth.tsv"):
    for x in (p[1].split(",") if len(p) > 1 else []):
        if x:
            owner[x] = p[0]
nat = {}
for k in (2, 3):
    for p in rows(f"train/train_source{k}.tsv"):
        if is_native(p[1]):
            nat[p[0]] = p[1]
need = {owner[e] for e in nat if e in owner}
s1name = {}
for p in rows("train/train_source1.tsv"):
    if p[0] in need:
        s1name[p[0]] = p[1]
fit_a, fit_b, held = [], [], []
for e, n in nat.items():
    s = owner.get(e)
    if not s:
        continue
    if zlib.crc32(s.encode()) % 10 < 7:
        fit_a.append(s1name[s]); fit_b.append(n)
    else:
        held.append(n)
tr = learn_translit(fit_a, fit_b)
print("translit entries learned on 70% of train:", len(tr))


def coverage(names):
    tot = hit = 0
    for n in names:
        for w in n.split():
            if is_native(w):
                tot += 1
                hit += w in tr
    return hit / max(1, tot), tot


print("held-out train native-token coverage: %.4f (%d tokens)" % coverage(held))
test_nat = []
for k in (2, 3):
    for p in rows(f"test/test_source{k}.tsv"):
        if is_native(p[1]):
            test_nat.append(p[1])
print("test native-token coverage:           %.4f (%d tokens)" % coverage(test_nat))
miss = collections.Counter(w for n in test_nat for w in n.split() if is_native(w) and w not in tr)
print("most frequent uncovered test tokens:", miss.most_common(30))
