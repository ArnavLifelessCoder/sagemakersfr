"""Learn the transliteration table from ALL training native-name pairs (as Kaggle training does)."""
import io, sys, zipfile, json
sys.path.insert(0, r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\code\business_entity_resolution")
from src.normalize import is_native
from src.prep import learn_translit

Z = r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\6ab10eb3b23ba_student_resource.zip"
OUT = sys.argv[1]
z = zipfile.ZipFile(Z)


def rows(m):
    f = io.TextIOWrapper(z.open("student_resource/dataset/" + m), encoding="utf-8")
    next(f)
    for l in f:
        yield l.rstrip("\n").split("\t")


owner = {}
for p in rows("train/train_ground_truth.tsv"):
    for x in (p[1].split(",") if len(p) > 1 else []):
        if x:
            owner[x] = p[0]
nat = {}
for k in (2, 3):
    for p in rows(f"train/train_source{k}.tsv"):
        if p[0] in owner and is_native(p[1]):
            nat[p[0]] = p[1]
need = {owner[e] for e in nat}
s1 = {p[0]: p[1] for p in rows("train/train_source1.tsv") if p[0] in need}
tr = learn_translit([s1[owner[e]] for e in nat], list(nat.values()))
json.dump(tr, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
print("entries", len(tr))
