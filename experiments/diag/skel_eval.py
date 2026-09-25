"""Evaluate phonetic-skeleton transliteration fallback against the learned table (as ground truth)."""
import io, sys, zipfile, collections, re, zlib
sys.path.insert(0, r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\code\business_entity_resolution")
sys.stdout.reconfigure(encoding="utf-8")
from unidecode import unidecode
from rapidfuzz.distance import Levenshtein
from src.normalize import is_native, fold
from src.prep import learn_translit

Z = r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\6ab10eb3b23ba_student_resource.zip"
z = zipfile.ZipFile(Z)


def rows(member):
    f = io.TextIOWrapper(z.open("student_resource/dataset/" + member), encoding="utf-8")
    next(f)
    for l in f:
        yield l.rstrip("\n").split("\t")


def skeleton(s):
    s = s.lower()
    for a, b in (("ph", "f"), ("bh", "b"), ("dh", "d"), ("th", "t"), ("kh", "k"), ("gh", "g"), ("sh", "s"),
                 ("ch", "c"), ("ck", "k"), ("q", "k"), ("w", "v"), ("x", "ks"), ("z", "j")):
        s = s.replace(a, b)
    s = re.sub(r"[^a-z]", "", s)
    s = re.sub(r"[aeiouy]", "", s)
    s = re.sub(r"(.)\1+", r"\1", s)
    return s


owner = {}
for p in rows("train/train_ground_truth.tsv"):
    for x in (p[1].split(",") if len(p) > 1 else []):
        if x:
            owner[x] = p[0]
nat, latin = {}, collections.Counter()
for k in (2, 3):
    for p in rows(f"train/train_source{k}.tsv"):
        if p[3] != "India":
            continue
        if is_native(p[1]):
            nat[p[0]] = p[1]
        else:
            latin.update(re.findall(r"[a-z]+", fold(p[1])))
need = {owner[e] for e in nat if e in owner}
s1name = {}
for p in rows("train/train_source1.tsv"):
    if p[3] == "India":
        latin.update(re.findall(r"[a-z]+", fold(p[1])))
    if p[0] in need:
        s1name[p[0]] = p[1]
tr = learn_translit([s1name[owner[e]] for e in nat if e in owner], [nat[e] for e in nat if e in owner])
print("table size", len(tr))

vocab = {w: c for w, c in latin.items() if c >= 20 and len(w) >= 3}
by_skel = collections.defaultdict(list)
for w, c in vocab.items():
    by_skel[skeleton(w)].append((c, w))
for k in by_skel:
    by_skel[k].sort(reverse=True)
skels = list(by_skel)


def fallback(tok):
    sk = skeleton(unidecode(tok))
    if len(sk) < 2:
        return None
    if sk in by_skel:
        return by_skel[sk][0][1]
    if len(sk) >= 4:
        best = None
        for k in skels:
            if abs(len(k) - len(sk)) <= 1 and k[0] == sk[0] and Levenshtein.distance(k, sk) <= 1:
                c, w = by_skel[k][0]
                if best is None or c > best[0]:
                    best = (c, w)
        return best[1] if best else None
    return None


ok = wrong = none = 0
ex = []
for nt, lt in tr.items():
    if not is_native(nt):
        continue
    g = fallback(nt)
    if g is None:
        none += 1
    elif g == lt:
        ok += 1
    else:
        wrong += 1
        if len(ex) < 25:
            ex.append((nt, lt, g, skeleton(unidecode(nt)), skeleton(lt)))
tot = ok + wrong + none
print(f"skeleton fallback vs table: correct {ok / tot:.3f}, wrong {wrong / tot:.3f}, no-guess {none / tot:.3f} (n={tot})")
for e in ex:
    print("  ", e)
for w in ["मोटर्स", "स्टोर्स", "एजेंसीज", "ज्वेलर्स", "जनरल", "बेकरी", "प्रोविजन", "स्वीट्स", "फार्मेसी", "मेडिकल्स",
          "ऑटोमोबाइल्स", "ट्रेडर्स", "हार्डवेयर", "गारमेंट्स", "स्टील", "इलेक्ट्रॉनिक्स", "फर्नीचर", "रेस्टोरेंट",
          "टेक्सटाइल्स", "ಬೇಕರಿ", "ಫರ್ನಿಚರ್", "ఫార్మసీ", "స్టోర్స్", "ఎలక్ట్రానిక్స్"]:
    print(w, unidecode(w), "->", fallback(w))
