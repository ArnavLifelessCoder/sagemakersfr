import re, collections, pickle, sys, os
sys.stdout.reconfigure(encoding="utf-8")
recs, owner = pickle.load(open(os.path.join(os.path.dirname(__file__), "vt.pkl"), "rb"))
s1 = [k for k in recs if k.startswith("S1")]
un = [k for k in recs if not k.startswith("S1") and k not in owner]
m = [k for k in recs if not k.startswith("S1") and k in owner and owner[k] in recs]


def hn(a):
    x = re.search(r"(?<![\w/])#*(\d+)", a)
    return x.group(1) if x else None


def rel(a, b):
    if a is None or b is None:
        return "missing"
    if a == b:
        return "same"
    if a.lstrip("0") == b.lstrip("0"):
        return "zeropad"
    if a.startswith(b) or b.startswith(a):
        return "prefix"
    d = abs(int(a) - int(b))
    return "diff<=20" if d <= 20 else "diff>20"


c = collections.Counter()
ex = collections.defaultdict(list)
for k in m:
    r = rel(hn(recs[owner[k]][1]), hn(recs[k][1]))
    c[r] += 1
    if len(ex[r]) < 4:
        ex[r].append((recs[owner[k]][1], recs[k][1]))
print("TRUE MATCH house-number relation:", c.most_common())
for r, v in ex.items():
    if r in ("diff<=20", "diff>20"):
        for e in v:
            print("   ", r, e)

# distractors vs best S1 by name token overlap
idx = collections.defaultdict(set)
tok = lambda s: set(re.findall(r"[a-z0-9]+", s.lower()))
for k in s1:
    for t in tok(recs[k][0]):
        idx[t].add(k)
c2 = collections.Counter()
extra = collections.Counter()
extra_true = collections.Counter()
for u in un:
    sc = collections.Counter()
    for t in tok(recs[u][0]):
        if len(idx[t]) < 200:
            for k in idx[t]:
                sc[k] += 1
    if not sc:
        c2["no-name-neighbour"] += 1
        continue
    b = sc.most_common(1)[0][0]
    c2[rel(hn(recs[b][1]), hn(recs[u][1]))] += 1
    for t in tok(recs[u][0]) - tok(recs[b][0]):
        extra[t] += 1
for k in m:
    for t in tok(recs[k][0]) - tok(recs[owner[k]][0]):
        extra_true[t] += 1
print("DISTRACTOR vs best-name S1 house-number relation:", c2.most_common())
print("extra tokens in distractors:", extra.most_common(40))
print("extra tokens in true matches:", extra_true.most_common(40))

# are distractors themselves clustered? count unmatched records sharing (hn, street word) with another unmatched
def akey(a):
    h = hn(a)
    w = [t for t in re.findall(r"[a-z]+", a.lower()) if len(t) > 3]
    return (h, w[0] if w else None)
g = collections.Counter(akey(recs[u][1]) for u in un if recs[u][1])
print("unmatched records whose (hn,street) key is shared by another unmatched:",
      sum(v for v in g.values() if v > 1), "/", sum(g.values()))
