"""Compare v2 and v3 final matches per country + sample disagreements."""
import sys, random, collections
sys.stdout.reconfigure(encoding="utf-8")
H = __file__.rsplit("\\", 1)[0]
D = r"C:\Users\ARNAVG~1\AppData\Local\Temp\claude\C--Users-Arnav-Gawade-pro--OneDrive-Desktop-amazon-ml\89303762-6110-432e-ae87-2d206a5ca21f\scratchpad\sr\student_resource\dataset\test"


def load(p):
    out = set()
    with open(p, encoding="utf-8") as f:
        next(f)
        for l in f:
            s, _, r = l.rstrip("\n").partition("\t")
            out.update((s, x) for x in r.split(",") if x)
    return out


ctry = {}
with open(D + r"\test_source1.tsv", encoding="utf-8") as f:
    next(f)
    for l in f:
        p = l.split("\t")
        ctry[p[0]] = p[3].strip()
a, b = load(H + r"\v2\matching_results.tsv"), load(H + r"\v3\matching_results.tsv")
st = collections.defaultdict(collections.Counter)
for s, q in a | b:
    st[ctry[s]][("both" if (s, q) in a and (s, q) in b else "v2only" if (s, q) in a else "v3only")] += 1
for c, v in sorted(st.items()):
    print(c, dict(v))
random.seed(2)
need, samples = set(), {}
for c in ("India", "US"):
    o2 = [x for x in a - b if ctry[x[0]] == c]
    o3 = [x for x in b - a if ctry[x[0]] == c]
    samples[c] = (random.sample(o2, 10), random.sample(o3, 10))
    for lst in samples[c]:
        for s, q in lst:
            need |= {s, q}
rec = {}
for k in (1, 2, 3):
    with open(D + rf"\test_source{k}.tsv", encoding="utf-8") as f:
        next(f)
        for l in f:
            p = l.rstrip("\n").split("\t")
            if p[0] in need:
                rec[p[0]] = (p[1], p[2])
for c, (o2, o3) in samples.items():
    for tag, lst in (("v2 only", o2), ("v3 only", o3)):
        print(f"\n##### {c} {tag}")
        for s, q in lst:
            print("  S1", rec[s], "\n  Q ", rec[q])
