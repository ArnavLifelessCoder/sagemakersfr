"""Why were true pairs never blocked? split by cause, plus examples; and test India state sizes."""
import sys, os, collections, random
import pandas as pd
sys.path.insert(0, r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\code\business_entity_resolution")
sys.stdout.reconfigure(encoding="utf-8")
from src.make_dev_subset import state_of
from src.normalize import parse_name
SL, OUT = sys.argv[1], sys.argv[2]
T = r"C:\Users\ARNAVG~1\AppData\Local\Temp\claude\C--Users-Arnav-Gawade-pro--OneDrive-Desktop-amazon-ml\89303762-6110-432e-ae87-2d206a5ca21f\scratchpad\sr\student_resource\dataset\test"
gt = pd.read_csv(os.path.join(SL, "gt.tsv"), sep="\t", dtype=str, keep_default_na=False)
rec = {}
for k in (1, 2, 3):
    d = pd.read_csv(os.path.join(SL, "test", f"test_source{k}.tsv"), sep="\t", dtype=str, keep_default_na=False, quoting=3)
    rec.update(dict(zip(d.entity_id, zip(d.business_name, d.business_address, d.country))))
ps = pd.read_parquet(os.path.join(OUT, "pair_scores.parquet"), columns=["s1", "q"])
cand = set(zip(ps.s1.values, ps.q.values))
cause = collections.Counter()
ex = collections.defaultdict(list)
for s, m in zip(gt.source1_entity_id, gt.matched_entity_ids):
    for q in (x for x in m.split(",") if x):
        if (s, q) in cand:
            continue
        sn, sa, c = rec[s]
        qn, qa, _ = rec[q]
        ss = state_of(sa.replace('"', ''), c)
        qs = state_of(qa.replace('"', ''), c) if qa else "EMPTY"
        if qs == "EMPTY":
            k = "q address empty"
        elif qs != ss:
            k = f"other state {ss}->{qs}"
        else:
            nat = parse_name(qn)["native"]
            k = "same state, native-script name" if nat else "same state, latin name"
        cause[(c, k)] += 1
        if len(ex[(c, k)]) < 6:
            ex[(c, k)].append((sn, sa[:70], "||", qn, qa[:70]))
tot = collections.Counter()
for (c, k), v in cause.items():
    tot[c] += v
for (c, k), v in sorted(cause.items(), key=lambda kv: -kv[1]):
    print(f"{c:6s} {k:34s} {v:7d}  ({v / tot[c]:.1%} of {c} blocking misses)")
random.seed(0)
for key in [k for k, _ in cause.most_common(6)]:
    print("\n##", key)
    for e in ex[key]:
        print("   ", e)
# test India S1 per state
st = collections.Counter()
with open(os.path.join(T, "test_source1.tsv"), encoding="utf-8") as f:
    next(f)
    for l in f:
        p = l.rstrip("\n").split("\t")
        if p[3] == "India":
            st[state_of(p[2].replace('"', ''), p[3])] += 1
print("\ntest India S1 by state:", st.most_common(22))
