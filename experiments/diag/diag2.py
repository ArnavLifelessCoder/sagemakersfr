"""Inspect India test clusters where we predicted more matches than the training prior makes likely."""
import random, sys, collections
sys.stdout.reconfigure(encoding="utf-8")
OLD = r"C:\Users\ARNAVG~1\AppData\Local\Temp\claude\C--Users-Arnav-Gawade-pro--OneDrive-Desktop-amazon-ml\89303762-6110-432e-ae87-2d206a5ca21f\scratchpad"
D = OLD + r"\sr\student_resource\dataset\test"


def rows(p):
    with open(p, encoding="utf-8") as f:
        next(f)
        for l in f:
            yield l.rstrip("\n").split("\t")


s1 = {p[0]: (p[1], p[2]) for p in rows(D + r"\test_source1.tsv") if p[3] == "India"}
pred = {p[0]: [x for x in p[1].split(",") if x] for p in rows(OLD + r"\kout\output\matching_results.tsv") if p[0] in s1}
random.seed(3)
big = [s for s, m in pred.items() if sum(x.startswith("S2") for x in m) >= 3]
samp = random.sample(big, 14)
need = {x for s in samp for x in pred[s]}
rec = {}
for k in (2, 3):
    for p in rows(D + rf"\test_source{k}.tsv"):
        if p[0] in need:
            rec[p[0]] = (p[1], p[2])
for s in samp:
    print("\nS1  ", s1[s])
    for x in sorted(pred[s]):
        print("   ", x[:2], rec[x])
