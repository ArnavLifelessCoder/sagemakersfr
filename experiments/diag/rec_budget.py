"""Per country: S2/S3 records per S1 in test, assigned per S1 (v7 fallback), unassigned per S1 -> compare with the
labelled reference's (true per S1, distractors per S1) to see where France/India lose recall."""
import os, sys, collections
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
T = r"C:\Users\ARNAVG~1\AppData\Local\Temp\claude\C--Users-Arnav-Gawade-pro--OneDrive-Desktop-amazon-ml\89303762-6110-432e-ae87-2d206a5ca21f\scratchpad\sr\student_resource\dataset\test"
OLD = r"C:\Users\ARNAVG~1\AppData\Local\Temp\claude\C--Users-Arnav-Gawade-pro--OneDrive-Desktop-amazon-ml\61cd2e14-f061-440d-8f3f-a2caed65f90a\scratchpad"
SUB = r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\submissions\v7_seed202_post\matching_results.tsv"

def counts(d):
    c = collections.Counter(); ctry = {}
    for k in (1, 2, 3):
        with open(os.path.join(d, f"test_source{k}.tsv"), encoding="utf-8") as f:
            next(f)
            for l in f:
                p = l.rstrip("\n").split("\t")
                c[(k, p[3])] += 1
                ctry[p[0]] = p[3]
    return c, ctry

c, ctry = counts(T)
m = pd.read_csv(SUB, sep="\t", dtype=str, keep_default_na=False)
asg = collections.Counter()
for s, x in zip(m.iloc[:, 0], m.iloc[:, 1]):
    for q in x.split(","):
        if q:
            asg[(q.split("-")[0], ctry[s])] += 1
print("TEST")
for co in ("US", "India", "France"):
    n1 = c[(1, co)]
    print(f"{co:7s} S1 {n1:8d} | S2/S1 {c[(2, co)]/n1:.3f} S3/S1 {c[(3, co)]/n1:.3f} | assigned S2/S1 {asg[('S2', co)]/n1:.3f} "
          f"S3/S1 {asg[('S3', co)]/n1:.3f} | unassigned S2/S1 {(c[(2, co)]-asg[('S2', co)])/n1:.3f} S3/S1 {(c[(3, co)]-asg[('S3', co)])/n1:.3f}")
SL = os.path.join(OLD, "eslice")
rc, rctry = counts(os.path.join(SL, "test"))
gt = pd.read_csv(os.path.join(SL, "gt.tsv"), sep="\t", dtype=str, keep_default_na=False)
tr = collections.Counter()
for s, x in zip(gt.source1_entity_id, gt.matched_entity_ids):
    for q in x.split(","):
        if q:
            tr[(q.split("-")[0], rctry[s])] += 1
print("REFERENCE (labelled)")
for co in ("US", "India"):
    n1 = rc[(1, co)]
    print(f"{co:7s} S1 {n1:8d} | S2/S1 {rc[(2, co)]/n1:.3f} S3/S1 {rc[(3, co)]/n1:.3f} | true S2/S1 {tr[('S2', co)]/n1:.3f} "
          f"S3/S1 {tr[('S3', co)]/n1:.3f} | distractor S2/S1 {(rc[(2, co)]-tr[('S2', co)])/n1:.3f} S3/S1 {(rc[(3, co)]-tr[('S3', co)])/n1:.3f}")
