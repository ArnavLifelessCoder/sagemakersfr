"""Write a small TEST slice (a few states, all sources) in dataset layout: slice/test/test_source{k}.tsv"""
import os, sys
sys.path.insert(0, r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\code\business_entity_resolution")
from src.make_dev_subset import state_of

D = r"C:\Users\ARNAVG~1\AppData\Local\Temp\claude\C--Users-Arnav-Gawade-pro--OneDrive-Desktop-amazon-ml\89303762-6110-432e-ae87-2d206a5ca21f\scratchpad\sr\student_resource\dataset\test"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get("SLICE", "tslice"), "test")
STATES = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else {"IN-TN", "IN-RJ"}
os.makedirs(OUT, exist_ok=True)
for k in (1, 2, 3):
    n = 0
    with open(os.path.join(D, f"test_source{k}.tsv"), encoding="utf-8") as f, \
            open(os.path.join(OUT, f"test_source{k}.tsv"), "w", encoding="utf-8", newline="\n") as g:
        g.write(next(f))
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 4 and p[2] and state_of(p[2], p[3]) in STATES:
                g.write(line)
                n += 1
    print(k, n)
