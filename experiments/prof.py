import sys, time
sys.path.insert(0, r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\code\business_entity_resolution")
import pandas as pd
from src.prep import parse_frame
if __name__ == "__main__":
    s1 = pd.read_parquet("dev/src1.parquet").head(50000)
    t=time.time(); p=parse_frame(s1, None, workers=1); print("parse 50k single", time.time()-t, flush=True)
    t=time.time(); p=parse_frame(s1, None, workers=6); print("parse 50k pool6", time.time()-t, flush=True)
