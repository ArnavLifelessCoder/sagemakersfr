import sys, time
sys.path.insert(0, r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\code\business_entity_resolution")
import numpy as np, pandas as pd
from src.prep import parse_frame
from src import blocking as B
from sklearn.feature_extraction.text import TfidfVectorizer
if __name__ == "__main__":
    s1 = pd.read_parquet("dev/src1.parquet"); s1 = s1[s1.country=="US"].head(30000)
    q = pd.read_parquet("dev/src2.parquet"); q = q[q.country=="US"].head(60000)
    s1 = parse_frame(s1, None, workers=6).reset_index(drop=True); q = parse_frame(q, None, workers=6).reset_index(drop=True)
    print("parsed", flush=True)
    nv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True, dtype=np.float32, max_df=0.05)
    t=time.time(); Sn = nv.fit_transform(s1.name_core.values); Qn = nv.transform(q.name_core.values); print("tfidf", time.time()-t, Sn.shape, Sn.nnz, flush=True)
    t=time.time(); r=B._topk(Qn, Sn, 20, 8); print("topk name", time.time()-t, len(r[0]), flush=True)
    av = TfidfVectorizer(analyzer=B._identity, sublinear_tf=True, dtype=np.float32, max_df=0.05)
    t=time.time(); Sa = av.fit_transform(B._addr_docs(s1)); Qa = av.transform(B._addr_docs(q)); print("tfidf addr", time.time()-t, Sa.shape, Sa.nnz, flush=True)
    t=time.time(); r=B._topk(Qa, Sa, 20, 8); print("topk addr", time.time()-t, flush=True)
    t=time.time(); c = B.generate_candidates(s1, q, 20, 20, 8); print("full gen", time.time()-t, len(c), flush=True)
