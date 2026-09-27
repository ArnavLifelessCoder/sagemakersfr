"""Candidate generation.

Every Source-2/3 record can belong to at most one Source-1 entity, so blocking is run
from the S2/S3 side: for each query record we retrieve the top-K most similar S1
records through two independent channels:

* name channel    - TF-IDF over character 3-grams of the normalised core name
* address channel - TF-IDF over canonical address tokens plus a composite
                    (house number, first street token) token

Retrieval is an exact sparse top-K inner product (sparse_dot_topn, multithreaded).
Search is partitioned by (country, canonical state); query records without a state
(e.g. empty address) are searched against the whole country on name only.
The union of both channels is the candidate set; both channel cosines are then
computed exactly for every pair in the union.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
import scipy.sparse as sp

COMB_W = 0.5 ** 0.5


def _addr_docs(df):
    docs = []
    for t, h, s, st in zip(df.addr_toks.values, df.hn.values, df.street.values, df.state.values):
        d = t.split()
        s0 = s.split()[0] if s else ""
        if h and s0:
            d.append(f"HS_{h}_{s0}")
        docs.append(d)
    return docs


def _identity(x):
    return x


def _topk(Q, S_T, k, threads, min_score=1e-6):
    if Q.shape[0] == 0 or S_T.shape[1] == 0 or k <= 0:
        e = np.zeros((0,), np.int64)
        return e, e, np.zeros((0,), np.float32)
    C = sp_matmul_topn(Q, S_T, top_n=min(k, S_T.shape[1]), threshold=min_score, n_threads=threads).tocoo()
    return C.row.astype(np.int64), C.col.astype(np.int64), C.data.astype(np.float32)


def _rowdot(A, B):
    return np.asarray(A.multiply(B).sum(axis=1)).ravel().astype(np.float32)


class _Index:
    """TF-IDF matrices of one S1 partition."""

    def __init__(self, S, max_df, use_addr=True):
        self.nv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True,
                                  dtype=np.float32, max_df=max_df if len(S) > 2000 else 1.0)
        self.Sn = self.nv.fit_transform(S.name_core.values).tocsr()
        self.SnT = self.Sn.T.tocsr()
        self.av = None
        if use_addr:
            self.av = TfidfVectorizer(analyzer=_identity, sublinear_tf=True, dtype=np.float32,
                                      max_df=max_df if len(S) > 2000 else 1.0)
            self.Sa = self.av.fit_transform(_addr_docs(S)).tocsr()
            self.SaT = self.Sa.T.tocsr()
            # combined channel: cosine = (name_cos + addr_cos) / 2. Breaks the ties of generic names shared by many
            # businesses (France) and of addresses holding many businesses: the S1 agreeing on BOTH ranks first.
            self.ScT = (sp.hstack([self.Sn, self.Sa]).tocsr() * np.float32(COMB_W)).T.tocsr()

    def query(self, Qf, k_name, k_addr, threads, min_score=1e-6):
        Qn = self.nv.transform(Qf.name_core.values).tocsr()
        r1, c1, _ = _topk(Qn, self.SnT, k_name, threads, min_score)
        if self.av is not None and k_addr > 0:
            Qa = self.av.transform(_addr_docs(Qf)).tocsr()
            r2, c2, _ = _topk(Qa, self.SaT, k_addr, threads, min_score)
            Qc = sp.hstack([Qn, Qa]).tocsr() * np.float32(COMB_W)
            r3, c3, _ = _topk(Qc, self.ScT, max(k_name, k_addr), threads, min_score)
        else:
            Qa = None
            r2 = c2 = r3 = c3 = np.zeros((0,), np.int64)
        pairs = np.unique(np.concatenate([r1 * (1 << 32) + c1, r2 * (1 << 32) + c2, r3 * (1 << 32) + c3]))
        r = pairs >> 32
        c = pairs & ((1 << 32) - 1)
        name_cos = _rowdot(Qn[r], self.Sn[c]) if len(r) else np.zeros(0, np.float32)
        if Qa is not None and len(r):
            addr_cos = _rowdot(Qa[r], self.Sa[c])
        else:
            addr_cos = np.zeros(len(r), np.float32)
        return r, c, name_cos, addr_cos


def generate_candidates(s1, q, **kw):
    """Return DataFrame [qi, si, name_cos, addr_cos] of positional indices into q and s1."""
    parts = [p for _, p in iter_candidates(s1, q, **kw)]
    res = pd.concat(parts, ignore_index=True)
    return res.drop_duplicates(["qi", "si"]).reset_index(drop=True)


# S1 state -> other states its true S2/S3 records are written in (learned from the TRAINING ground
# truth, see experiments/diag/state_shift.py): 18.5% of Telangana pairs are written "Andhra Pradesh"
# (Hyderabad), 8% of DC pairs parse as "Washington" (shuffled components). Queries of those states are
# also searched against the S1 partition.
STATE_NEIGHBOURS = {"IN-TG": ["IN-AP"], "US-DC": ["US-WA"], "US-WA": ["US-DC"], "IN-AP": ["IN-TG"]}


def iter_candidates(s1, q, k_name=10, k_addr=10, k_name_nostate=10, threads=-1,
                    chunk=200000, max_df=0.05, log=print):
    """Yield (partition_name, DataFrame[qi, si, name_cos, addr_cos]) one partition at a time.

    Indices are positional into q and s1. Pairs are unique within a partition; the
    no-state fallback partitions may repeat a pair already produced by a state partition
    only for S1 records without a state (deduplicate if you concatenate)."""
    s_country = s1.country.values
    s_state = s1.state.values
    q_country = q.country.values
    q_state = q.state.values
    for country in sorted(set(s_country) & set(q_country)):
        sc = s_country == country
        qc = q_country == country
        states = sorted(set(s_state[sc]) - {""})
        for st in states:
            s_idx = np.flatnonzero(sc & (s_state == st))
            q_idx = np.flatnonzero(qc & np.isin(q_state, [st] + STATE_NEIGHBOURS.get(st, [])))
            if len(q_idx) == 0:
                continue
            yield f"{country}/{st}", _run(s1, q, s_idx, q_idx, max_df, k_name, k_addr, threads, chunk)
        # queries without a recognised state (or whose state has no S1): country-wide search
        known = list(states)
        q_rest = np.flatnonzero(qc & ~np.isin(q_state, known))
        if len(q_rest):
            s_all = np.flatnonzero(sc)
            yield f"{country}/-", _run(s1, q, s_all, q_rest, max_df, k_name_nostate, k_addr, threads, chunk)
        # S1 records without a state: reachable from every query of the country
        s_nostate = np.flatnonzero(sc & (s_state == ""))
        if len(s_nostate):
            q_st = np.flatnonzero(qc & np.isin(q_state, known))
            yield f"{country}/s1-nostate", _run(s1, q, s_nostate, q_st, max_df, 3, 3, threads, chunk, min_score=0.3)
        log(f"  blocking {country}: {len(states)} states, {qc.sum()} queries, {len(q_rest)} without state")


def _run(s1, q, s_idx, q_idx, max_df, k_name, k_addr, threads, chunk, min_score=1e-6):
    idx = _Index(s1.iloc[s_idx], max_df)
    out = []
    for a in range(0, len(q_idx), chunk):
        qq = q_idx[a:a + chunk]
        r, c, nc, ac = idx.query(q.iloc[qq], k_name, k_addr, threads, min_score)
        out.append(pd.DataFrame({"qi": qq[r], "si": s_idx[c], "name_cos": nc, "addr_cos": ac}))
    return pd.concat(out, ignore_index=True)
