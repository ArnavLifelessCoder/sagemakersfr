"""Pairwise features for (S1 record, S2/S3 record) candidate pairs."""
import math
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

# states the generator uses interchangeably for the same place (see blocking.STATE_NEIGHBOURS)
STATE_GROUP = {"IN-AP": "IN-TG", "US-DC": "US-WA"}

HN_REL = {"both_missing": 0, "q_missing": 1, "s_missing": 2, "equal": 3, "equal_suffix": 4,
          "truncated": 5, "diff_small": 6, "diff_mid": 7, "diff_large": 8}


def _hn_rel(a, b, sa, sb):
    if not a and not b:
        return 0, -1.0
    if not b:
        return 1, -1.0
    if not a:
        return 2, -1.0
    if a == b:
        return (3 if sa == sb else 4), 0.0
    d = abs(int(a) - int(b)) if len(a) < 12 and len(b) < 12 else 1e9
    if a.startswith(b) or b.startswith(a) or a.endswith(b) or b.endswith(a):
        return 5, math.log1p(d)
    if d <= 20:
        return 6, math.log1p(d)
    if d <= 200:
        return 7, math.log1p(d)
    return 8, math.log1p(d)


def _covered(x, others):
    """x is 'the same identifier' as one of others: equal, or a prefix/suffix relation (truncation, zero-pad)."""
    for o in others:
        if x == o or (len(x) >= 2 and len(o) >= 2 and abs(len(x) - len(o)) <= 2 and
                      (x.startswith(o) or o.startswith(x) or x.endswith(o) or o.endswith(x))):
            return True
    return False


def _ident_feats(sa, qa):
    """Composite house-identifier relation between the S1 (sa) and the record (qa).

    Returns jaccard, q_only, s_only, min edit distance between uncovered tokens (-1 if none),
    sets equal (nonempty), n_replaced (S1 identifiers the record does not carry while it carries others),
    n_added (record identifiers not in the S1 while every S1 identifier is carried)."""
    ss, qs = set(sa), set(qa)
    if not ss and not qs:
        return (-1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0)
    jac = len(ss & qs) / len(ss | qs)
    s_unc = [x for x in ss if not _covered(x, qs)]
    q_unc = [x for x in qs if not _covered(x, ss)]
    ed = -1.0
    if s_unc and q_unc:
        ed = min(Levenshtein.distance(x, y) for x in s_unc for y in q_unc)
    n_rep = float(len(s_unc)) if (ss and qs) else 0.0
    n_add = float(len(q_unc)) if (not s_unc and qs) else 0.0
    return (jac, float(len(qs - ss)), float(len(ss - qs)), ed, float(bool(ss) and ss == qs), n_rep, n_add)


def _fuzzy_diff(ta, tb):
    """Tokens of tb not matched (exactly or fuzzily) in ta, and vice versa."""
    sa = set(ta)
    sb = set(tb)
    ea = sa - sb
    eb = sb - sa
    if ea and eb:
        for x in list(eb):
            best = None
            for y in ea:
                if JaroWinkler.normalized_similarity(x, y) >= 0.88 or (
                        len(x) >= 4 and len(y) >= 4 and fuzz.ratio(x, y) >= 75):
                    best = y
                    break
            if best is not None:
                eb.discard(x)
                ea.discard(best)
    return ea, eb


def _pair_rows(args):
    (s_core, s_legal, s_alt, s_concat, s_hn, s_suf, s_nums, s_toks, s_street, s_state, s_city, s_idents,
     q_core, q_legal, q_alt, q_concat, q_hn, q_suf, q_nums, q_toks, q_street, q_state, q_city, q_idents,
     q_half) = args
    rows = np.zeros((len(s_core), N_PY), np.float32)
    extras = []
    misses = []
    for i in range(len(s_core)):
        ta = s_core[i].split()
        tb = q_core[i].split()
        miss, extra = _fuzzy_diff(ta, tb)
        sa, sb = set(ta), set(tb)
        exact = len(sa & sb)
        rows[i, 0] = len(extra)
        rows[i, 1] = len(miss)
        rows[i, 2] = exact / max(1, len(sa | sb))
        rows[i, 3] = float(bool(ta) and bool(tb) and ta[0] == tb[0])
        la = set(s_legal[i].split())
        lb = set(q_legal[i].split())
        # legal-form differences go into the lexicon as prefixed tokens
        extras.append(" ".join(sorted(extra) + ["L:" + t for t in sorted(lb - la)]))
        misses.append(" ".join(sorted(miss) + ["L:" + t for t in sorted(la - lb)]))
        rows[i, 4] = len(tb)
        rows[i, 5] = len(ta)
        rows[i, 6] = float(" ".join(ta) == " ".join(tb))
        rows[i, 7] = float(sorted(ta) == sorted(tb))
        rows[i, 8] = float(sa <= sb) if sa else -1
        rows[i, 9] = float(sb <= sa) if sb else -1
        rows[i, 10] = float(bool(la) and bool(lb) and not (la & lb))
        rows[i, 11] = float(la == lb)
        rows[i, 12] = len(lb - la)
        rows[i, 13] = len(la - lb)
        rel, ld = _hn_rel(s_hn[i], q_hn[i], s_suf[i], q_suf[i])
        rows[i, 14] = rel
        rows[i, 15] = ld
        na = set(s_nums[i].split())
        nb = set(q_nums[i].split())
        rows[i, 16] = len(na & nb) / max(1, len(na | nb)) if (na or nb) else -1
        rows[i, 17] = len(nb - na)
        rows[i, 18] = len(na - nb)
        rows[i, 19] = float(bool(q_hn[i]) and q_hn[i] in na)
        # min numeric distance from q house number to any s number
        if q_hn[i] and na:
            try:
                qh = int(q_hn[i])
                rows[i, 20] = math.log1p(min(abs(qh - int(x)) for x in na if len(x) < 12))
            except ValueError:
                rows[i, 20] = -1
        else:
            rows[i, 20] = -1
        xa = set(s_toks[i].split())
        xb = set(q_toks[i].split())
        xa_alpha = {t for t in xa if not t.isdigit()}
        xb_alpha = {t for t in xb if not t.isdigit()}
        rows[i, 21] = len(xa_alpha & xb_alpha) / max(1, len(xa_alpha | xb_alpha)) if (xa_alpha or xb_alpha) else -1
        rows[i, 22] = len(xb_alpha - xa_alpha)
        rows[i, 23] = len(xa_alpha - xb_alpha)
        st_a, st_b = STATE_GROUP.get(s_state[i], s_state[i]), STATE_GROUP.get(q_state[i], q_state[i])
        rows[i, 24] = -1 if (not st_a or not st_b) else float(st_a == st_b)
        rows[i, 25] = float(q_half[i])
        # alt name (before DBA/formerly) vs s core
        if q_alt[i]:
            rows[i, 26] = fuzz.token_set_ratio(s_core[i], q_alt[i])
        else:
            rows[i, 26] = -1
        if s_alt[i]:
            rows[i, 27] = fuzz.token_set_ratio(s_alt[i], q_core[i])
        else:
            rows[i, 27] = -1
        # extra tokens that are pure legal/generic words vs content words
        rows[i, 28] = sum(1 for t in extra if len(t) <= 2)
        rows[i, 29] = float(bool(s_suf[i]) or bool(q_suf[i])) * float(s_suf[i] != q_suf[i])
        rows[i, 30:37] = _ident_feats(s_idents[i].split(), q_idents[i].split())
    return rows, extras, misses


PY_FEATS = ["n_extra", "n_missing", "tok_jacc", "first_tok_eq", "q_ntok", "s_ntok",
            "name_exact", "name_sorted_eq", "s_subset_q", "q_subset_s", "legal_conflict",
            "legal_equal", "legal_q_only", "legal_s_only", "hn_rel", "hn_logdiff", "num_jacc",
            "num_q_only", "num_s_only", "qhn_in_snums", "qhn_min_logdist", "addr_alpha_jacc",
            "addr_alpha_q_only", "addr_alpha_s_only", "state_eq", "q_half", "alt_q_tset",
            "alt_s_tset", "n_extra_short", "hn_suffix_diff",
            "ident_jacc", "ident_q_only", "ident_s_only", "ident_edit_min", "ident_eq", "ident_replaced", "ident_added"]
N_PY = len(PY_FEATS)

S_COLS = ["name_core", "legal", "name_alt", "concat", "hn", "hn_suf", "nums", "addr_toks", "street",
          "state", "city", "idents"]
Q_COLS = S_COLS + ["half"]


def _vec_str(scorer, a, b, workers):
    return process.cpdist(a, b, scorer=scorer, workers=workers).astype(np.float32)


def cheap_features(cand, s1, q, workers=None):
    """Vectorised (C++) features only: used by the stage-1 filter."""
    workers = workers or (os.cpu_count() or 2)
    si = cand.si.values
    qi = cand.qi.values
    S = {c: s1[c].to_numpy(dtype=object)[si] for c in ("name_core", "concat", "addr_toks", "street", "city")}
    Q = {c: q[c].to_numpy(dtype=object)[qi] for c in ("name_core", "concat", "addr_toks", "street", "city")}
    f = pd.DataFrame(index=cand.index)
    f["name_cos"] = cand.name_cos.values
    f["addr_cos"] = cand.addr_cos.values
    f["n_ratio"] = _vec_str(fuzz.ratio, S["name_core"], Q["name_core"], workers)
    f["n_tsort"] = _vec_str(fuzz.token_sort_ratio, S["name_core"], Q["name_core"], workers)
    f["n_tset"] = _vec_str(fuzz.token_set_ratio, S["name_core"], Q["name_core"], workers)
    f["n_partial"] = _vec_str(fuzz.partial_ratio, S["name_core"], Q["name_core"], workers)
    f["n_jw"] = _vec_str(JaroWinkler.normalized_similarity, S["name_core"], Q["name_core"], workers)
    f["c_ratio"] = _vec_str(fuzz.ratio, S["concat"], Q["concat"], workers)
    f["c_partial"] = _vec_str(fuzz.partial_ratio, S["concat"], Q["concat"], workers)
    f["a_tset"] = _vec_str(fuzz.token_set_ratio, S["addr_toks"], Q["addr_toks"], workers)
    f["a_tsort"] = _vec_str(fuzz.token_sort_ratio, S["addr_toks"], Q["addr_toks"], workers)
    f["street_ratio"] = _vec_str(fuzz.ratio, S["street"], Q["street"], workers)
    f["city_ratio"] = _vec_str(fuzz.ratio, S["city"], Q["city"], workers)
    f["q_native"] = q["native"].values[qi]
    f["q_domain"] = q["domain"].values[qi]
    f["q_addr_empty"] = q["addr_empty"].values[qi]
    f["q_src"] = q["src"].values[qi] if "src" in q else 0
    f["q_street_empty"] = np.array([not x for x in Q["street"]], np.int8)
    f["hn_eq"] = (s1["hn"].values[si] == q["hn"].values[qi]).astype(np.int8)
    f["hn_q_empty"] = (q["hn"].values[qi] == "").astype(np.int8)
    add_context(f, cand)
    return f


def python_features(cand, s1, q, workers=None, chunk=50000):
    """Python-level pair features. Returns (DataFrame, extra-token strings, missing strings)."""
    workers = workers or (os.cpu_count() or 2)
    si = cand.si.values
    qi = cand.qi.values
    S = {c: s1[c].to_numpy(dtype=object)[si] for c in S_COLS}
    Q = {c: q[c].to_numpy(dtype=object)[qi] for c in Q_COLS}
    jobs = []
    for st in range(0, len(cand), chunk):
        sl = slice(st, st + chunk)
        jobs.append(tuple(S[c][sl] for c in S_COLS) + tuple(Q[c][sl] for c in Q_COLS))
    if workers > 1 and len(jobs) > 1:
        with Pool(workers) as pool:
            res = pool.map(_pair_rows, jobs)
    else:
        res = [_pair_rows(j) for j in jobs]
    py = np.vstack([r[0] for r in res]) if res else np.zeros((0, N_PY), np.float32)
    extra = np.array([e for r in res for e in r[1]], dtype=object)
    miss = np.array([e for r in res for e in r[2]], dtype=object)
    return pd.DataFrame(py, columns=PY_FEATS, index=cand.index), extra, miss


def compute_features(cand, s1, q, workers=None, chunk=50000):
    """cheap + python features in one frame. Returns (features, extra, missing)."""
    f = cheap_features(cand, s1, q, workers)
    py, extra, miss = python_features(cand, s1, q, workers, chunk)
    return pd.concat([f, py], axis=1), extra, miss


FREQ_FEATS = ["s_name_rate", "s_name_cnt", "q_name_rate", "q_name_cnt", "s_addr_cnt", "q_addr_cnt", "q_name_top"]


def _addr_key(df):
    """Address identity key: state | identifiers (or house number) | first street token | city."""
    keys = []
    idents = df.idents.values if "idents" in df else df.hn.values
    for st, ids, hn, street, city in zip(df.state.values, idents, df.hn.values, df.street.values, df.city.values):
        num = ids if ids else hn
        if not num:
            keys.append("")
            continue
        keys.append(f"{st}|{num}|{street.split()[0] if street else ''}|{city}")
    return np.array(keys, dtype=object)


def freq_features(cand, s1, q):
    """Label-free ambiguity evidence: how many S1 of the country share the S1's / the record's exact core name
    or address key. Rates are per S1 of the country (comparable between a training slice and the full test);
    counts are logged. q_name_top: how many S1 share the record's name among its candidates (0 when the name
    belongs to no S1)."""
    n = len(cand)
    si, qi = cand.si.values, cand.qi.values
    out = np.zeros((n, len(FREQ_FEATS)), np.float32)
    ctry = s1.country.values
    s_name = s1.name_core.values
    q_name = q.name_core.values
    s_key = _addr_key(s1)
    q_key = _addr_key(q)
    for c in np.unique(ctry):
        m_s = ctry == c
        n_c = max(1, int(m_s.sum()))
        name_cnt = pd.Series(s_name[m_s]).value_counts()
        key_cnt = pd.Series([k for k in s_key[m_s] if k]).value_counts()
        rows = np.flatnonzero(m_s[si])
        sn = pd.Series(s_name[si[rows]]).map(name_cnt).fillna(0).values
        qn = pd.Series(q_name[qi[rows]]).map(name_cnt).fillna(0).values
        sk = pd.Series(s_key[si[rows]]).map(key_cnt).fillna(0).values
        qk = pd.Series(q_key[qi[rows]]).map(key_cnt).fillna(0).values
        out[rows, 0] = np.log((sn + 1) / n_c)
        out[rows, 1] = np.log1p(sn)
        out[rows, 2] = np.log((qn + 1) / n_c)
        out[rows, 3] = np.log1p(qn)
        out[rows, 4] = np.log1p(sk)
        out[rows, 5] = np.log1p(qk)
    # among the record's own candidates: how many S1 carry exactly its name
    d = pd.DataFrame({"qi": qi, "eq": (s_name[si] == q_name[qi]).astype(np.int32)})
    out[:, 6] = d.groupby("qi").eq.transform("sum").values
    return pd.DataFrame(out, columns=FREQ_FEATS, index=cand.index)


ANCHOR_FEATS = ["anc_p1", "anc_strong", "anc_hn_eq_q", "anc_hn_eq_s", "anc_name_q", "anc_nums_jacc_q",
                "s_n_strong", "s_strong_hn_q", "s_strong_hn_s", "q_is_anchor"]


def anchor_features(cand, p1, n_tsort, s1, q, workers=None, strong=0.5):
    """Cluster-consistency evidence for (q, s): compare q with the best *other* candidate of s.

    Sibling distractors disagree with the S1's real records (other house number, extra word) and
    often agree with each other; a singleton S1 has no strong anchor at all.
    """
    workers = workers or (os.cpu_count() or 2)
    n = len(cand)
    si = cand.si.values
    qi = cand.qi.values
    score = p1 + 1e-4 * n_tsort
    order = np.lexsort((-score, si))
    s_sorted = si[order]
    first = np.r_[True, s_sorted[1:] != s_sorted[:-1]]
    grp_start = np.maximum.accumulate(np.where(first, np.arange(n), 0))
    rank = np.arange(n) - grp_start
    top1 = np.full(n, -1)
    top2 = np.full(n, -1)
    top1_of = {}
    top2_of = {}
    for pos in np.flatnonzero(rank <= 1):
        (top1_of if rank[pos] == 0 else top2_of)[s_sorted[pos]] = order[pos]
    top1 = np.array([top1_of.get(s, -1) for s in si])
    top2 = np.array([top2_of.get(s, -1) for s in si])
    anc = np.where(top1 == np.arange(n), top2, top1)
    has = anc >= 0
    a = np.where(has, anc, 0)
    q_hn = q["hn"].values[qi]
    a_hn = q["hn"].values[qi[a]]
    s_hn = s1["hn"].values[si]
    out = np.zeros((n, len(ANCHOR_FEATS)), np.float32)
    out[:, 0] = np.where(has, p1[a], 0)
    out[:, 1] = (has & (p1[a] >= strong)).astype(np.float32)
    out[:, 2] = np.where(has & (q_hn != "") & (a_hn != ""), (q_hn == a_hn).astype(np.float32), -1)
    out[:, 3] = np.where(has & (s_hn != "") & (a_hn != ""), (s_hn == a_hn).astype(np.float32), -1)
    qn = q["name_core"].to_numpy(dtype=object)[qi]
    an = q["name_core"].to_numpy(dtype=object)[qi[a]]
    out[:, 4] = np.where(has, process.cpdist(qn, an, scorer=fuzz.token_sort_ratio, workers=workers), -1)
    qnum = q["nums"].values[qi]
    anum = q["nums"].values[qi[a]]
    jac = np.full(n, -1.0, np.float32)
    for i in np.flatnonzero(has):
        x, y = qnum[i], anum[i]
        if x or y:
            sx, sy = set(x.split()), set(y.split())
            jac[i] = len(sx & sy) / len(sx | sy)
    out[:, 5] = jac
    st = p1 >= strong
    d = pd.DataFrame({"si": si, "hn": q_hn, "st": st.astype(np.int32)})
    n_st = d.groupby("si").st.transform("sum").values - st
    d["key"] = d.si.astype(str) + "|" + d.hn
    same_hn_st = d.groupby("key").st.transform("sum").values - st
    out[:, 6] = n_st
    out[:, 7] = np.where((n_st > 0) & (q_hn != ""), same_hn_st / np.maximum(n_st, 1), -1)
    d2 = pd.DataFrame({"key": pd.Series(si).astype(str).values + "|" + s_hn})
    cnt = d.groupby("key").st.sum()
    s_same = d2.key.map(cnt).fillna(0).values - (st & (q_hn == s_hn))
    out[:, 8] = np.where((n_st > 0) & (s_hn != ""), s_same / np.maximum(n_st, 1), -1)
    out[:, 9] = (top1 == np.arange(n)).astype(np.float32)
    return pd.DataFrame(out, columns=ANCHOR_FEATS, index=cand.index)


PEER_FEATS = ["peer_ex_share", "peer_ex_share_p", "peer_num_share", "peer_num_share_p", "peer_both",
              "q_n_extra_exact", "q_n_newnum", "s_n_cand_all"]


def peer_features(cand, p1, s1, q):
    """Collective evidence among all candidates of the same S1.

    A sibling distractor is a hidden business with its own records, so its modifier word
    ("exports") and its new house number repeat across several of them; per-record noise
    (typos, 'services', zero-padding) does not. For each (q, s):
      peer_ex_share   - max over q's extra name tokens of how many OTHER candidates of s carry it
      peer_num_share  - same for q's numbers that are not in the S1 address
      *_p             - same, weighted by the peers' stage-1 probability
      peer_both       - number of other candidates sharing both an extra token and a new number
    """
    n = len(cand)
    si = cand.si.values
    qi = cand.qi.values
    s_core = s1["name_core"].values
    s_nums = s1["nums"].values
    q_core = q["name_core"].values
    q_nums = q["nums"].values
    ex = [None] * n
    nn = [None] * n
    for i in range(n):
        a = set(s_core[si[i]].split())
        ex[i] = frozenset(t for t in q_core[qi[i]].split() if t not in a and len(t) > 2 and not t.isdigit())
        b = set(s_nums[si[i]].split())
        nn[i] = frozenset(x for x in q_nums[qi[i]].split() if x not in b)
    out = np.zeros((n, len(PEER_FEATS)), np.float32)
    order = np.argsort(si, kind="stable")
    s_sorted = si[order]
    bounds = np.flatnonzero(np.r_[True, s_sorted[1:] != s_sorted[:-1], True])
    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        idx = order[b0:b1]
        if len(idx) == 1:
            i = idx[0]
            out[i, 5], out[i, 6], out[i, 7] = len(ex[i]), len(nn[i]), 1
            continue
        tok_c, tok_p, num_c, num_p = {}, {}, {}, {}
        big = len(idx) > 300  # popular names at test time: thousands of candidates, the exact pair loop is quadratic
        both_c = {}
        for i in idx:
            for t in ex[i]:
                tok_c[t] = tok_c.get(t, 0) + 1
                tok_p[t] = tok_p.get(t, 0.0) + p1[i]
            for x in nn[i]:
                num_c[x] = num_c.get(x, 0) + 1
                num_p[x] = num_p.get(x, 0.0) + p1[i]
            if big and ex[i] and nn[i]:
                for t in ex[i]:
                    for x in nn[i]:
                        both_c[(t, x)] = both_c.get((t, x), 0) + 1
        for i in idx:
            if ex[i]:
                out[i, 0] = max(tok_c[t] for t in ex[i]) - 1
                out[i, 1] = max(tok_p[t] for t in ex[i]) - p1[i]
            if nn[i]:
                out[i, 2] = max(num_c[x] for x in nn[i]) - 1
                out[i, 3] = max(num_p[x] for x in nn[i]) - p1[i]
            if ex[i] and nn[i]:
                if big:
                    out[i, 4] = max(both_c.get((t, x), 0) for t in ex[i] for x in nn[i]) - 1
                else:
                    out[i, 4] = sum(1 for j in idx if j != i and (ex[i] & ex[j]) and (nn[i] & nn[j]))
            out[i, 5], out[i, 6], out[i, 7] = len(ex[i]), len(nn[i]), len(idx)
    return pd.DataFrame(out, columns=PEER_FEATS, index=cand.index)


def add_context(f, cand, score=None, prefix=""):
    """Features describing competition among candidates of the same query / same S1."""
    if score is None:
        score = (f["n_tsort"].values / 100.0 + f["a_tset"].values / 100.0 + f["name_cos"].values +
                 f["addr_cos"].values)
        f["blk_score"] = score.astype(np.float32)
    g = pd.DataFrame({"qi": cand.qi.values, "si": cand.si.values, "s": score})
    grp = g.groupby("qi").s
    mx = grp.transform("max").values
    # best score among OTHER candidates of this query
    srt = g.sort_values(["qi", "s"], ascending=[True, False])
    second = srt.groupby("qi").s.nth(1)
    second_map = pd.Series(second.values, index=srt.loc[second.index, "qi"].values)
    sec = g.qi.map(second_map).fillna(0).values
    other_best = np.where(score >= mx, sec, mx)
    f[prefix + "q_rank"] = grp.rank(ascending=False, method="first").values.astype(np.float32)
    f[prefix + "q_gap"] = (score - other_best).astype(np.float32)
    f[prefix + "q_ncand"] = grp.transform("size").values.astype(np.float32)
    gs = g.groupby("si").s
    f[prefix + "s_rank"] = gs.rank(ascending=False, method="first").values.astype(np.float32)
    f[prefix + "s_ncand"] = gs.transform("size").values.astype(np.float32)
    f[prefix + "s_gap_top"] = (score - gs.transform("max").values).astype(np.float32)
