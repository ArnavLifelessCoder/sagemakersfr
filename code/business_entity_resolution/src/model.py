"""Token lexicon, LightGBM matcher, assignment and the challenge metric."""
import collections
import math

import lightgbm as lgb
import numpy as np
import pandas as pd

LEX_FEATS = ["lex_extra_max", "lex_extra_min", "lex_extra_sum", "lex_miss_max", "lex_miss_min",
             "lex_miss_sum"]


# ----------------------------------------------------------------------------
# Lexicon: how "dangerous" a token is when present in only one of the two names
# ----------------------------------------------------------------------------
def learn_lexicon(extra, miss, y, hard_mask, min_count=5):
    """extra/miss: arrays of space-joined token strings; y: labels.

    Only 'hard' pairs (names already very similar) are used, so the lexicon captures
    which word differences separate true matches from near-duplicate distractors.
    Value = smoothed log-odds of the token appearing in a true pair vs a false pair.
    """
    lex = {}
    for kind, arr in (("extra", extra), ("missing", miss)):
        pos, neg = collections.Counter(), collections.Counter()
        for s, lab in zip(arr[hard_mask], y[hard_mask]):
            if not s:
                continue
            (pos if lab else neg).update(s.split())
        P = sum(pos.values()) + 1
        N = sum(neg.values()) + 1
        base = math.log(P / N)
        d = {}
        for t in set(pos) | set(neg):
            p, n = pos[t], neg[t]
            if p + n >= min_count:
                d[t] = math.log((p + 1) / (n + 1)) - base
        lex[kind] = d
    return lex


def lex_features(extra, miss, lex):
    out = np.zeros((len(extra), 6), np.float32)
    lx, lm = lex.get("extra", {}), lex.get("missing", {})
    for i, (e, m) in enumerate(zip(extra, miss)):
        if e:
            v = [lx.get(t, 0.0) for t in e.split()]
            out[i, 0], out[i, 1], out[i, 2] = max(v), min(v), sum(v)
        if m:
            v = [lm.get(t, 0.0) for t in m.split()]
            out[i, 3], out[i, 4], out[i, 5] = max(v), min(v), sum(v)
    return out


def add_lex(f, extra, miss, lex):
    arr = lex_features(extra, miss, lex)
    for k, c in enumerate(LEX_FEATS):
        f[c] = arr[:, k]


def oof_lexicon_features(f, extra, miss, y, groups, hard_mask, n_folds=2, seed=0):
    """Out-of-fold lexicon features for training rows (grouped by S1 id)."""
    rng = np.random.RandomState(seed)
    ug = pd.unique(groups)
    fold_of = dict(zip(ug, rng.randint(0, n_folds, len(ug))))
    folds = np.array([fold_of[g] for g in groups])
    arr = np.zeros((len(f), 6), np.float32)
    for k in range(n_folds):
        tr = folds != k
        lex = learn_lexicon(extra[tr], miss[tr], y[tr], hard_mask[tr])
        te = np.flatnonzero(~tr)
        arr[te] = lex_features(extra[te], miss[te], lex)
    for k, c in enumerate(LEX_FEATS):
        f[c] = arr[:, k]


# ----------------------------------------------------------------------------
# Vocabulary features (label-free, transfer to unseen countries such as France)
# ----------------------------------------------------------------------------
VOCAB_FEATS = ["ex_vocab_max", "ex_vocab_min", "ex_n_vocab", "ex_n_rare", "mi_vocab_max",
               "mi_vocab_min", "mi_n_vocab", "mi_n_rare", "substitution"]


def name_vocab(core_names):
    """Document frequency of core-name tokens among S1 records of one country."""
    cnt = collections.Counter()
    for s in core_names:
        if s:
            cnt.update(set(s.split()))
    return cnt, max(1, len(core_names))


def vocab_features(extra, miss, vocab, n_docs, common=1e-4):
    """Distractors add/replace *real* vocabulary words; true-match noise adds typos/junk.

    For the extra (only in S2/S3 name) and missing (only in S1 name) content tokens, report
    the log document frequency of the token among S1 names of the same country.
    """
    out = np.zeros((len(extra), len(VOCAB_FEATS)), np.float32)
    thr = common * n_docs
    ln = math.log(n_docs)
    for i, (e, m) in enumerate(zip(extra, miss)):
        ne = nm = 0
        for j, s in ((0, e), (4, m)):
            toks = [t for t in s.split() if not t.startswith("L:")] if s else []
            if not toks:
                out[i, j:j + 2] = -1
                continue
            fr = [vocab.get(t, 0) for t in toks]
            lf = [math.log1p(x) - ln for x in fr]
            out[i, j] = max(lf)
            out[i, j + 1] = min(lf)
            nv = sum(1 for x in fr if x >= thr)
            out[i, j + 2] = nv
            out[i, j + 3] = len(fr) - nv
            if j == 0:
                ne = nv
            else:
                nm = nv
        out[i, 8] = float(ne > 0 and nm > 0)
    return out


def add_vocab(f, extra, miss, vocab, n_docs):
    arr = vocab_features(extra, miss, vocab, n_docs)
    for k, c in enumerate(VOCAB_FEATS):
        f[c] = arr[:, k]


# ----------------------------------------------------------------------------
# LightGBM
# ----------------------------------------------------------------------------
DEFAULT_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=50,
                      feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                      verbose=-1, num_threads=0)


def train_lgb(X, y, Xv=None, yv=None, rounds=1500, params=None, log_every=100):
    p = dict(DEFAULT_PARAMS, **(params or {}))
    dtr = lgb.Dataset(X, y, free_raw_data=True)
    valid = [dtr]
    names = ["train"]
    if Xv is not None:
        valid.append(lgb.Dataset(Xv, yv, reference=dtr))
        names.append("valid")
    cb = [lgb.log_evaluation(log_every)] if log_every else []
    if Xv is not None:
        cb.append(lgb.early_stopping(100, verbose=False))
    return lgb.train(p, dtr, rounds, valid_sets=valid, valid_names=names, callbacks=cb)


# ----------------------------------------------------------------------------
# Assignment + metric
# ----------------------------------------------------------------------------
def assign(qi, si, p, threshold):
    """Each query goes to its highest-probability S1 if p >= threshold. Returns (qi, si)."""
    d = pd.DataFrame({"qi": qi, "si": si, "p": p})
    d = d.sort_values("p", ascending=False).drop_duplicates("qi")
    d = d[d.p >= threshold]
    return d.qi.values, d.si.values


def fbeta_macro(pred, truth, s1_ids, beta=0.5):
    """pred/truth: dict s1 -> set(ids). Macro average over s1_ids (singletons included)."""
    b2 = beta * beta
    tot = 0.0
    for s in s1_ids:
        P = pred.get(s, set())
        T = truth.get(s, set())
        if not T and not P:
            tot += 1.0
            continue
        if not T or not P:
            continue
        tp = len(P & T)
        if tp == 0:
            continue
        prec = tp / len(P)
        rec = tp / len(T)
        tot += (1 + b2) * prec * rec / (b2 * prec + rec)
    return tot / max(1, len(s1_ids))


def to_mapping(s1_ids_of_pairs, q_ids_of_pairs):
    m = collections.defaultdict(set)
    for s, q in zip(s1_ids_of_pairs, q_ids_of_pairs):
        m[s].add(q)
    return m
