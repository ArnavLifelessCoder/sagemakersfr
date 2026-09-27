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


def oof_lexicon_features(f, extra, miss, y, groups, hard_mask, n_folds=2, seed=0, drop_neg=0.0):
    """Out-of-fold lexicon features for training rows (grouped by S1 id).

    drop_neg: share of distractor-like (negative) lexicon entries hidden in each fold, so the
    model also learns to decide on words the lexicon does not know (new/foreign modifiers),
    using the label-free modifier features instead."""
    rng = np.random.RandomState(seed)
    ug = pd.unique(groups)
    fold_of = dict(zip(ug, rng.randint(0, n_folds, len(ug))))
    folds = np.array([fold_of[g] for g in groups])
    arr = np.zeros((len(f), 6), np.float32)
    for k in range(n_folds):
        tr = folds != k
        lex = learn_lexicon(extra[tr], miss[tr], y[tr], hard_mask[tr])
        if drop_neg > 0:
            for kind in lex:
                lex[kind] = {t: v for t, v in lex[kind].items()
                             if not (v < 0 and not t.startswith("L:") and rng.rand() < drop_neg)}
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


MOD_FEATS = ["ex_mod_max", "ex_mod_n", "ex_qdf_max", "mi_mod_max", "mi_s1df_max", "mod_substitution"]


def modifier_features(extra, miss, s_vocab, n_s, q_vocab, n_q, min_q=30, mod_ratio=3.0):
    """Label-free 'distractor word' evidence, computed on the data being scored.

    Distractor generators append/swap modifier words ("eastgate", "holding", "groupe", "bakery")
    that are far more frequent in S2/S3 names than in S1 names; typos are rare everywhere.
    ratio(t) = log( (dfQ(t)/nQ + e) / (dfS1(t)/nS1 + e) ), only for tokens with dfQ >= min_q.
    """
    out = np.zeros((len(extra), len(MOD_FEATS)), np.float32)
    eps = 1.0 / max(n_s, 1)
    lr = math.log(mod_ratio)
    for i, (e, m) in enumerate(zip(extra, miss)):
        best = 0.0
        n_mod = 0
        qmax = 0.0
        for t in (e.split() if e else ()):
            if t.startswith("L:"):
                continue
            dq = q_vocab.get(t, 0)
            qmax = max(qmax, math.log1p(dq))
            if dq < min_q:
                continue
            r = math.log((dq / n_q + eps) / (s_vocab.get(t, 0) / n_s + eps))
            best = max(best, r)
            n_mod += r > lr
        mbest = 0.0
        smax = 0.0
        for t in (m.split() if m else ()):
            if t.startswith("L:"):
                continue
            ds = s_vocab.get(t, 0)
            smax = max(smax, math.log1p(ds))
            dq = q_vocab.get(t, 0)
            if dq >= min_q:
                mbest = max(mbest, math.log((dq / n_q + eps) / (ds / n_s + eps)))
        out[i] = (best, n_mod, qmax, mbest, smax, float(n_mod > 0 and smax >= math.log1p(min_q)))
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


def train_lgb(X, y, Xv=None, yv=None, rounds=1500, params=None, log_every=100, w=None, wv=None):
    p = dict(DEFAULT_PARAMS, **(params or {}))
    dtr = lgb.Dataset(X, y, weight=w, free_raw_data=True)
    valid = [dtr]
    names = ["train"]
    if Xv is not None:
        valid.append(lgb.Dataset(Xv, yv, weight=wv, reference=dtr))
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


def _pb_pmf(ps):
    """Poisson-binomial PMF of the number of successes among independent Bernoulli(p)."""
    pmf = np.zeros(len(ps) + 1)
    pmf[0] = 1.0
    for i, p in enumerate(ps):
        pmf[1:i + 2] = pmf[1:i + 2] * (1 - p) + pmf[0:i + 1] * p
        pmf[0] *= (1 - p)
    return pmf


def expected_f05(ps, m, p_unfound=0.0):
    """Expected F0.5 of predicting the m most probable of the records ps (sorted desc), truths independent.

    F = 1.25 tp / (m + 0.25 (tp + fn)); tp ~ PB(ps[:m]), fn ~ PB(ps[m:]) plus one hidden true record with
    probability p_unfound (blocking misses). m = 0 scores 1 only when nothing is true."""
    if m == 0:
        return float(np.prod(1 - ps)) * (1 - p_unfound)
    tp_pmf = _pb_pmf(ps[:m])
    fn_pmf = _pb_pmf(ps[m:]) if m < len(ps) else np.array([1.0])
    if p_unfound > 0:
        fn_pmf = np.convolve(fn_pmf, [1 - p_unfound, p_unfound])
    tp = np.arange(m + 1)[:, None]
    fn = np.arange(len(fn_pmf))[None, :]
    f = np.where(tp > 0, 1.25 * tp / np.maximum(m + 0.25 * (tp + fn), 1e-9), 0.0)
    return float((tp_pmf[:, None] * fn_pmf[None, :] * f).sum())


def assign_expected(qi, si, p, p_unfound=0.02, min_p=0.05, odds_div=1.0, sure=0.985):
    """Per-S1 expected-F0.5 decision on top of the argmax assignment.

    Each record goes to its most probable S1; then, per S1, the prefix (by p) of its records with the
    highest expected F0.5 is kept, the empty prediction included. odds_div > 1 shrinks every probability
    (odds / odds_div) before deciding, a per-country calibration knob. Returns (qi, si) of kept records."""
    d = pd.DataFrame({"qi": qi, "si": si, "p": p})
    d = d.sort_values("p", ascending=False).drop_duplicates("qi")
    d = d[d.p >= min_p]
    if odds_div != 1.0:
        d["p"] = d.p / (d.p + odds_div * (1 - d.p))
        d = d[d.p >= min_p]
    d = d.sort_values(["si", "p"], ascending=[True, False])
    g_min = d.groupby("si").p.transform("min").values
    easy = g_min >= sure  # every record sure: keep all (the rule always would)
    out_q = [d.qi.values[easy]]
    out_s = [d.si.values[easy]]
    rest = d[~easy]
    if len(rest):
        si_r = rest.si.values
        bounds = np.flatnonzero(np.r_[True, si_r[1:] != si_r[:-1], True])
        pv = rest.p.values.astype(float)
        qv = rest.qi.values
        ks, kq = [], []
        for b0, b1 in zip(bounds[:-1], bounds[1:]):
            ps = pv[b0:b1]
            best_m, best_e = 0, expected_f05(ps, 0, p_unfound)
            for m in range(1, len(ps) + 1):
                e = expected_f05(ps, m, p_unfound)
                if e > best_e:
                    best_m, best_e = m, e
            if best_m:
                ks.append(si_r[b0:b0 + best_m])
                kq.append(qv[b0:b0 + best_m])
        if ks:
            out_q.append(np.concatenate(kq))
            out_s.append(np.concatenate(ks))
    return np.concatenate(out_q), np.concatenate(out_s)


def fbeta_macro(pred, truth, s1_ids, beta=0.5, fp_weight=1.0):
    """pred/truth: dict s1 -> set(ids). Macro average over s1_ids (singletons included).

    fp_weight > 1 emulates a test set with proportionally more distractors: each false merge is
    counted fp_weight times (test holds ~1.7x more sibling distractors per S1 than training)."""
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
        prec = tp / (tp + fp_weight * (len(P) - tp))
        rec = tp / len(T)
        tot += (1 + b2) * prec * rec / (b2 * prec + rec)
    return tot / max(1, len(s1_ids))


def to_mapping(s1_ids_of_pairs, q_ids_of_pairs):
    m = collections.defaultdict(set)
    for s, q in zip(s1_ids_of_pairs, q_ids_of_pairs):
        m[s].add(q)
    return m
