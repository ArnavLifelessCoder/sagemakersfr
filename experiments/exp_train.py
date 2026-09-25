import os, sys, time, pickle, zlib
sys.path.insert(0, r"C:\Users\Arnav Gawade(pro)\OneDrive\Desktop\amazon ml\code\business_entity_resolution")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from src.prep import parse_frame, learn_translit
from src.blocking import generate_candidates
from src.features import compute_features
from src.data import gt_pairs
from src import model as M

DEV = os.path.join(os.path.dirname(__file__), "dev")
K = int(os.environ.get("K", 5))
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def load():
    cache = f"{DEV}/parsed.pkl"
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    s1 = pd.read_parquet(f"{DEV}/src1.parquet")
    q = pd.concat([pd.read_parquet(f"{DEV}/src2.parquet"), pd.read_parquet(f"{DEV}/src3.parquet")], ignore_index=True)
    s1["src"] = 1
    q["src"] = q.entity_id.str[1].astype(int)
    pairs = gt_pairs(pd.read_parquet(f"{DEV}/gt.parquet"))
    name_of = dict(zip(s1.entity_id, s1.business_name))
    qname = dict(zip(q.entity_id, q.business_name))
    tr = learn_translit([name_of[a] for a in pairs.s1], [qname.get(b, "") for b in pairs.other])
    s1 = parse_frame(s1, tr, workers=6)
    q = parse_frame(q, tr, workers=6)
    log("parsed")
    c = generate_candidates(s1, q, k_name=10, k_addr=10, k_name_nostate=10, threads=8, log=log)
    log("blocked", len(c))
    res = (s1, q, c, pairs)
    pickle.dump(res, open(cache, "wb"))
    return res


def prune(c, k):
    rn = c.groupby("qi").name_cos.rank(ascending=False, method="first")
    ra = c.groupby("qi").addr_cos.rank(ascending=False, method="first")
    keep = ((rn <= k) & (c.name_cos > 0)) | ((ra <= k) & (c.addr_cos > 0))
    return c[keep.values].reset_index(drop=True)


def main():
    s1, q, c, pairs = load()
    c = prune(c, K)
    truth = dict(zip(pairs.other, pairs.s1))
    s1_ids = s1.entity_id.values
    q_ids = q.entity_id.values
    y = (pd.Series(q_ids[c.qi.values]).map(truth).values == s1_ids[c.si.values]).astype(np.int8)
    log(f"K={K} pairs {len(c)} per-q {len(c) / len(q):.2f} recall {y.sum() / len(pairs):.5f}")
    f, extra, miss = compute_features(c, s1, q, workers=6)
    log("features", f.shape)
    ctry = s1.country.values[c.si.values]
    for cc in np.unique(ctry):
        voc = M.name_vocab(s1.name_core.values[s1.country.values == cc])
        mk = ctry == cc
        arr = M.vocab_features(extra[mk], miss[mk], *voc)
        for k, col in enumerate(M.VOCAB_FEATS):
            if col not in f:
                f[col] = np.float32(0)
            f.loc[mk, col] = arr[:, k]
    # split S1 entities into train (A) / valid (B)
    is_b = np.array([zlib.crc32(x.encode()) % 2 == 1 for x in s1_ids])
    pb = is_b[c.si.values]
    hard = (f.n_tset.values >= 80)
    tr = ~pb
    fa = f[tr].copy()
    M.oof_lexicon_features(fa, extra[tr], miss[tr], y[tr], s1_ids[c.si.values][tr], hard[tr])
    lex = M.learn_lexicon(extra[tr], miss[tr], y[tr], hard[tr])
    fb = f[pb].copy()
    M.add_lex(fb, extra[pb], miss[pb], lex)
    log("lexicon sizes", {k: len(v) for k, v in lex.items()})
    xs = sorted(lex["extra"].items(), key=lambda t: t[1])
    log("most distractor-like extra tokens", xs[:25])
    log("most benign extra tokens", xs[-25:])
    feats = list(fa.columns)
    bst = M.train_lgb(fa[feats], y[tr], fb[feats], y[pb], rounds=3000)
    pv = bst.predict(fb[feats], num_iteration=bst.best_iteration)
    imp = sorted(zip(bst.feature_importance("gain"), feats), reverse=True)
    log("top features", [(n, int(g)) for g, n in imp[:25]])
    # evaluation on B entities: every query may be assigned to a B entity
    qi_b = c.qi.values[pb]
    si_b = c.si.values[pb]
    # queries assigned elsewhere (to an A entity with higher prob) must not count for B:
    pa = bst.predict(fa[feats], num_iteration=bst.best_iteration)
    all_qi = np.concatenate([c.qi.values[tr], qi_b])
    all_si = np.concatenate([c.si.values[tr], si_b])
    all_p = np.concatenate([pa, pv])
    b_ents = s1_ids[is_b]
    truth_map = M.to_mapping(pairs.s1.values, pairs.other.values)
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        aq, as_ = M.assign(all_qi, all_si, all_p, t)
        pred = M.to_mapping(s1_ids[as_], q_ids[aq])
        log(f"thr {t:.2f}  F0.5(B) = {M.fbeta_macro(pred, truth_map, b_ents):.5f}")
    pickle.dump((c, f, extra, miss, y, is_b, lex, feats), open(f"{DEV}/feat.pkl", "wb"))
    bst.save_model(f"{DEV}/lgb.txt")


if __name__ == "__main__":
    main()
