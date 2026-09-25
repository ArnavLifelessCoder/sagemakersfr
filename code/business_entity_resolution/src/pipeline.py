"""End-to-end pipeline: train on the labelled training split, predict the test split.

    # 1) train (writes artifacts: models, lexicon, transliteration table, thresholds)
    python -m src.pipeline train   --data-dir DATASET_DIR --work ARTIFACT_DIR
    # 2) predict (writes output/matching_results.tsv and output/candidate_pairs.tsv)
    python -m src.pipeline predict --data-dir DATASET_DIR --work ARTIFACT_DIR --out OUTPUT_DIR

DATASET_DIR is the folder holding train/ and test/ (student_resource/dataset).

Stages
  1. parse / normalise every record                                  (prep.py, normalize.py)
  2. blocking: per (country, state) top-K by name and address TF-IDF  (blocking.py)
  3. stage-1 filter: small LightGBM on cheap vectorised features; pairs below a
     recall-preserving threshold are dropped. The survivors are the candidate set
     written to candidate_pairs.tsv.
  4. stage-2 matcher: LightGBM on the full feature set (incl. token lexicon,
     vocabulary and stage-1 context features)
  5. assignment: each S2/S3 record joins its best S1 if p >= threshold
"""
import argparse
import gc
import json
import os
import time
import zlib

import numpy as np
import pandas as pd

from . import model as M
from .blocking import iter_candidates
from .data import read_sources, read_ground_truth, gt_pairs, write_id_lists
from .features import cheap_features, python_features, add_context, anchor_features
from .prep import parse_frame, learn_translit, extend_translit

T0 = time.time()
DEFAULT_CFG = dict(k_block=10, k_keep=5, threads=-1, workers=os.cpu_count() or 4, train_frac=0.25,
                   val_pct=20, seed=0, rounds=3000, s1_recall=0.9995, refit_full=False, lex_drop=0.4)


def log(*a):
    print(f"[{time.time() - T0:8.1f}s]", *a, flush=True)


def prune(c, k):
    """Keep, per query, the top-k S1 by each channel cosine."""
    if len(c) == 0:
        return c
    rn = c.groupby("qi").name_cos.rank(ascending=False, method="first").values
    ra = c.groupby("qi").addr_cos.rank(ascending=False, method="first").values
    keep = ((rn <= k) & (c.name_cos.values > 0)) | ((ra <= k) & (c.addr_cos.values > 0))
    return c[keep].reset_index(drop=True)


def slim(df):
    """Drop raw text columns to save memory."""
    return df.drop(columns=["business_name", "business_address"], errors="ignore")


# ----------------------------------------------------------------------------
# shared stages
# ----------------------------------------------------------------------------
def block_and_cheap(s1, q, cfg, stage1=None):
    """Blocking + cheap features (+ stage-1 filtering when a stage-1 model is given).

    Returns (cand DataFrame[qi, si], F1 DataFrame, p1 array or None)."""
    cands, feats, ps = [], [], []
    for part, cand in iter_candidates(s1, q, k_name=cfg["k_block"], k_addr=cfg["k_block"],
                                      k_name_nostate=cfg["k_block"], threads=cfg["threads"], log=log):
        cand = prune(cand, cfg["k_keep"])
        if len(cand) == 0:
            continue
        f1 = cheap_features(cand, s1, q, workers=cfg["workers"])
        if stage1 is not None:
            bst1, feats1, thr1 = stage1
            p1 = bst1.predict(f1[feats1].values.astype(np.float32), num_threads=cfg["threads"])
            keep = p1 >= thr1
            cand, f1, p1 = cand[keep].reset_index(drop=True), f1[keep].reset_index(drop=True), p1[keep]
            ps.append(p1.astype(np.float32))
        cands.append(cand[["qi", "si"]])
        feats.append(f1)
    cand = pd.concat(cands, ignore_index=True)
    F1 = pd.concat(feats, ignore_index=True)
    p1 = np.concatenate(ps) if ps else None
    # fallback partitions can repeat a pair: keep the first occurrence
    dup = cand.duplicated(["qi", "si"]).values
    if dup.any():
        cand, F1 = cand[~dup].reset_index(drop=True), F1[~dup].reset_index(drop=True)
        p1 = p1[~dup] if p1 is not None else None
    return cand, F1, p1


def stage2_features(cand, F1, p1, s1, q, cfg, anc=None, vocabs=None):
    """Full feature frame for stage 2. Returns (F, extra, miss).

    anc: precomputed anchor features for these rows (needed when scoring in slices, because
    anchors must see every candidate of an S1). vocabs: {country: (s1 vocab, n, q vocab, n)}."""
    py, extra, miss = python_features(cand, s1, q, workers=cfg["workers"])
    F = pd.concat([F1.reset_index(drop=True), py.reset_index(drop=True)], axis=1)
    ctry = s1.country.values[cand.si.values]
    vocabs = vocabs or country_vocabs(s1, q)
    arr = np.zeros((len(F), len(M.VOCAB_FEATS)), np.float32)
    mod = np.zeros((len(F), len(M.MOD_FEATS)), np.float32)
    for c in np.unique(ctry):
        sv, ns, qv, nq = vocabs[c]
        mk = ctry == c
        arr[mk] = M.vocab_features(extra[mk], miss[mk], sv, ns)
        mod[mk] = M.modifier_features(extra[mk], miss[mk], sv, ns, qv, nq)
    for k, col in enumerate(M.VOCAB_FEATS):
        F[col] = arr[:, k]
    for k, col in enumerate(M.MOD_FEATS):
        F[col] = mod[:, k]
    F["p1"] = p1
    if anc is None:
        anc = group_features(cand, p1, F1, s1, q, cfg)
    for col in anc.columns:
        F[col] = anc[col].values
    return F, extra, miss


def group_features(cand, p1, F1, s1, q, cfg):
    """Features that depend on all candidates of a query / an S1 (compute on the full set)."""
    g = anchor_features(cand, p1, F1.n_tsort.values, s1, q, workers=cfg["workers"])
    add_context(g, cand, score=p1, prefix="p1_")
    return g


def country_vocabs(s1, q):
    """Token document frequencies of core names, per country, for S1 and for S2/S3."""
    out = {}
    for c in np.unique(s1.country.values):
        sv, ns = M.name_vocab(s1.name_core.values[s1.country.values == c])
        qv, nq = M.name_vocab(q.name_core.values[q.country.values == c])
        out[c] = (sv, ns, qv, nq)
    return out


# ----------------------------------------------------------------------------
# TRAIN
# ----------------------------------------------------------------------------
def select_training_slice(src, pairs, frac, seed):
    """Pick a random subset of (country, state) partitions holding ~frac of S1 entities,
    all ground-truth matches of their S1, unmatched records in the same partitions and a
    proportional sample of empty-address unmatched records."""
    is_s1 = src.src.values == 1
    parts = src[is_s1].groupby(["country", "state"]).size().reset_index(name="n")
    parts = parts[parts.state != ""].sample(frac=1.0, random_state=seed)
    target = frac * is_s1.sum()
    chosen, tot = set(), 0
    for c, s, n in parts.itertuples(index=False):
        if tot >= target:
            break
        chosen.add((c, s))
        tot += n
    key = pd.Series(list(zip(src.country.values, src.state.values)))
    in_part = key.isin(chosen).values
    ids = src.entity_id.values
    s1_sel = set(ids[is_s1 & in_part])
    matched_all = set(pairs.other.values)
    keep_other = set(pairs.other.values[pairs.s1.isin(s1_sel).values])
    unmatched = ~pd.Series(ids).isin(matched_all).values
    is_q = ~is_s1
    empty = src.addr_empty.values == 1
    rnd = np.random.RandomState(seed + 1).rand(len(src))
    keep = ((is_s1 & in_part) | pd.Series(ids).isin(keep_other).values |
            (is_q & unmatched & in_part) | (is_q & unmatched & empty & (rnd < frac)))
    return src[keep].reset_index(drop=True), s1_sel


def load_train(a, cfg):
    """Returns (records DataFrame with src column, pairs, translit table, selected S1 ids)."""
    if a.dev:
        s1 = pd.read_parquet(os.path.join(a.dev, "src1.parquet"))
        q = pd.concat([pd.read_parquet(os.path.join(a.dev, f"src{k}.parquet")) for k in (2, 3)],
                      ignore_index=True)
        s1["src"] = 1
        q["src"] = q.entity_id.str[1].astype(int)
        src = pd.concat([s1, q], ignore_index=True)
        pairs = gt_pairs(pd.read_parquet(os.path.join(a.dev, "gt.parquet")))
        name_of = dict(zip(src.entity_id.values, src.business_name.values))
        tr = learn_translit([name_of.get(x, "") for x in pairs.s1.values],
                            [name_of.get(x, "") for x in pairs.other.values])
        return src, pairs, tr, set(s1.entity_id.values)
    src = read_sources(a.data_dir, "train")
    pairs = gt_pairs(read_ground_truth(os.path.join(a.data_dir, "train", "train_ground_truth.tsv")))
    log("loaded train", len(src), "pairs", len(pairs))
    name_of = dict(zip(src.entity_id.values, src.business_name.values))
    tr = learn_translit([name_of.get(x, "") for x in pairs.s1.values],
                        [name_of.get(x, "") for x in pairs.other.values])
    del name_of
    from .make_dev_subset import state_of
    src["state"] = [state_of(x, c) if x else "" for x, c in zip(src.business_address.values, src.country.values)]
    src["addr_empty"] = (src.business_address.str.strip() == "").astype(np.int8)
    sl, s1_sel = select_training_slice(src, pairs, cfg["train_frac"], cfg["seed"])
    del src
    gc.collect()
    return sl.drop(columns=["state", "addr_empty"]), pairs, tr, s1_sel


def _is_val(ids, pct):
    return np.array([zlib.crc32(x.encode()) % 100 < pct for x in ids])


def cmd_train(a):
    cfg = {**DEFAULT_CFG, **(json.load(open(a.config)) if a.config else {})}
    if a.threads:
        cfg["threads"] = a.threads
    if a.workers:
        cfg["workers"] = a.workers
    os.makedirs(a.work, exist_ok=True)
    src, pairs, tr, s1_sel = load_train(a, cfg)
    json.dump(tr, open(os.path.join(a.work, "translit.json"), "w", encoding="utf-8"), ensure_ascii=False)
    log("translit entries", len(tr), "| training records", len(src), "S1", len(s1_sel))
    src = slim(parse_frame(src, extend_translit(tr, src.business_name.values), workers=cfg["workers"]))
    log("parsed")
    s1 = src[src.src == 1].reset_index(drop=True)
    q = src[src.src != 1].reset_index(drop=True)
    del src
    gc.collect()
    truth = pd.Series(pairs.s1.values, index=pairs.other.values)
    s1_ids, q_ids = s1.entity_id.values, q.entity_id.values
    n_pos_total = int(pairs.s1.isin(s1_sel).sum())

    # ---- blocking + cheap features
    cand, F1, _ = block_and_cheap(s1, q, cfg)
    y = (q_ids[cand.qi.values] != "") & (pd.Series(q_ids[cand.qi.values]).map(truth).values == s1_ids[cand.si.values])
    y = y.astype(np.int8)
    grp = s1_ids[cand.si.values]
    is_val = _is_val(grp, cfg["val_pct"])
    log(f"blocked pairs {len(cand)}  positives {y.sum()} / {n_pos_total} (recall {y.sum() / n_pos_total:.5f})")

    # ---- stage 1: OOF predictions on training rows, plain predictions on validation rows
    feats1 = list(F1.columns)
    X1 = F1[feats1].values.astype(np.float32)
    trn = ~is_val
    fold = np.array([zlib.crc32(g.encode()) % 2 for g in grp])
    p1 = np.zeros(len(cand), np.float32)
    s1_params = dict(num_leaves=63, learning_rate=0.1)
    for k in (0, 1):
        m = trn & (fold != k)
        b = M.train_lgb(X1[m], y[m], rounds=300, params=s1_params, log_every=0)
        p1[trn & (fold == k)] = b.predict(X1[trn & (fold == k)])
    bst1 = M.train_lgb(X1[trn], y[trn], rounds=300, params=s1_params, log_every=0)
    p1[is_val] = bst1.predict(X1[is_val])
    # threshold keeping the target share of true pairs present after blocking
    pos_p = np.sort(p1[trn & (y == 1)])
    thr1 = float(pos_p[int((1 - cfg["s1_recall"]) * len(pos_p))])
    keep = p1 >= thr1
    log(f"stage-1 threshold {thr1:.5f}: keeps {keep.mean():.3f} of pairs "
        f"({keep.sum() / len(q):.2f}/query), recall kept {y[keep].sum() / max(1, y.sum()):.5f}")
    cand, F1, p1, y, grp, is_val = (cand[keep].reset_index(drop=True), F1[keep].reset_index(drop=True),
                                    p1[keep], y[keep], grp[keep], is_val[keep])
    del X1
    gc.collect()

    # ---- stage 2 features
    F, extra, miss = stage2_features(cand, F1, p1, s1, q, cfg)
    del F1
    gc.collect()
    log("stage-2 features", F.shape)
    hard = F.n_tset.values >= 80
    trn = ~is_val
    Ftr = F[trn].copy()
    M.oof_lexicon_features(Ftr, extra[trn], miss[trn], y[trn], grp[trn], hard[trn], drop_neg=cfg["lex_drop"])
    lex = M.learn_lexicon(extra[trn], miss[trn], y[trn], hard[trn])
    Fv = F[is_val].copy()
    M.add_lex(Fv, extra[is_val], miss[is_val], lex)
    feats2 = list(Ftr.columns)
    bst2 = M.train_lgb(Ftr[feats2].values.astype(np.float32), y[trn],
                       Fv[feats2].values.astype(np.float32), y[is_val], rounds=cfg["rounds"])
    best_it = bst2.best_iteration or cfg["rounds"]
    imp = sorted(zip(bst2.feature_importance("gain"), feats2), reverse=True)
    log("best iteration", best_it, "| top features", [(n, int(g)) for g, n in imp[:20]])

    # ---- threshold tuning on validation S1 entities (all queries compete)
    p2 = np.empty(len(F), np.float32)
    p2[trn] = bst2.predict(Ftr[feats2].values.astype(np.float32), num_iteration=best_it)
    p2[is_val] = bst2.predict(Fv[feats2].values.astype(np.float32), num_iteration=best_it)
    val_ents = [s for s in s1_sel if zlib.crc32(s.encode()) % 100 < cfg["val_pct"]]
    truth_map = M.to_mapping(pairs.s1.values, pairs.other.values)
    best = (0.0, 0.5)
    for t in np.arange(0.3, 0.951, 0.025):
        aq, as_ = M.assign(cand.qi.values, cand.si.values, p2, t)
        sc = M.fbeta_macro(M.to_mapping(s1_ids[as_], q_ids[aq]), truth_map, val_ents)
        log(f"  thr {t:.3f}  F0.5 {sc:.5f}")
        if sc > best[0]:
            best = (sc, float(t))
    log("best validation F0.5 %.5f at threshold %.3f (on %d entities)" % (best[0], best[1], len(val_ents)))
    # predicted / true matches on validation at the chosen threshold -> used to calibrate test thresholds
    aq, as_ = M.assign(cand.qi.values, cand.si.values, p2, best[1])
    val_set = set(val_ents)
    n_pred = sum(1 for s in s1_ids[as_] if s in val_set)
    n_true = sum(len(truth_map.get(s, ())) for s in val_ents)
    target_mps = n_pred / max(1, len(val_ents))  # singletons included in the denominator
    log(f"validation: true matches/S1 {n_true / max(1, len(val_ents)):.3f}, predicted/S1 {target_mps:.3f}"
        f" (ratio {n_pred / max(1, n_true):.4f}) -> calibration target {target_mps:.3f}")
    if cfg["refit_full"]:
        Fall = F.copy()
        M.oof_lexicon_features(Fall, extra, miss, y, grp, hard, drop_neg=cfg["lex_drop"])
        lex = M.learn_lexicon(extra, miss, y, hard)
        bst2 = M.train_lgb(Fall[feats2].values.astype(np.float32), y, rounds=int(best_it * 1.15))
    bst1.save_model(os.path.join(a.work, "stage1.txt"))
    bst2.save_model(os.path.join(a.work, "stage2.txt"), num_iteration=None if cfg["refit_full"] else best_it)
    json.dump(lex, open(os.path.join(a.work, "lexicon.json"), "w"))
    json.dump({"threshold": best[1], "val_f05": best[0], "stage1_threshold": thr1, "target_mps": target_mps, "features1": feats1,
               "features2": feats2, "cfg": cfg}, open(os.path.join(a.work, "meta.json"), "w"), indent=1)
    log("saved artifacts to", a.work)


# ----------------------------------------------------------------------------
# PREDICT
# ----------------------------------------------------------------------------
def cmd_predict(a):
    import lightgbm as lgb
    meta = json.load(open(os.path.join(a.work, "meta.json")))
    cfg = {**DEFAULT_CFG, **meta["cfg"]}
    if a.threads:
        cfg["threads"] = a.threads
    if a.workers:
        cfg["workers"] = a.workers
    thr = a.threshold if a.threshold is not None else meta["threshold"]
    meta.setdefault("target_mps", 3.375)  # older artifacts: training prior 3.46 x ~0.975 recall ratio
    tr = json.load(open(os.path.join(a.work, "translit.json"), encoding="utf-8"))
    lex = json.load(open(os.path.join(a.work, "lexicon.json")))
    bst1 = lgb.Booster(model_file=os.path.join(a.work, "stage1.txt"))
    bst2 = lgb.Booster(model_file=os.path.join(a.work, "stage2.txt"))
    stage1 = (bst1, meta["features1"], meta["stage1_threshold"])
    feats2 = meta["features2"]
    src = read_sources(a.data_dir, "test")
    log("loaded test", len(src), src.country.value_counts().to_dict())
    s1_order = src.entity_id.values[src.src.values == 1].copy()
    res = []
    for country in sorted(src.country.unique()):
        part = src[src.country == country]
        tr_c = extend_translit(tr, part.business_name.values)
        log(country, f"translit: {len(tr)} learned + {len(tr_c) - len(tr)} phonetic fallbacks")
        part = slim(parse_frame(part, tr_c, workers=cfg["workers"]))
        s1 = part[part.src == 1].reset_index(drop=True)
        q = part[part.src != 1].reset_index(drop=True)
        del part
        gc.collect()
        log(country, "S1", len(s1), "queries", len(q))
        cand, F1, p1 = block_and_cheap(s1, q, cfg, stage1=stage1)
        log(country, "candidate pairs after stage 1:", len(cand))
        p2 = np.zeros(len(cand), np.float32)
        anc = group_features(cand, p1, F1, s1, q, cfg)
        vocabs = country_vocabs(s1, q)
        step = 3_000_000
        for st in range(0, len(cand), step):
            sl = slice(st, st + step)
            F, extra, miss = stage2_features(cand.iloc[sl].reset_index(drop=True),
                                             F1.iloc[sl].reset_index(drop=True), p1[sl], s1, q, cfg,
                                             anc=anc.iloc[sl], vocabs=vocabs)
            M.add_lex(F, extra, miss, lex)
            p2[sl] = bst2.predict(F[feats2].values.astype(np.float32), num_threads=cfg["threads"])
            del F, extra, miss
            gc.collect()
        res.append(pd.DataFrame({"s1": s1.entity_id.values[cand.si.values],
                                 "q": q.entity_id.values[cand.qi.values], "p": p2, "country": country}))
        del s1, q, cand, F1, p1
        gc.collect()
    d = pd.concat(res, ignore_index=True).sort_values("p", ascending=False)
    os.makedirs(a.out, exist_ok=True)
    cand_map = d.groupby("s1", sort=False).q.apply(list).to_dict()
    write_id_lists(os.path.join(a.out, "candidate_pairs.tsv"), s1_order, cand_map, "candidate_entity_ids")
    best = d.drop_duplicates("q")
    n_s1 = src[src.src == 1].country.value_counts().to_dict()
    thr_c = calibrate_thresholds(best, n_s1, thr, meta.get("target_mps"), a.calibrate)
    best = best[best.p.values >= best.country.map(thr_c).values]
    match = best.groupby("s1", sort=False).q.apply(list).to_dict()
    write_id_lists(os.path.join(a.out, "matching_results.tsv"), s1_order, match, "matched_entity_ids")
    if not a.no_scores:
        d.to_parquet(os.path.join(a.out, "pair_scores.parquet"), index=False)
    log(f"wrote outputs: {len(d)} candidate pairs, {len(best)} matches, "
        f"{sum(1 for s in s1_order if s not in match)} S1 without match, thresholds {thr_c}")


def calibrate_thresholds(best, n_s1, base_thr, target_mps, mode, tol=0.02):
    """Per-country decision thresholds.

    Training truth has the same cluster-size distribution in every country (3.46 matches per
    S1), and the test set carries ~1.7x more distractors per S1 than training, which inflates
    probabilities. With mode='shape' a country whose predicted matches per S1 exceed the
    calibration target (learned on validation) by more than `tol` gets its threshold raised
    (never lowered) until it meets the target.
    """
    thr_c = {c: float(base_thr) for c in n_s1}
    if mode != "shape" or not target_mps:
        return thr_c
    for c, n in n_s1.items():
        p = np.sort(best.p.values[best.country.values == c])[::-1]
        k = int(round(target_mps * n))
        above = int((p >= base_thr).sum())
        if 0 < k and above > k * (1 + tol):  # only act on clear over-matching
            thr_c[c] = float(max(base_thr, p[k - 1]))
        log(f"  calibrate {c}: {above / n:.3f} matches/S1 at base thr {base_thr:.3f} -> "
            f"thr {thr_c[c]:.4f} ({min(above, k) / n:.3f} matches/S1, target {target_mps:.3f})")
    return thr_c


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data-dir", default=None)
    t.add_argument("--dev", default=None, help="folder with a dev subset (make_dev_subset.py)")
    t.add_argument("--work", required=True)
    t.add_argument("--config", default=None)
    t.add_argument("--threads", type=int, default=None)
    t.add_argument("--workers", type=int, default=None)
    p = sub.add_parser("predict")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--no-scores", action="store_true", help="do not write pair_scores.parquet")
    p.add_argument("--calibrate", choices=["shape", "none"], default="shape",
                   help="per-country threshold calibration to the training cluster-size prior")
    a = ap.parse_args()
    {"train": cmd_train, "predict": cmd_predict}[a.cmd](a)


if __name__ == "__main__":
    main()
