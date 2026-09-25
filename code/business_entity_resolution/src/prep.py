"""Record-level parsing: turns raw rows into normalised columns used by blocking/features."""
import collections
import os
import re
from multiprocessing import Pool

import numpy as np
import pandas as pd

from rapidfuzz.distance import Levenshtein
from unidecode import unidecode

from .normalize import parse_address, parse_name, is_native, fold

_TRANSLIT = None


def _init(translit):
    global _TRANSLIT
    _TRANSLIT = translit


def _parse_chunk(rows):
    out = []
    for name, addr, country in rows:
        n = parse_name(name, _TRANSLIT)
        a = parse_address(addr, country)
        out.append((
            " ".join(n["core"]), " ".join(n["legal"]), " ".join(n["alt"]), n["native"], n["domain"],
            n["concat"], a["state"], a["hn"], a["hn_suf"], a["half"], " ".join(a["nums"]),
            " ".join(a["toks"]), " ".join(a["street"]), a["city"], a["empty"],
        ))
    return out


PARSED_COLS = ["name_core", "legal", "name_alt", "native", "domain", "concat", "state", "hn",
               "hn_suf", "half", "nums", "addr_toks", "street", "city", "addr_empty"]


def parse_frame(df, translit=None, workers=None, chunk=20000):
    """Add parsed columns to a frame with business_name/business_address/country."""
    rows = list(zip(df.business_name.values, df.business_address.values, df.country.values))
    chunks = [rows[i:i + chunk] for i in range(0, len(rows), chunk)]
    workers = workers or max(1, (os.cpu_count() or 2))
    if workers > 1 and len(chunks) > 1:
        with Pool(workers, initializer=_init, initargs=(translit,)) as pool:
            res = pool.map(_parse_chunk, chunks)
    else:
        _init(translit)
        res = [_parse_chunk(c) for c in chunks]
    flat = [r for part in res for r in part]
    parsed = pd.DataFrame(flat, columns=PARSED_COLS, index=df.index)
    for c in ("native", "domain", "half", "addr_empty"):
        parsed[c] = parsed[c].astype(np.int8)
    return pd.concat([df, parsed], axis=1)


_SKEL_SUBS = (("ph", "f"), ("bh", "b"), ("dh", "d"), ("th", "t"), ("kh", "k"), ("gh", "g"), ("sh", "s"),
              ("ch", "c"), ("ck", "k"), ("q", "k"), ("w", "v"), ("x", "ks"), ("z", "j"))


def skeleton(s):
    """Consonant skeleton used to match a romanised native word to a Latin word by sound."""
    s = s.lower()
    for a, b in _SKEL_SUBS:
        s = s.replace(a, b)
    s = re.sub(r"c(?=[eiy])", "s", s)
    s = s.replace("c", "k")
    s = re.sub(r"[^a-z]", "", s)
    s = re.sub(r"[aeiouy]", "", s)
    return re.sub(r"(.)\1+", r"\1", s)


def extend_translit(table, names, min_count=20):
    """Add a phonetic fallback for native-script tokens the learned table does not know.

    Unknown native tokens (e.g. test-only business-type words such as "ज्वेलर्स") are romanised
    and matched by consonant skeleton to the most frequent Latin name token of the same data
    (exact skeleton, or edit distance 1 for skeletons of length >= 5). Uses only the names given.
    """
    latin = collections.Counter()
    unknown = set()
    for n in names:
        if not isinstance(n, str) or not n:
            continue
        if is_native(n):
            for w in n.split():
                if is_native(w) and w not in table:
                    unknown.add(w)
        else:
            latin.update(re.findall(r"[a-z]+", fold(n)))
    if not unknown:
        return dict(table)
    by_skel = {}
    for w, c in latin.items():
        if c < min_count or len(w) < 3:
            continue
        k = skeleton(w)
        if len(k) >= 2 and (k not in by_skel or by_skel[k][0] < c):
            by_skel[k] = (c, w)
    by_len = collections.defaultdict(list)
    for k, v in by_skel.items():
        by_len[len(k)].append((k, v))
    out = dict(table)
    for w in unknown:
        sk = skeleton(unidecode(w))
        if len(sk) < 3:
            continue
        hit = by_skel.get(sk)
        if hit is None and len(sk) >= 5:
            best = None
            for L in (len(sk) - 1, len(sk), len(sk) + 1):
                for k, v in by_len.get(L, ()):
                    if k[0] == sk[0] and Levenshtein.distance(k, sk) <= 1 and (best is None or v[0] > best[0]):
                        best = v
            hit = best
        if hit is not None:
            out[w] = hit[1]
    return out


def learn_translit(s1_names, other_names, min_count=2):
    """Learn native-script token -> latin token mapping from matched name pairs.

    Pairs whose token counts agree are aligned positionally.
    """
    cnt = collections.defaultdict(collections.Counter)
    for a, b in zip(s1_names, other_names):
        if not isinstance(b, str) or not is_native(b) or is_native(a):
            continue
        nt = b.split()
        lt = [t for t in fold(a).replace("&", " and ").split() if t]
        lt = ["".join(ch for ch in t if ch.isalnum()) for t in lt]
        lt = [t for t in lt if t]
        if len(nt) != len(lt):
            continue
        for x, y in zip(nt, lt):
            cnt[x][y] += 1
    table = {}
    for x, c in cnt.items():
        y, n = c.most_common(1)[0]
        if n >= min_count and n >= 0.5 * sum(c.values()):
            table[x] = y
    return table
