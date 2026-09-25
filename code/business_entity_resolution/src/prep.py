"""Record-level parsing: turns raw rows into normalised columns used by blocking/features."""
import collections
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

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
