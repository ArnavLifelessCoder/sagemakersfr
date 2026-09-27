"""Synthetic test-style sibling distractors for training.

The test set contains many more sibling businesses than training, of a type training barely has (measured
by cell densities, FINDINGS 5.15): same name + legal form added/changed + house number within +-20
("Physical Therapy Care Ltd, 6245" next to S1 "Physical Therapy Care, 6240"), and in France same address
with one name word swapped ("Lille Sportive SAS" next to "Lille Maison SAS"). A model trained on training
odds therefore accepts them. Here such records are generated from REAL true S2/S3 records of training S1s
(so they keep the source's formatting and noise) and added to the training data as unmatched records:

  type A  legal form changed + house number shifted by 1..20
  type B  one core name word replaced by another common name word of the same country (address kept)
"""
import collections
import re

import numpy as np
import pandas as pd

from .normalize import LEGAL_CANON, fold

LEGAL_WORDS = {
    "US": ["LLC", "Inc", "Corp", "Co", "Ltd", "LP", "Inc.", "L.L.C.", "Corporation", "Company"],
    "India": ["Pvt Ltd", "Private Limited", "Limited", "LLP", "Ltd", "Pvt. Ltd."],
    "France": ["SAS", "SARL", "SASU", "EURL", "SCI", "SA", "SNC"],
}
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'&.\-]*")


def _strip_legal(name):
    toks = name.split()
    keep = [t for t in toks if fold(t).strip(".,()[]").replace(".", "") not in LEGAL_CANON]
    return " ".join(keep) if keep else name, len(keep) != len(toks)


def _shift_number(addr, rng):
    m = re.search(r"(?<![\w/-])#*0*(\d{1,6})(?![\d/])", addr)
    if not m:
        return None
    n = int(m.group(1))
    d = int(rng.integers(1, 21)) * (1 if rng.random() < 0.6 or n <= 20 else -1)
    new = max(1, n + d)
    if new == n:
        new = n + 1
    return addr[:m.start(1)] + str(new) + addr[m.end(1):]


def _vocab(names, top=3000):
    cnt = collections.Counter()
    for n in names:
        for t in WORD_RE.findall(n):
            tl = fold(t).strip(".")
            if len(tl) >= 4 and tl not in LEGAL_CANON and tl.isalpha():
                cnt[t.title()] += 1
    return [w for w, _ in cnt.most_common(top)]


def make_siblings(src, pairs, rate_a=0.15, rate_b=0.10, seed=0, log=print):
    """src: raw records (entity_id, business_name, business_address, country, src); pairs: true (s1, other)."""
    rng = np.random.default_rng(seed)
    rec = src.set_index("entity_id")
    s1_ids = set(src.entity_id.values[src.src.values == 1])
    true_of = collections.defaultdict(list)
    present = set(src.entity_id.values)
    for s, o in zip(pairs.s1.values, pairs.other.values):
        if s in s1_ids and o in present:
            true_of[s].append(o)
    vocab = {c: _vocab(src.business_name.values[(src.country.values == c) & (src.src.values == 1)])
             for c in src.country.unique()}
    rows = []
    for s, lst in true_of.items():
        if not lst:
            continue
        for kind, rate in (("A", rate_a), ("B", rate_b)):
            if rng.random() >= rate:
                continue
            base = lst[int(rng.integers(len(lst)))]
            name, addr, ctry, srcid = rec.at[base, "business_name"], rec.at[base, "business_address"], \
                rec.at[base, "country"], rec.at[base, "src"]
            if kind == "A":
                if not addr:
                    continue
                new_addr = _shift_number(addr, rng)
                if new_addr is None:
                    continue
                core, had = _strip_legal(name)
                words = LEGAL_WORDS.get(ctry, LEGAL_WORDS["US"])
                new_name = f"{core} {words[int(rng.integers(len(words)))]}"
                if new_name.strip().lower() == name.strip().lower():
                    new_name = f"{core} {words[(int(rng.integers(len(words))) + 1) % len(words)]}"
                rows.append((new_name, new_addr, ctry, srcid))
            else:
                toks = name.split()
                idx = [i for i, t in enumerate(toks)
                       if len(t) >= 4 and fold(t).strip(".,()[]") not in LEGAL_CANON and t[:1].isalpha()]
                vc = vocab.get(ctry) or []
                if not idx or not vc:
                    continue
                i = idx[int(rng.integers(len(idx)))]
                w = vc[int(rng.integers(len(vc)))]
                if w.lower() == toks[i].lower():
                    continue
                toks[i] = w.upper() if toks[i].isupper() else w
                rows.append((" ".join(toks), addr, ctry, srcid))
    out = pd.DataFrame(rows, columns=["business_name", "business_address", "country", "src"])
    out.insert(0, "entity_id", [f"S{r}-SYN{i}" for i, r in enumerate(out.src.values)])
    log(f"synthetic sibling distractors: {len(out)} "
        f"({(len(out) / max(1, len(s1_ids))):.3f} per S1) by country {out.country.value_counts().to_dict()}")
    return out
