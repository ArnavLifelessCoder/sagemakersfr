"""Loading source files / ground truth and writing submission files."""
import csv
import io
import os
import zipfile

import pandas as pd

COLS = ["entity_id", "business_name", "business_address", "country"]


def _open(path_or_zip, member=None):
    if member is None:
        return open(path_or_zip, encoding="utf-8")
    z = zipfile.ZipFile(path_or_zip)
    name = [n for n in z.namelist() if n.endswith(member) and "__MACOSX" not in n][0]
    return io.TextIOWrapper(z.open(name), encoding="utf-8")


def read_tsv(path, zip_path=None):
    """Read a challenge TSV. `path` is a file path, or a member suffix when zip_path is set."""
    f = _open(zip_path, path) if zip_path else _open(path)
    with f:
        df = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False, na_filter=False,
                         quoting=csv.QUOTE_NONE, engine="c")
    return df


def read_sources(data_dir, split, zip_path=None):
    """Return concatenated dataframe of source1/2/3 with a `src` column (1/2/3)."""
    parts = []
    for k in (1, 2, 3):
        rel = f"{split}_source{k}.tsv"
        df = read_tsv(rel if zip_path else os.path.join(data_dir, split, rel), zip_path)
        df = df[COLS]
        df["src"] = k
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def read_ground_truth(path, zip_path=None):
    df = read_tsv(path, zip_path)
    return df


def gt_pairs(gt):
    """Explode ground truth into (s1, other) pairs."""
    g = gt[gt["matched_entity_ids"] != ""].copy()
    g["other"] = g["matched_entity_ids"].str.split(",")
    g = g.explode("other")[["source1_entity_id", "other"]]
    return g.rename(columns={"source1_entity_id": "s1"}).reset_index(drop=True)


def write_id_lists(path, s1_ids, mapping, header):
    """Write a submission-style TSV. mapping: s1 -> list of ids (order kept, deduped)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{header}\n")
        for s1 in s1_ids:
            ids = mapping.get(s1, ())
            seen = dict.fromkeys(i for i in ids if i and i[:3] in ("S2-", "S3-"))
            f.write(f"{s1}\t{','.join(seen)}\n")
