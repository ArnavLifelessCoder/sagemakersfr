"""Build the final submission zip in the layout the challenge requires.

    python v5/make_package.py --out-dir RUN_OUT_DIR --team TEAM --zip TEAM_submission.zip

RUN_OUT_DIR must hold matching_results.tsv and candidate_pairs.tsv (validated). The code folder is
v5/business_entity_resolution (src/, README.md, requirements.txt); the methodology write-up is
v5/Documentation_template.md.
"""
import argparse
import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--code", default=os.path.join(HERE, "business_entity_resolution"))
    ap.add_argument("--doc", default=os.path.join(HERE, "Documentation_template.md"))
    a = ap.parse_args()
    code = a.code
    doc = a.doc
    with zipfile.ZipFile(a.zip, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ("matching_results.tsv", "candidate_pairs.tsv"):
            z.write(os.path.join(a.out_dir, f), f"output/{f}")
        for root, dirs, files in os.walk(code):
            dirs[:] = [d for d in dirs if d not in ("__pycache__",)]
            for f in files:
                if f.endswith((".pyc",)):
                    continue
                p = os.path.join(root, f)
                z.write(p, "code/business_entity_resolution/" + os.path.relpath(p, code).replace(os.sep, "/"))
        z.write(doc, "Documentation_template.md")
    print("wrote", a.zip, os.path.getsize(a.zip) / 1e6, "MB")
    with zipfile.ZipFile(a.zip) as z:
        for i in z.infolist():
            print(f"  {i.file_size / 1e6:8.1f} MB  {i.filename}")


if __name__ == "__main__":
    main()
