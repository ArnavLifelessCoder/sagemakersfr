"""Build <team>_submission.zip in the required layout from this repository.

    python build_final_package.py --team TEAM_NAME [--outputs submissions/ens3_blend_raw]
"""
import argparse, os, shutil, zipfile
ROOT = os.path.dirname(os.path.abspath(__file__))

def copytree(src, dst):
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.log"), dirs_exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--outputs", default=os.path.join(ROOT, "submissions", "ens3_blend_raw"))
    a = ap.parse_args()
    pkg = os.path.join(ROOT, "pkg")
    shutil.rmtree(pkg, ignore_errors=True)
    code = os.path.join(pkg, "code", "business_entity_resolution")
    os.makedirs(os.path.join(pkg, "output"), exist_ok=True)
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        shutil.copy(os.path.join(a.outputs, f), os.path.join(pkg, "output", f))
    copytree(os.path.join(ROOT, "v5", "business_entity_resolution", "src"), os.path.join(code, "src"))
    copytree(os.path.join(ROOT, "v6", "business_entity_resolution", "src"), os.path.join(code, "src", "v6_variant"))
    copytree(os.path.join(ROOT, "code", "business_entity_resolution", "src"), os.path.join(code, "src", "partner_line"))
    for f in ("README.md", "requirements.txt", "kaggle_run.ipynb"):
        p = os.path.join(ROOT, "code", "business_entity_resolution", f)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(code, "src", "partner_line", f))
    copytree(os.path.join(ROOT, "solution", "artifacts"), os.path.join(code, "artifacts"))
    shutil.copy(os.path.join(ROOT, "pkg_README.md"), os.path.join(code, "README.md"))
    shutil.copy(os.path.join(ROOT, "pkg_requirements.txt"), os.path.join(code, "requirements.txt"))
    shutil.copy(os.path.join(ROOT, "pkg_requirements_gpu.txt"), os.path.join(code, "requirements-gpu.txt"))
    shutil.copy(os.path.join(ROOT, "APPROACH.md"), os.path.join(code, "APPROACH.md"))
    shutil.copy(os.path.join(ROOT, "v5", "Documentation_template.md"), os.path.join(pkg, "Documentation_template.md"))
    # the cross-encoder script must also be runnable as a plain file (README uses `python src/cross_encoder.py`)
    zpath = os.path.join(ROOT, "submissions", f"{a.team}_submission.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(pkg):
            for f in files:
                p = os.path.join(root, f)
                z.write(p, os.path.relpath(p, pkg).replace(os.sep, "/"))
    print("wrote", zpath, round(os.path.getsize(zpath) / 1e6, 1), "MB")
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        print(len(names), "files; top level:", sorted({n.split('/')[0] for n in names}))
        for n in names:
            if n.count("/") <= 2 and not n.startswith("code/business_entity_resolution/src/"):
                print("  ", n)

if __name__ == "__main__":
    main()
