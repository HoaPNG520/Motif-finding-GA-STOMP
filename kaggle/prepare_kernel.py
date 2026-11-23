#!/usr/bin/env python3
"""
prepare_kernel.py -- stage kaggle_run.ipynb for `kaggle kernels push`.

    python kaggle/prepare_kernel.py --ref <git sha or branch> --user <kaggle username> \
        --out build/kernel

Writes into --out:
    kaggle_run.ipynb       copy of the repo notebook with GIT_REF pinned to --ref
                           and all outputs cleared
    kernel-metadata.json   private GPU notebook with the CWRU dataset attached

No credentials are read or written here; the Kaggle CLI picks up
KAGGLE_API_TOKEN from the environment at push time.
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

DEFAULT_SLUG = "ga-stomp-cwru"
DEFAULT_TITLE = "GA-STOMP CWRU"          # Kaggle derives the slug from the title
DATASETS = ["sufian79/cwru-mat-full-dataset"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="git ref the notebook should check out")
    ap.add_argument("--user", required=True, help="Kaggle username (kernel owner)")
    ap.add_argument("--out", required=True, help="staging folder")
    ap.add_argument("--slug", default=DEFAULT_SLUG)
    ap.add_argument("--title", default=DEFAULT_TITLE)
    ap.add_argument("--notebook", default=os.path.join(REPO, "kaggle_run.ipynb"))
    args = ap.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9._/\-]+", args.ref):
        sys.exit(f"refusing suspicious ref: {args.ref!r}")
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", args.user):
        sys.exit(f"refusing suspicious username: {args.user!r}")

    with open(args.notebook) as f:
        nb = json.load(f)

    pinned = False
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        cell["outputs"] = []
        cell["execution_count"] = None
        src = []
        for line in cell["source"]:
            if line.startswith("GIT_REF") and "=" in line:
                line = f'GIT_REF  = "{args.ref}"   # pinned by prepare_kernel.py\n'
                pinned = True
            src.append(line)
        cell["source"] = src
    if not pinned:
        sys.exit("no 'GIT_REF = ...' line found in the notebook")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "kaggle_run.ipynb"), "w") as f:
        json.dump(nb, f, indent=1)

    meta = {
        "id": f"{args.user}/{args.slug}",
        "title": args.title,
        "code_file": "kaggle_run.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,     # needed to git clone the repo
        "dataset_sources": DATASETS,
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    with open(os.path.join(args.out, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"staged {meta['id']} @ {args.ref} -> {args.out}")


if __name__ == "__main__":
    main()
