#!/usr/bin/env python3
"""Prepare a download manifest for Tahoe-100M from Hugging Face."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


HF_API_URL = "https://huggingface.co/api/datasets/tahoebio/Tahoe-100M"
HF_BASE = "https://huggingface.co/datasets/tahoebio/Tahoe-100M/resolve/main"


def fetch_hf_dataset() -> dict:
    with urllib.request.urlopen(HF_API_URL) as resp:
        return json.load(resp)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Tahoe-100M download manifest (HF).")
    parser.add_argument(
        "--out",
        default="configs/download/tahoe100m_urls.tsv",
        help="Output manifest path (TSV: url, relpath).",
    )
    parser.add_argument(
        "--prefix",
        default="tahoe-100m",
        help="Relative path prefix under data/raw/.",
    )
    parser.add_argument(
        "--include",
        default="",
        help="Optional substring filter to include only matching paths (empty = all).",
    )
    args = parser.parse_args()

    data = fetch_hf_dataset()
    siblings = data.get("siblings", [])
    if not siblings:
        raise SystemExit("No file list (siblings) found in HF API response.")

    include = args.include
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    with out_path.open("w", encoding="utf-8") as f:
        for item in siblings:
            rfilename = item.get("rfilename")
            if not rfilename:
                continue
            if include and include not in rfilename:
                continue
            url = f"{HF_BASE}/{rfilename}?download=1"
            relpath = f"{args.prefix}/{rfilename}"
            f.write(f"{url}\t{relpath}\n")
            kept += 1

    print(f"Wrote {kept} entries to {out_path}")


if __name__ == "__main__":
    main()

