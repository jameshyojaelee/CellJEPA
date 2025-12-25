#!/usr/bin/env python3
"""Prepare a download manifest for scPerturb from Zenodo."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


def fetch_record(record_id: str) -> dict:
    url = f"https://zenodo.org/api/records/{record_id}"
    with urllib.request.urlopen(url) as resp:
        return json.load(resp)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare scPerturb download manifest (Zenodo).")
    parser.add_argument(
        "--record-id",
        default="13350497",
        help="Zenodo record ID for scPerturb h5ad files (default: 13350497, v1.4).",
    )
    parser.add_argument(
        "--out",
        default="configs/download/scperturb_v1_4_urls.tsv",
        help="Output manifest path (TSV: url, relpath, checksum).",
    )
    parser.add_argument(
        "--prefix",
        default="scperturb",
        help="Relative path prefix under data/raw/.",
    )
    args = parser.parse_args()

    record = fetch_record(args.record_id)
    files = record.get("files", [])
    if not files:
        raise SystemExit("No files found in Zenodo record.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for entry in files:
            key = entry["key"]
            # Use direct download URL from Zenodo API.
            url = entry["links"].get("download") or entry["links"]["self"]
            checksum = entry.get("checksum", "")
            relpath = f"{args.prefix}/{key}"
            f.write(f"{url}\t{relpath}\t{checksum}\n")

    print(f"Wrote {len(files)} entries to {out_path}")


if __name__ == "__main__":
    main()
