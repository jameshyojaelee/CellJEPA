#!/usr/bin/env python3
"""Convert Single Cell Portal bulk curl config into a TSV manifest for Slurm array downloads."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path


def normalize_line(line: str) -> str:
    return line.strip().strip("\r")


def parse_cfg(cfg_path: Path):
    lines = cfg_path.read_text(encoding="utf-8").splitlines()
    entries = []
    current_out = None
    current_url = None

    for raw in lines:
        line = normalize_line(raw)
        if not line or line.startswith("#"):
            continue

        if line.startswith("--output") or line.startswith("-o "):
            parts = shlex.split(line)
            if len(parts) >= 2:
                current_out = parts[1]
            if current_out and current_url:
                entries.append((current_url, current_out))
                current_out = None
                current_url = None
            continue
        if line.startswith("output="):
            current_out = line.split("=", 1)[1].strip().strip('"')
            if current_out and current_url:
                entries.append((current_url, current_out))
                current_out = None
                current_url = None
            continue

        if line.startswith("--url"):
            parts = shlex.split(line)
            if len(parts) >= 2:
                current_url = parts[1]
        elif line.startswith("url="):
            current_url = line.split("=", 1)[1].strip().strip('"')
        elif line.startswith("http://") or line.startswith("https://"):
            current_url = line.strip().strip('"')

        if current_out and current_url:
            entries.append((current_url, current_out))
            current_out = None
            current_url = None

    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SCP bulk download manifest.")
    parser.add_argument("--cfg", required=True, help="Path to SCP curl config (cfg.txt).")
    parser.add_argument(
        "--out",
        default="configs/download/perturb_cite_seq_urls.tsv",
        help="Output TSV manifest (url, relpath).",
    )
    parser.add_argument(
        "--prefix",
        default="perturb_cite_seq",
        help="Relative path prefix under data/raw/.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.cfg)
    entries = parse_cfg(cfg_path)
    if not entries:
        raise SystemExit("No (url, output) pairs found in cfg file.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for url, out in entries:
            relpath = f"{args.prefix}/{out.lstrip('./')}"
            f.write(f"{url}\t{relpath}\n")

    print(f"Wrote {len(entries)} entries to {out_path}")


if __name__ == "__main__":
    main()
