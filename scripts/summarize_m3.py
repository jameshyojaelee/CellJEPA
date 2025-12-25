#!/usr/bin/env python3
"""Aggregate M3 metrics into a single markdown report."""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Iterable


def iter_metric_paths() -> Iterable[Path]:
    patterns = [
        "runs/m3_*/metrics.json",
        "runs/m3_*/*/metrics.json",
    ]
    seen = set()
    for pattern in patterns:
        for path in glob.glob(pattern):
            p = Path(path)
            if p in seen:
                continue
            seen.add(p)
            yield p


def parse_name(tag: str) -> tuple[str, str]:
    tokens = [t for t in tag.split("_") if t]
    if tokens and tokens[0].lower() == "m3":
        tokens = tokens[1:]

    dataset_tokens = []
    for token in tokens:
        lower = token.lower()
        if lower in {"s1", "s2", "proto", "prototype", "set"}:
            break
        dataset_tokens.append(token)

    dataset = "_".join(dataset_tokens) if dataset_tokens else "unknown"
    split = "unknown"
    for token in tokens:
        lower = token.lower()
        if lower == "s1":
            split = "S1"
        elif lower == "s2":
            split = "S2"
    return dataset, split


def parse_path(p: Path) -> dict:
    parts = p.parts
    run_id = "unknown"
    if len(parts) >= 2 and parts[0] == "runs":
        run_id = parts[1]

    if len(parts) >= 4 and parts[-1] == "metrics.json":
        tag = parts[-2]
    else:
        tag = run_id

    dataset, split = parse_name(tag)
    return {
        "run_id": run_id,
        "dataset": dataset,
        "split": split,
    }


def load_metrics() -> list[dict]:
    rows = []
    for p in iter_metric_paths():
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        meta = parse_path(p)
        row = {
            **meta,
            "path": str(p),
            "mode": data.get("mode"),
            "n_train": data.get("n_train"),
            "n_test": data.get("n_test"),
            "loss": data.get("loss"),
        }
        test = data.get("test", {})
        row.update({
            "test_mse": test.get("mse_mean"),
            "test_cos": test.get("cosine_mean"),
            "test_edist": test.get("edist_mean"),
            "skipped_pairs": test.get("skipped_pairs"),
            "n_eval": test.get("n_eval"),
        })
        rows.append(row)
    return rows


def render_table(rows: list[dict]) -> str:
    header = [
        "run_id", "dataset", "split", "mode", "n_train", "n_test", "loss",
        "test_mse", "test_cos", "test_edist", "skipped_pairs", "n_eval",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]

    for r in rows:
        line = [
            str(r.get("run_id", "")),
            str(r.get("dataset", "")),
            str(r.get("split", "")),
            str(r.get("mode", "")),
            str(r.get("n_train", "")),
            str(r.get("n_test", "")),
            f"{r.get('loss', float('nan')):.4f}" if isinstance(r.get("loss"), (int, float)) else "",
            f"{r.get('test_mse', float('nan')):.4f}" if isinstance(r.get("test_mse"), (int, float)) else "",
            f"{r.get('test_cos', float('nan')):.4f}" if isinstance(r.get("test_cos"), (int, float)) else "",
            f"{r.get('test_edist', float('nan')):.4f}" if isinstance(r.get("test_edist"), (int, float)) else "",
            str(r.get("skipped_pairs", "")) if r.get("skipped_pairs") is not None else "",
            str(r.get("n_eval", "")) if r.get("n_eval") is not None else "",
        ]
        lines.append("| " + " | ".join(line) + " |")

    return "\n".join(lines)


def main() -> None:
    rows = load_metrics()
    rows = sorted(rows, key=lambda r: (r.get("run_id", ""), r.get("dataset", ""), r.get("split", ""), r.get("mode", ""), r["path"]))
    out = Path("reports/m3_summary.md")
    out.parent.mkdir(parents=True, exist_ok=True)

    content = [
        "# M3 Summary (Auto-generated)",
        "",
        "This table aggregates all `runs/m3_*/*/metrics.json` and `runs/m3_*/metrics.json` files.",
        "",
        render_table(rows),
        "",
        "Notes:",
        "- `loss` is the training loss.",
        "- `test_mse`/`test_cos` are prototype test metrics.",
        "- `test_edist` is the set model test E-distance.",
        "- `skipped_pairs`/`n_eval` indicate NaN filtering behavior.",
    ]
    out.write_text("\n".join(content), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
