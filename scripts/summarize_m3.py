#!/usr/bin/env python3
"""Aggregate M3 metrics into a single markdown report."""

from __future__ import annotations

import glob
import json
from pathlib import Path


def load_metrics() -> list[dict]:
    rows = []
    for path in glob.glob("runs/m3_*/*/metrics.json"):
        p = Path(path)
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        row = {
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
        "dataset", "split", "mode", "n_train", "n_test", "loss",
        "test_mse", "test_cos", "test_edist", "skipped_pairs", "n_eval",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]

    def parse_dataset(p: str) -> tuple[str, str]:
        # runs/m3_full_v3/<dataset_split_mode>/metrics.json
        parts = Path(p).parts
        try:
            run = parts[2]  # e.g., m3_full_v3
            name = parts[3]  # e.g., sciplex3_s1_proto
        except Exception:
            return "unknown", "unknown"
        # best-effort split extraction
        split = "unknown"
        if "_s1_" in name:
            split = "S1"
        elif "_s2_" in name:
            split = "S2"
        return name, split

    for r in rows:
        dataset, split = parse_dataset(r["path"])
        line = [
            dataset,
            split,
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
    rows = sorted(rows, key=lambda r: r["path"])
    out = Path("reports/m3_summary.md")
    out.parent.mkdir(parents=True, exist_ok=True)

    content = [
        "# M3 Summary (Auto-generated)",
        "",
        "This table aggregates all `runs/m3_*/*/metrics.json` files.",
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

