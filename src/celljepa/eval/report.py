"""Minimal report writer for M0."""

from __future__ import annotations

from pathlib import Path
from typing import Dict


def write_report(path: str | Path, summary: Dict[str, object], metrics: Dict[str, object]) -> None:
    """Write a simple markdown report with dataset summary + metrics."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# CellJEPA Report (M0 stub)", ""]
    lines.append("## Summary")
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Metrics")
    for key, value in metrics.items():
        lines.append(f"- {key}: {value}")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")

