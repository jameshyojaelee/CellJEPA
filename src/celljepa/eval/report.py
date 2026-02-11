"""Report generation for CellJEPA evaluation (P5).

Generates markdown comparison tables and full benchmark reports.
Extends the M0 stub report writer.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

import numpy as np

if TYPE_CHECKING:
    from celljepa.eval.benchmark_runner import BenchmarkResult


def write_report(path: str | Path, summary: Dict[str, object], metrics: Dict[str, object]) -> None:
    """Write a simple markdown report with dataset summary + metrics (M0)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# CellJEPA Report", ""]
    lines.append("## Summary")
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Metrics")
    for key, value in metrics.items():
        lines.append(f"- {key}: {value}")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def generate_comparison_table(results: "BenchmarkResult") -> str:
    """Generate a markdown comparison table from benchmark results.

    Rows = metrics, Columns = methods. Best value per metric is bolded.
    Includes 95% CIs where available.
    """
    methods = list(results.method_results.keys())
    if not methods:
        return "No results to display.\n"

    # Collect all metric names across methods
    all_metrics: List[str] = []
    for mr in results.method_results.values():
        for m in mr.metrics:
            if m not in all_metrics and m != "error":
                all_metrics.append(m)

    # Metrics where lower is better
    lower_is_better = {
        "cosine_distance", "mse", "energy_distance", "perturbench_rank",
        "calibrated_edist",
    }

    lines = [
        f"# Benchmark Comparison",
        f"",
        f"- Train samples: {results.n_train}",
        f"- Test samples: {results.n_test}",
        f"- Timestamp: {results.timestamp}",
        f"",
    ]

    # Header
    header = "| Metric |"
    sep = "|--------|"
    for method in methods:
        header += f" {method} |"
        sep += "--------|"
    lines.append(header)
    lines.append(sep)

    # Rows
    for metric in all_metrics:
        row = f"| {metric} |"
        values = {}
        for method in methods:
            mr = results.method_results[method]
            if metric in mr.metrics:
                val = mr.metrics[metric]
                if isinstance(val, (int, float, np.floating)):
                    values[method] = float(val)
                else:
                    values[method] = None
            else:
                values[method] = None

        # Find best
        valid = {m: v for m, v in values.items() if v is not None and not np.isnan(v)}
        if valid:
            if metric in lower_is_better:
                best_method = min(valid, key=valid.get)
            else:
                best_method = max(valid, key=valid.get)
        else:
            best_method = None

        for method in methods:
            mr = results.method_results[method]
            v = values.get(method)
            if v is None:
                row += " — |"
                continue

            # Format value
            cell = f"{v:.4f}"

            # Add CI if available
            if metric in mr.metrics_ci:
                lo, hi = mr.metrics_ci[metric]
                cell += f" ({lo:.4f}–{hi:.4f})"

            # Bold best
            if method == best_method:
                cell = f"**{cell}**"

            row += f" {cell} |"

        lines.append(row)

    # Timing
    lines.append("")
    lines.append("### Timing")
    lines.append("")
    header = "| Method | Fit (s) | Predict (s) |"
    sep = "|--------|---------|-------------|"
    lines.append(header)
    lines.append(sep)
    for method in methods:
        mr = results.method_results[method]
        lines.append(f"| {method} | {mr.fit_time_s:.2f} | {mr.predict_time_s:.2f} |")

    lines.append("")
    return "\n".join(lines)


def generate_benchmark_report(
    results: "BenchmarkResult",
    output_path: str | Path,
    title: str = "CellJEPA Benchmark Report",
) -> None:
    """Generate a full benchmark report with comparison tables.

    Args:
        results: BenchmarkResult from BenchmarkRunner.
        output_path: path to write the report markdown.
        title: report title.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sections = [
        f"# {title}",
        "",
        generate_comparison_table(results),
    ]

    # Per-method detail
    sections.append("## Method Details\n")
    for name, mr in results.method_results.items():
        sections.append(f"### {name}")
        sections.append("")
        if "error" in mr.metrics:
            sections.append(f"> **Error**: {mr.metrics['error']}")
        else:
            for metric, value in mr.metrics.items():
                ci_str = ""
                if metric in mr.metrics_ci:
                    lo, hi = mr.metrics_ci[metric]
                    ci_str = f" (95% CI: {lo:.4f}–{hi:.4f})"
                if isinstance(value, (int, float, np.floating)):
                    sections.append(f"- {metric}: {value:.4f}{ci_str}")
                else:
                    sections.append(f"- {metric}: {value}")
        sections.append("")

    path.write_text("\n".join(sections), encoding="utf-8")
