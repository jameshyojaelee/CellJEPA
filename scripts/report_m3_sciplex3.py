#!/usr/bin/env python3
"""Generate a focused M3 sciplex3 report with baselines and CIs."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Iterable


def iter_metric_paths(patterns: Iterable[str]) -> Iterable[Path]:
    seen = set()
    for pattern in patterns:
        for path in glob.glob(pattern):
            p = Path(path)
            if p in seen:
                continue
            seen.add(p)
            yield p


def split_label(split_name: str, tag: str) -> str:
    lowered = split_name.lower()
    if "s1" in lowered:
        return "S1"
    if "s2" in lowered:
        return "S2"
    if "_s1_" in tag:
        return "S1"
    if "_s2_" in tag:
        return "S2"
    return "unknown"


def format_ci(mean, ci):
    if mean is None:
        return ""
    try:
        if not isinstance(ci, (list, tuple)) or len(ci) != 2:
            return f"{mean:.4f}"
        return f"{mean:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"
    except Exception:
        return str(mean)


def extract_ablation(metrics: dict) -> str:
    jepa_cfg = metrics.get("jepa_config") or {}
    run_cfg = metrics.get("jepa_run_config") or {}
    train_cfg = run_cfg.get("train", {}) if isinstance(run_cfg, dict) else {}
    model_cfg = run_cfg.get("model", {}) if isinstance(run_cfg, dict) else {}

    ema = model_cfg.get("ema_decay", jepa_cfg.get("ema_decay"))
    mask_type = train_cfg.get("mask_type", "random")
    mask_ratio = model_cfg.get("mask_ratio", jepa_cfg.get("mask_ratio"))
    var_w = model_cfg.get("variance_weight", jepa_cfg.get("variance_weight"))
    cov_w = model_cfg.get("covariance_weight", jepa_cfg.get("covariance_weight"))

    return f"ema={ema}, mask={mask_type}:{mask_ratio}, reg=({var_w},{cov_w})"


def render_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize M3 sciplex3 metrics with baselines.")
    parser.add_argument("--out", default="reports/m3_sciplex3_report.md")
    parser.add_argument("--patterns", nargs="*", default=["runs/m3_*/metrics.json", "runs/m3_*/*/metrics.json"])
    args = parser.parse_args()

    rows_set = []
    rows_proto = []
    missing_baselines = 0

    for path in iter_metric_paths(args.patterns):
        try:
            metrics = json.loads(path.read_text())
        except Exception:
            continue
        dataset_id = metrics.get("dataset_id") or ""
        if "sciplex3" not in dataset_id.lower() and "sciplex3" not in str(path).lower():
            continue

        parts = path.parts
        run_id = parts[1] if len(parts) > 1 and parts[0] == "runs" else "unknown"
        tag = parts[-2] if len(parts) >= 2 else path.stem
        mode = metrics.get("mode", "")
        seed = metrics.get("seed", "")
        split_name = metrics.get("split_name", "")
        split = split_label(split_name, tag)
        ablation = extract_ablation(metrics)
        n_eval = metrics.get("test", {}).get("n_eval")
        skipped = metrics.get("test", {}).get("skipped_pairs")

        if mode == "set":
            test = metrics.get("test", {})
            model_edist = format_ci(test.get("edist_mean"), test.get("edist_ci95"))
            baselines = metrics.get("baselines") or {}
            if not baselines:
                missing_baselines += 1
            no_change = baselines.get("no_change", {})
            mean_shift = baselines.get("mean_shift", {})
            ridge = baselines.get("ridge", {})
            ridge_alpha = baselines.get("ridge_alpha")
            rows_set.append(
                [
                    run_id,
                    split,
                    str(seed),
                    ablation,
                    str(n_eval if n_eval is not None else ""),
                    str(skipped if skipped is not None else ""),
                    model_edist,
                    format_ci(no_change.get("edist_mean"), no_change.get("edist_ci95")),
                    format_ci(mean_shift.get("edist_mean"), mean_shift.get("edist_ci95")),
                    format_ci(ridge.get("edist_mean"), ridge.get("edist_ci95")),
                    str(ridge_alpha if ridge_alpha is not None else ""),
                    tag,
                ]
            )
        elif mode == "prototype":
            test = metrics.get("test", {})
            mse = format_ci(test.get("mse_mean"), test.get("mse_ci95"))
            cos = format_ci(test.get("cosine_mean"), test.get("cosine_ci95"))
            rows_proto.append(
                [
                    run_id,
                    split,
                    str(seed),
                    ablation,
                    str(n_eval if n_eval is not None else ""),
                    str(skipped if skipped is not None else ""),
                    mse,
                    cos,
                    tag,
                ]
            )

    rows_set.sort(key=lambda r: (r[1], r[0], r[2], r[-1]))
    rows_proto.sort(key=lambda r: (r[1], r[0], r[2], r[-1]))

    content = [
        "# M3 Sciplex3 Report (Auto-generated)",
        "",
        "This report summarizes sciplex3-only M3 runs with set/prototype metrics and baselines.",
        "",
        f"Baselines missing in {missing_baselines} set-mode runs (older metrics files)." if missing_baselines else "Baselines present for all set-mode runs.",
        "",
        "## Set-level results (E-distance)",
        "",
        render_table(
            [
                "run_id",
                "split",
                "seed",
                "ablation",
                "n_eval",
                "skipped",
                "model_edist",
                "no_change",
                "mean_shift",
                "ridge",
                "ridge_alpha",
                "tag",
            ],
            rows_set,
        ),
        "",
        "## Prototype results",
        "",
        render_table(
            [
                "run_id",
                "split",
                "seed",
                "ablation",
                "n_eval",
                "skipped",
                "model_mse",
                "model_cosine",
                "tag",
            ],
            rows_proto,
        ),
        "",
        "Notes:",
        "- CIs are bootstrap 95% over condition pairs if available.",
        "- `ablation` reflects JEPA config fields saved alongside the checkpoint when available.",
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(content), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
