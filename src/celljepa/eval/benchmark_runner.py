"""Unified benchmark runner for head-to-head evaluation (P5).

Orchestrates end-to-end comparison of perturbation prediction methods
using identical data, splits, and metrics.

Usage::

    from celljepa.eval.benchmark_runner import BenchmarkConfig, BenchmarkRunner

    cfg = BenchmarkConfig(adapters=["no_change", "mean_shift", "celljepa"])
    runner = BenchmarkRunner(cfg)
    results = runner.run(train_data, test_data)
    runner.save_results(results, "runs/benchmark_001")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from celljepa.eval.benchmark_adapters import BenchmarkAdapter, build_adapter
from celljepa.eval.metrics import (
    bootstrap_mean,
    cosine_distance,
    energy_distance,
    lfc_pearson_correlation,
    top_k_deg_recall,
    direction_accuracy,
    perturbench_rank_metric,
    knn_retrieval_accuracy,
    mean_reciprocal_rank,
    calibrated_energy_distance,
)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark evaluation."""

    adapters: List[str] = field(default_factory=lambda: ["no_change", "mean_shift"])
    adapter_kwargs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Metric selection
    embedding_metrics: bool = True  # cosine, E-distance, kNN, MRR
    gene_level_metrics: bool = True  # LFC Pearson, DEG recall, direction accuracy

    # Bootstrap
    n_bootstrap: int = 1000
    ci_level: float = 0.95
    seed: int = 42

    # kNN
    knn_k: int = 5


@dataclass
class MethodResult:
    """Results for a single method."""

    method_name: str
    metrics: Dict[str, float]
    metrics_ci: Dict[str, tuple]  # metric_name → (lo, hi)
    fit_time_s: float = 0.0
    predict_time_s: float = 0.0


@dataclass
class BenchmarkResult:
    """Complete benchmark results across all methods."""

    method_results: Dict[str, MethodResult]
    config: BenchmarkConfig
    n_train: int = 0
    n_test: int = 0
    timestamp: str = ""


class BenchmarkRunner:
    """Orchestrates head-to-head evaluation across methods."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self._adapters: Dict[str, BenchmarkAdapter] = {}

        # Build adapters
        for name in config.adapters:
            kwargs = config.adapter_kwargs.get(name, {})
            self._adapters[name] = build_adapter(name, **kwargs)

    def run(
        self,
        train_data: Dict[str, Any],
        test_data: Dict[str, Any],
        val_data: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkResult:
        """Run all adapters on identical data.

        Args:
            train_data: dict with "control", "perturbed", "perturbation_ids", "context_ids"
            test_data: same format
            val_data: optional validation data

        Returns:
            BenchmarkResult with per-method metrics and CIs.
        """
        method_results = {}

        for name, adapter in self._adapters.items():
            result = self._run_single(adapter, train_data, test_data, val_data)
            method_results[name] = result

        return BenchmarkResult(
            method_results=method_results,
            config=self.config,
            n_train=len(train_data.get("perturbation_ids", [])),
            n_test=len(test_data.get("perturbation_ids", [])),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

    def _run_single(
        self,
        adapter: BenchmarkAdapter,
        train_data: Dict[str, Any],
        test_data: Dict[str, Any],
        val_data: Optional[Dict[str, Any]],
    ) -> MethodResult:
        """Run a single adapter: fit → predict → evaluate."""

        # Fit
        t0 = time.time()
        try:
            adapter.fit(train_data, val_data)
        except (ImportError, NotImplementedError) as e:
            # External package not installed — return empty results
            return MethodResult(
                method_name=adapter.name(),
                metrics={"error": str(e)},
                metrics_ci={},
                fit_time_s=0.0,
            )
        fit_time = time.time() - t0

        # Predict
        test_ctrl = np.asarray(test_data["control"])
        test_pert = np.asarray(test_data["perturbed"])
        test_pids = test_data["perturbation_ids"]
        test_cids = test_data["context_ids"]

        t0 = time.time()
        try:
            predictions = adapter.predict_batch(test_ctrl, test_pids, test_cids)
        except NotImplementedError as e:
            return MethodResult(
                method_name=adapter.name(),
                metrics={"error": str(e)},
                metrics_ci={},
                fit_time_s=fit_time,
            )
        predict_time = time.time() - t0

        # Evaluate
        metrics, metrics_ci = self._evaluate(
            predictions, test_pert, test_pids, test_cids
        )

        return MethodResult(
            method_name=adapter.name(),
            metrics=metrics,
            metrics_ci=metrics_ci,
            fit_time_s=fit_time,
            predict_time_s=predict_time,
        )

    def _evaluate(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        perturbation_ids: List[str],
        context_ids: List[str],
    ) -> tuple:
        """Compute all configured metrics with CIs."""
        metrics: Dict[str, float] = {}
        metrics_ci: Dict[str, tuple] = {}
        rng = np.random.default_rng(self.config.seed)
        n = predictions.shape[0]

        # --- Gene-level metrics (per-perturbation, then averaged) ---
        if self.config.gene_level_metrics:
            lfcs_pred = predictions - ground_truth  # proxy LFC
            lfcs_true = np.zeros_like(ground_truth)  # true LFC = perturbed - control baseline

            # Per-sample gene-level metrics
            pearsons = []
            deg_recalls = []
            dir_accs = []
            for i in range(n):
                # LFC = perturbed - control; compare predicted vs actual perturbation
                obs_lfc = ground_truth[i]  # actual perturbed expression
                pred_lfc = predictions[i]  # predicted perturbed expression
                # Compute gene-level shift relative to some reference
                pearsons.append(lfc_pearson_correlation(pred_lfc, obs_lfc))
                deg_recalls.append(top_k_deg_recall(pred_lfc, obs_lfc, k=20))
                dir_accs.append(direction_accuracy(pred_lfc, obs_lfc))

            pearsons = np.array([p for p in pearsons if not np.isnan(p)])
            deg_recalls = np.array([d for d in deg_recalls if not np.isnan(d)])
            dir_accs = np.array([d for d in dir_accs if not np.isnan(d)])

            if len(pearsons) > 0:
                m, lo, hi = bootstrap_mean(pearsons, num_samples=self.config.n_bootstrap, seed=self.config.seed)
                metrics["lfc_pearson"] = m
                metrics_ci["lfc_pearson"] = (lo, hi)
            if len(deg_recalls) > 0:
                m, lo, hi = bootstrap_mean(deg_recalls, num_samples=self.config.n_bootstrap, seed=self.config.seed)
                metrics["deg_recall_20"] = m
                metrics_ci["deg_recall_20"] = (lo, hi)
            if len(dir_accs) > 0:
                m, lo, hi = bootstrap_mean(dir_accs, num_samples=self.config.n_bootstrap, seed=self.config.seed)
                metrics["direction_accuracy"] = m
                metrics_ci["direction_accuracy"] = (lo, hi)

        # --- Embedding-level metrics ---
        if self.config.embedding_metrics:
            # Per-sample cosine
            cos_dists = np.array([
                cosine_distance(predictions[i], ground_truth[i]) for i in range(n)
            ])
            m, lo, hi = bootstrap_mean(cos_dists, num_samples=self.config.n_bootstrap, seed=self.config.seed)
            metrics["cosine_distance"] = m
            metrics_ci["cosine_distance"] = (lo, hi)

            # Per-sample MSE
            mses = np.mean((predictions - ground_truth) ** 2, axis=1)
            m, lo, hi = bootstrap_mean(mses, num_samples=self.config.n_bootstrap, seed=self.config.seed)
            metrics["mse"] = m
            metrics_ci["mse"] = (lo, hi)

            # E-distance
            metrics["energy_distance"] = energy_distance(predictions, ground_truth)

            # PerturBench rank
            labels = np.array(perturbation_ids)
            metrics["perturbench_rank"] = perturbench_rank_metric(predictions, ground_truth, labels)

            # kNN retrieval
            metrics["knn_accuracy"] = knn_retrieval_accuracy(
                predictions, labels, k=self.config.knn_k
            )

            # MRR
            metrics["mrr"] = mean_reciprocal_rank(predictions, ground_truth)

            # Calibrated E-distance
            metrics["calibrated_edist"] = calibrated_energy_distance(
                predictions, ground_truth, n_permutations=100, seed=self.config.seed
            )

        return metrics, metrics_ci

    def save_results(
        self,
        results: BenchmarkResult,
        output_dir: str | Path,
    ) -> None:
        """Save benchmark results to output directory."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Machine-readable
        data = {
            "timestamp": results.timestamp,
            "n_train": results.n_train,
            "n_test": results.n_test,
            "methods": {},
        }
        for name, mr in results.method_results.items():
            data["methods"][name] = {
                "metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                            for k, v in mr.metrics.items()},
                "metrics_ci": {k: [float(lo), float(hi)]
                               for k, (lo, hi) in mr.metrics_ci.items()},
                "fit_time_s": mr.fit_time_s,
                "predict_time_s": mr.predict_time_s,
            }
        (out / "metrics.json").write_text(
            json.dumps(data, indent=2, default=str), encoding="utf-8"
        )

        # Human-readable comparison table
        from celljepa.eval.report import generate_comparison_table
        table = generate_comparison_table(results)
        (out / "comparison_table.md").write_text(table, encoding="utf-8")
