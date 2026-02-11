#!/usr/bin/env python3
"""Smoke test for P5 benchmark evaluation suite.

Tests extended metrics, adapter interface, benchmark runner, and report
generation using synthetic data. No external dependencies required.

Usage:
    python3 scripts/test_benchmark_suite.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_perturbench_rank() -> bool:
    print("--- perturbench_rank_metric ---")
    from celljepa.eval.metrics import perturbench_rank_metric

    rng = np.random.default_rng(42)

    # Perfect match: pred == true → rank should be 0
    embeddings = rng.standard_normal((10, 8))
    labels = np.array([f"pert_{i}" for i in range(10)])
    rank = perturbench_rank_metric(embeddings, embeddings, labels)
    assert rank == 0.0, f"Perfect match should give rank=0, got {rank}"
    print(f"  Perfect match rank: {rank:.4f} ✓")

    # Random predictions: rank should be ~0.5
    random_pred = rng.standard_normal((10, 8))
    rank_random = perturbench_rank_metric(random_pred, embeddings, labels)
    assert 0.0 < rank_random < 1.0, f"Random rank out of range: {rank_random}"
    print(f"  Random match rank: {rank_random:.4f} ✓")

    return True


def test_knn_retrieval() -> bool:
    print("--- knn_retrieval_accuracy ---")
    from celljepa.eval.metrics import knn_retrieval_accuracy

    # Clustered embeddings: same-label points are near each other
    np.random.seed(42)
    embeddings = np.vstack([
        np.random.randn(5, 4) + np.array([5, 0, 0, 0]),  # cluster A
        np.random.randn(5, 4) + np.array([0, 5, 0, 0]),  # cluster B
        np.random.randn(5, 4) + np.array([0, 0, 5, 0]),  # cluster C
    ])
    labels = np.array(["A"] * 5 + ["B"] * 5 + ["C"] * 5)

    acc = knn_retrieval_accuracy(embeddings, labels, k=3)
    assert 0.5 < acc <= 1.0, f"Clustered data should have high kNN acc, got {acc}"
    print(f"  Clustered kNN accuracy (k=3): {acc:.4f} ✓")

    # Random embeddings: accuracy should be ~1/3
    random_emb = np.random.randn(15, 4)
    acc_rand = knn_retrieval_accuracy(random_emb, labels, k=3)
    print(f"  Random kNN accuracy (k=3): {acc_rand:.4f} ✓")

    return True


def test_mean_reciprocal_rank() -> bool:
    print("--- mean_reciprocal_rank ---")
    from celljepa.eval.metrics import mean_reciprocal_rank

    rng = np.random.default_rng(42)

    # Perfect match → MRR should be 1.0
    embeddings = rng.standard_normal((8, 5))
    mrr = mean_reciprocal_rank(embeddings, embeddings)
    assert abs(mrr - 1.0) < 1e-6, f"Perfect match MRR should be 1.0, got {mrr}"
    print(f"  Perfect match MRR: {mrr:.4f} ✓")

    # Random → MRR should be low
    random_pred = rng.standard_normal((8, 5))
    mrr_rand = mean_reciprocal_rank(random_pred, embeddings)
    assert 0.0 < mrr_rand <= 1.0
    print(f"  Random MRR: {mrr_rand:.4f} ✓")

    return True


def test_calibrated_edist() -> bool:
    print("--- calibrated_energy_distance ---")
    from celljepa.eval.metrics import calibrated_energy_distance

    rng = np.random.default_rng(42)

    # Identical sets → calibrated E-dist should be < 1 (much better than null)
    x = rng.standard_normal((20, 4))
    noise = rng.standard_normal((20, 4)) * 0.01
    cal = calibrated_energy_distance(x, x + noise, n_permutations=50, seed=42)
    assert cal < 1.0, f"Near-identical should beat null: {cal}"
    print(f"  Near-identical: {cal:.4f} (< 1.0) ✓")

    # Well-separated → calibrated E-dist should be > 0
    y = rng.standard_normal((20, 4)) + 10
    cal_sep = calibrated_energy_distance(x, y, n_permutations=50, seed=42)
    assert cal_sep > 0
    print(f"  Separated: {cal_sep:.4f} ✓")

    return True


def test_adapters() -> bool:
    print("--- Benchmark Adapters ---")
    from celljepa.eval.benchmark_adapters import (
        NoChangeAdapter, MeanShiftAdapter, build_adapter, ADAPTER_REGISTRY,
    )

    # Registry check
    assert len(ADAPTER_REGISTRY) >= 6
    print(f"  Registry: {len(ADAPTER_REGISTRY)} adapters ✓")

    # NoChange
    nc = build_adapter("no_change")
    assert nc.name() == "no_change"
    nc.fit({})
    ctrl = np.array([1.0, 2.0, 3.0])
    pred = nc.predict(ctrl, "pert_x", "ctx_a")
    assert np.array_equal(pred, ctrl)
    print(f"  NoChangeAdapter: predict == control ✓")

    # MeanShift
    ms = build_adapter("mean_shift")
    train_data = {
        "control": np.array([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0]]),
        "perturbed": np.array([[2.0, 3.0], [4.0, 5.0], [3.0, 4.0]]),
        "perturbation_ids": ["A", "B", "A"],
        "context_ids": ["c1", "c1", "c1"],
    }
    ms.fit(train_data)
    pred_a = ms.predict(np.array([0.0, 0.0]), "A", "c1")
    # Mean shift for A: ((2-1)+(3-1)) / 2 = 1.0, ((3-2)+(4-2)) / 2 = 1.5
    expected_shift = np.mean(train_data["perturbed"][[0, 2]] - train_data["control"][[0, 2]], axis=0)
    assert np.allclose(pred_a, expected_shift), f"MeanShift: {pred_a} vs {expected_shift}"
    print(f"  MeanShiftAdapter: shift = {expected_shift} ✓")

    # Batch predict
    preds = nc.predict_batch(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        ["A", "B"],
        ["c1", "c1"],
    )
    assert preds.shape == (2, 2)
    print(f"  Batch predict: {preds.shape} ✓")

    return True


def test_benchmark_runner() -> bool:
    print("--- BenchmarkRunner ---")
    from celljepa.eval.benchmark_runner import BenchmarkConfig, BenchmarkRunner

    rng = np.random.default_rng(42)
    n_train, n_test, n_genes = 30, 10, 20

    train_data = {
        "control": rng.standard_normal((n_train, n_genes)),
        "perturbed": rng.standard_normal((n_train, n_genes)),
        "perturbation_ids": [f"pert_{i % 5}" for i in range(n_train)],
        "context_ids": [f"ctx_{i % 2}" for i in range(n_train)],
    }
    test_data = {
        "control": rng.standard_normal((n_test, n_genes)),
        "perturbed": rng.standard_normal((n_test, n_genes)),
        "perturbation_ids": [f"pert_{i % 5}" for i in range(n_test)],
        "context_ids": [f"ctx_{i % 2}" for i in range(n_test)],
    }

    cfg = BenchmarkConfig(
        adapters=["no_change", "mean_shift"],
        n_bootstrap=50,  # Fewer for speed
    )
    runner = BenchmarkRunner(cfg)
    results = runner.run(train_data, test_data)

    assert "no_change" in results.method_results
    assert "mean_shift" in results.method_results
    print(f"  Methods run: {list(results.method_results.keys())} ✓")

    # Check metrics exist
    nc_metrics = results.method_results["no_change"].metrics
    assert "cosine_distance" in nc_metrics
    assert "mse" in nc_metrics
    assert "perturbench_rank" in nc_metrics
    print(f"  Metrics: {list(nc_metrics.keys())} ✓")

    # CIs exist
    nc_ci = results.method_results["no_change"].metrics_ci
    assert "cosine_distance" in nc_ci
    lo, hi = nc_ci["cosine_distance"]
    assert lo <= nc_metrics["cosine_distance"] <= hi
    print(f"  CIs present: {len(nc_ci)} metrics with CIs ✓")

    return True


def test_report_generation() -> bool:
    print("--- Report Generation ---")
    from celljepa.eval.benchmark_runner import BenchmarkConfig, BenchmarkRunner
    from celljepa.eval.report import generate_comparison_table, generate_benchmark_report

    rng = np.random.default_rng(42)
    n = 15

    train_data = {
        "control": rng.standard_normal((n, 10)),
        "perturbed": rng.standard_normal((n, 10)),
        "perturbation_ids": [f"p{i}" for i in range(n)],
        "context_ids": [f"c{i % 3}" for i in range(n)],
    }
    test_data = {
        "control": rng.standard_normal((n, 10)),
        "perturbed": rng.standard_normal((n, 10)),
        "perturbation_ids": [f"p{i}" for i in range(n)],
        "context_ids": [f"c{i % 3}" for i in range(n)],
    }

    cfg = BenchmarkConfig(adapters=["no_change", "mean_shift"], n_bootstrap=20)
    runner = BenchmarkRunner(cfg)
    results = runner.run(train_data, test_data)

    # Comparison table
    table = generate_comparison_table(results)
    assert "| Metric |" in table
    assert "no_change" in table
    assert "mean_shift" in table
    print(f"  Comparison table: {len(table)} chars ✓")

    # Full report
    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = Path(tmpdir) / "report.md"
        generate_benchmark_report(results, report_path)
        assert report_path.exists()
        content = report_path.read_text()
        assert "Benchmark" in content
        print(f"  Full report: {len(content)} chars ✓")

        # Save results
        runner.save_results(results, tmpdir)
        assert (Path(tmpdir) / "metrics.json").exists()
        assert (Path(tmpdir) / "comparison_table.md").exists()
        print(f"  Saved: metrics.json + comparison_table.md ✓")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    tests = [
        ("perturbench_rank_metric", test_perturbench_rank),
        ("knn_retrieval_accuracy", test_knn_retrieval),
        ("mean_reciprocal_rank", test_mean_reciprocal_rank),
        ("calibrated_energy_distance", test_calibrated_edist),
        ("Benchmark Adapters", test_adapters),
        ("BenchmarkRunner", test_benchmark_runner),
        ("Report Generation", test_report_generation),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, "PASS" if passed else "FAIL"))
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, "FAIL"))
        print()

    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    all_pass = True
    for name, result in results:
        status = "✓" if result == "PASS" else "✗"
        print(f"  {status} {name}: {result}")
        if result != "PASS":
            all_pass = False

    if all_pass:
        print(f"\nAll {len(results)} tests PASSED.")
    else:
        n_fail = sum(1 for _, r in results if r != "PASS")
        print(f"\n{n_fail}/{len(results)} tests FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
