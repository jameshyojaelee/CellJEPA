"""Benchmark adapters for SOTA perturbation response methods (P5).

Provides a uniform interface for running CellJEPA alongside GEARS, scGPT,
and CPA on identical data and splits. Each adapter wraps an external model
behind a consistent ``fit`` / ``predict`` API.

Usage::

    from celljepa.eval.benchmark_adapters import (
        NoChangeAdapter, CellJEPAAdapter, GEARSAdapter,
    )

    adapter = NoChangeAdapter()
    adapter.fit(train_data)
    pred = adapter.predict(control_expr, perturbation_id, context_id)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class BenchmarkAdapter(ABC):
    """Abstract benchmark adapter for perturbation prediction methods."""

    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the method."""
        ...

    @abstractmethod
    def fit(
        self,
        train_data: Dict[str, Any],
        val_data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Train/fit the method on training data.

        Args:
            train_data: dict with keys "control" (N, G), "perturbed" (N, G),
                        "perturbation_ids" (N,), "context_ids" (N,).
            val_data: optional validation data (same format).
        """
        ...

    @abstractmethod
    def predict(
        self,
        control_expr: np.ndarray,
        perturbation_id: str,
        context_id: str,
    ) -> np.ndarray:
        """Predict post-perturbation expression.

        Args:
            control_expr: (G,) control expression vector.
            perturbation_id: perturbation identifier.
            context_id: context (cell line / donor) identifier.

        Returns:
            (G,) predicted post-perturbation expression.
        """
        ...

    def predict_batch(
        self,
        control_exprs: np.ndarray,
        perturbation_ids: List[str],
        context_ids: List[str],
    ) -> np.ndarray:
        """Predict for a batch. Default loops over single predictions."""
        preds = []
        for ctrl, pid, cid in zip(control_exprs, perturbation_ids, context_ids):
            preds.append(self.predict(ctrl, pid, cid))
        return np.stack(preds)


# ---------------------------------------------------------------------------
# Trivial baselines
# ---------------------------------------------------------------------------

class NoChangeAdapter(BenchmarkAdapter):
    """Baseline: predict that perturbation has no effect (returns control)."""

    def name(self) -> str:
        return "no_change"

    def fit(self, train_data, val_data=None, **kwargs) -> None:
        pass  # Nothing to train

    def predict(self, control_expr, perturbation_id, context_id) -> np.ndarray:
        return control_expr.copy()


class MeanShiftAdapter(BenchmarkAdapter):
    """Baseline: predict control + mean shift observed in training for that perturbation."""

    def name(self) -> str:
        return "mean_shift"

    def fit(self, train_data, val_data=None, **kwargs) -> None:
        ctrl = np.asarray(train_data["control"])
        pert = np.asarray(train_data["perturbed"])
        pids = train_data["perturbation_ids"]

        self._shifts: Dict[str, np.ndarray] = {}
        self._global_shift = np.mean(pert - ctrl, axis=0)

        from collections import defaultdict
        groups: Dict[str, List[int]] = defaultdict(list)
        for i, pid in enumerate(pids):
            groups[pid].append(i)

        for pid, indices in groups.items():
            idx = np.array(indices)
            self._shifts[pid] = np.mean(pert[idx] - ctrl[idx], axis=0)

    def predict(self, control_expr, perturbation_id, context_id) -> np.ndarray:
        shift = self._shifts.get(perturbation_id, self._global_shift)
        return control_expr + shift


# ---------------------------------------------------------------------------
# CellJEPA adapter
# ---------------------------------------------------------------------------

class CellJEPAAdapter(BenchmarkAdapter):
    """Adapter wrapping CellJEPA model for benchmarking.

    Requires a trained CellJEPA checkpoint and configuration.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self._checkpoint_path = checkpoint_path
        self._config = config or {}
        self._model = None

    def name(self) -> str:
        return "celljepa"

    def fit(self, train_data, val_data=None, **kwargs) -> None:
        """Load model from checkpoint (no re-training in benchmark mode)."""
        if self._checkpoint_path is None:
            raise ValueError("CellJEPAAdapter requires checkpoint_path")

        import torch
        ckpt = torch.load(self._checkpoint_path, map_location="cpu")
        # Model loading depends on the specific architecture
        self._model_state = ckpt.get("model", ckpt)
        self._fitted = True

    def predict(self, control_expr, perturbation_id, context_id) -> np.ndarray:
        if not hasattr(self, "_fitted") or not self._fitted:
            raise RuntimeError("CellJEPAAdapter.fit() must be called first")
        # Placeholder: actual inference requires model instantiation
        # This will be filled in when models are trained
        return control_expr.copy()


# ---------------------------------------------------------------------------
# GEARS adapter
# ---------------------------------------------------------------------------

class GEARSAdapter(BenchmarkAdapter):
    """Adapter for GEARS (Roohani et al. 2023).

    Requires: ``pip install gears``
    """

    def __init__(self, **gears_kwargs):
        self._kwargs = gears_kwargs
        self._model = None

    def name(self) -> str:
        return "gears"

    def fit(self, train_data, val_data=None, **kwargs) -> None:
        try:
            import gears  # type: ignore
        except ImportError:
            raise ImportError(
                "GEARS is required: pip install gears\n"
                "See https://github.com/snap-stanford/GEARS"
            )
        # GEARS has its own data loading + training pipeline.
        # The adapter interface wraps it at the predict level.
        self._fitted = True

    def predict(self, control_expr, perturbation_id, context_id) -> np.ndarray:
        if not hasattr(self, "_fitted") or not self._fitted:
            raise RuntimeError("GEARSAdapter.fit() must be called first")
        raise NotImplementedError(
            "GEARS prediction requires a trained GEARS model. "
            "Use GEARS CLI to train, then load the checkpoint here."
        )


# ---------------------------------------------------------------------------
# scGPT adapter
# ---------------------------------------------------------------------------

class ScGPTAdapter(BenchmarkAdapter):
    """Adapter for scGPT (Cui et al. 2024).

    Requires: ``pip install scgpt``
    """

    def __init__(self, **scgpt_kwargs):
        self._kwargs = scgpt_kwargs
        self._model = None

    def name(self) -> str:
        return "scgpt"

    def fit(self, train_data, val_data=None, **kwargs) -> None:
        try:
            import scgpt  # type: ignore
        except ImportError:
            raise ImportError(
                "scGPT is required: pip install scgpt\n"
                "See https://github.com/bowang-lab/scGPT"
            )
        self._fitted = True

    def predict(self, control_expr, perturbation_id, context_id) -> np.ndarray:
        if not hasattr(self, "_fitted") or not self._fitted:
            raise RuntimeError("ScGPTAdapter.fit() must be called first")
        raise NotImplementedError(
            "scGPT prediction requires a fine-tuned scGPT model. "
            "Use scGPT's perturbation fine-tuning pipeline first."
        )


# ---------------------------------------------------------------------------
# CPA adapter
# ---------------------------------------------------------------------------

class CPAAdapter(BenchmarkAdapter):
    """Adapter for CPA (Lotfollahi et al. 2023).

    Requires: ``pip install cpa-tools``
    """

    def __init__(self, **cpa_kwargs):
        self._kwargs = cpa_kwargs
        self._model = None

    def name(self) -> str:
        return "cpa"

    def fit(self, train_data, val_data=None, **kwargs) -> None:
        try:
            import cpa  # type: ignore
        except ImportError:
            raise ImportError(
                "CPA is required: pip install cpa-tools\n"
                "See https://github.com/theislab/cpa"
            )
        self._fitted = True

    def predict(self, control_expr, perturbation_id, context_id) -> np.ndarray:
        if not hasattr(self, "_fitted") or not self._fitted:
            raise RuntimeError("CPAAdapter.fit() must be called first")
        raise NotImplementedError(
            "CPA prediction requires a trained CPA model. "
            "Use CPA's training pipeline first."
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ADAPTER_REGISTRY: Dict[str, type] = {
    "no_change": NoChangeAdapter,
    "mean_shift": MeanShiftAdapter,
    "celljepa": CellJEPAAdapter,
    "gears": GEARSAdapter,
    "scgpt": ScGPTAdapter,
    "cpa": CPAAdapter,
}


def build_adapter(name: str, **kwargs) -> BenchmarkAdapter:
    """Factory function for benchmark adapters."""
    cls = ADAPTER_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown adapter: {name}. Available: {list(ADAPTER_REGISTRY.keys())}"
        )
    return cls(**kwargs)
