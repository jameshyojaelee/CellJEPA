"""Evaluation utilities and report helpers."""

from .baselines import build_pairs, evaluate_baselines  # noqa: F401
from .metrics import cosine_distance, energy_distance  # noqa: F401
from .report import write_report  # noqa: F401
