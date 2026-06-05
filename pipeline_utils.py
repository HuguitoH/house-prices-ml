"""
Backwards compatibility shim.
All logic has moved to src/.

This file re-exports everything from src/ so existing notebooks
that import from pipeline_utils continue to work without changes.
"""

from src import (
    TransformationRegistry,
    engineer_base_features,
    engineer_extended_features,
    fit_neighborhood_medians,
    impute_semantic,
    build_column_transformer,
    compute_metrics,
    evaluate_model,
)

__all__ = [
    "TransformationRegistry",
    "engineer_base_features",
    "engineer_extended_features",
    "fit_neighborhood_medians",
    "impute_semantic",
    "build_column_transformer",
    "compute_metrics",
    "evaluate_model",
]
