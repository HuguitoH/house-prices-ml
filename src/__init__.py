"""
Shared ML pipeline modules for house-prices-ml.

Modules:
    registry  - TransformationRegistry
    features  - engineer_base_features, engineer_extended_features,
                fit_neighborhood_medians
    pipeline  - impute_semantic, build_column_transformer
    evaluate  - compute_metrics, evaluate_model
"""

from src.registry import TransformationRegistry
from src.features import (
    engineer_base_features,
    engineer_extended_features,
    fit_neighborhood_medians,
)
from src.pipeline import impute_semantic, build_column_transformer
from src.evaluate import compute_metrics, evaluate_model

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
