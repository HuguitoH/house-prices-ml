"""
Preprocessing pipeline construction and semantic imputation.

impute_semantic() handles domain-specific missing value logic before
the sklearn Pipeline takes over. The Pipeline itself never sees raw
NaN values from categorical absent-feature columns.

build_column_transformer() reads a TransformationRegistry and constructs
a ColumnTransformer with one sub-pipeline per transformation type.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

from src.registry import TransformationRegistry


# Columns where NaN means the feature does not exist
_CAT_ABSENT: list[str] = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
    "BsmtFinType2", "MasVnrType",
]

# Columns where NaN means the numeric quantity is zero
_NUM_ABSENT: list[str] = [
    "GarageYrBlt", "GarageArea", "GarageCars",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath", "MasVnrArea",
]

# Columns with isolated missing values - imputed with mode
_MODE_IMPUTE: list[str] = [
    "Electrical", "MSZoning", "Exterior1st", "Exterior2nd",
    "KitchenQual", "SaleType", "Functional",
]


def impute_semantic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply domain-driven imputation before the sklearn Pipeline.

    Three strategies based on missing value semantics:
        1. Categorical absent features - fill with 'None' (valid category)
        2. Numeric absent features - fill with 0
        3. LotFrontage - neighbourhood median (truly missing, local context)
        4. Isolated cases - column mode

    Must be called on both train and test before any transformation.
    """
    out = df.copy()

    for col in _CAT_ABSENT:
        if col in out.columns:
            out[col] = out[col].fillna("None")

    for col in _NUM_ABSENT:
        if col in out.columns:
            out[col] = out[col].fillna(0)

    if "LotFrontage" in out.columns and "Neighborhood" in out.columns:
        out["LotFrontage"] = out.groupby("Neighborhood")["LotFrontage"].transform(
            lambda x: x.fillna(x.median())
        )

    for col in _MODE_IMPUTE:
        if col in out.columns:
            out[col] = out[col].fillna(out[col].mode()[0])

    return out


def build_column_transformer(registry: TransformationRegistry) -> ColumnTransformer:
    """
    Build a ColumnTransformer from a TransformationRegistry.

    Each transformation type maps to a named sub-pipeline:
        passthrough      - median imputation only
        standard_scaler  - median imputation + StandardScaler
        robust_scaler    - median imputation + RobustScaler
        ordinal_encoder  - mode imputation + OrdinalEncoder
        one_hot_encoding - mode imputation + OneHotEncoder

    remainder='drop' ensures unregistered columns never silently pass through.
    Accepts the registry directly - call registry.features_only() before
    passing if the registry still contains target/drop columns.
    """

    def sub_pipeline(imputer: SimpleImputer, transformer=None) -> Pipeline:
        steps = [("imputer", imputer)]
        if transformer is not None:
            steps.append(("transformer", transformer))
        return Pipeline(steps)

    mapping = [
        (
            "passthrough",
            sub_pipeline(SimpleImputer(strategy="median")),
            registry.get_columns("passthrough"),
        ),
        (
            "std_scaler",
            sub_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
            registry.get_columns("standard_scaler"),
        ),
        (
            "rob_scaler",
            sub_pipeline(SimpleImputer(strategy="median"), RobustScaler()),
            registry.get_columns("robust_scaler"),
        ),
        (
            "ordinal",
            sub_pipeline(
                SimpleImputer(strategy="most_frequent"),
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
            registry.get_columns("ordinal_encoder"),
        ),
        (
            "ohe",
            sub_pipeline(
                SimpleImputer(strategy="most_frequent"),
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
            registry.get_columns("one_hot_encoding"),
        ),
    ]

    transformers = [(name, pipe, cols) for name, pipe, cols in mapping if cols]
    return ColumnTransformer(transformers=transformers, remainder="drop")
