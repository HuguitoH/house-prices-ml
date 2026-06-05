import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from src.pipeline import build_column_transformer, impute_semantic
from src.registry import TransformationRegistry


# Fixtures

@pytest.fixture
def raw_df() -> pd.DataFrame:
    return pd.DataFrame({
        "PoolQC":       [None,    None,     "Ex"],
        "FireplaceQu":  [None,    "Gd",     None],
        "GarageArea":   [np.nan,  400.0,    np.nan],
        "MasVnrArea":   [np.nan,  100.0,    0.0],
        "LotFrontage":  [np.nan,  70.0,     np.nan],
        "Neighborhood": ["NAmes", "NAmes",  "OldTown"],
        "Electrical":   [None,    "SBrkr",  None],
        "GrLivArea":    [1500,    1200,     1800],
    })


@pytest.fixture
def simple_registry() -> TransformationRegistry:
    df = pd.DataFrame({
        "num_col":  [1.0, 2.0, 3.0],
        "cat_col":  ["a", "b", "a"],
        "pass_col": [1,   2,   3],
    })
    reg = TransformationRegistry(df)
    reg.set("num_col",  "standard_scaler", "numeric")
    reg.set("cat_col",  "one_hot_encoding", "categorical")
    reg.set("pass_col", "passthrough",      "passthrough")
    return reg


# impute_semantic

class TestImputeSemantic:

    def test_categorical_absent_filled_with_none_string(self, raw_df):
        out = impute_semantic(raw_df)
        assert out["PoolQC"].iloc[0] == "None"
        assert out["FireplaceQu"].iloc[0] == "None"

    def test_numeric_absent_filled_with_zero(self, raw_df):
        out = impute_semantic(raw_df)
        assert out["GarageArea"].iloc[0] == pytest.approx(0.0)
        assert out["MasVnrArea"].iloc[0] == pytest.approx(0.0)

    def test_lot_frontage_filled_with_neighbourhood_median(self, raw_df):
        out = impute_semantic(raw_df)
        # NAmes has LotFrontage=70 -> NaN filled with 70
        assert out["LotFrontage"].iloc[0] == pytest.approx(70.0)

    def test_mode_impute(self, raw_df):
        out = impute_semantic(raw_df)
        assert out["Electrical"].iloc[0] == "SBrkr"
        assert out["Electrical"].iloc[2] == "SBrkr"

    def test_does_not_mutate_input(self, raw_df):
        original_nulls = raw_df["PoolQC"].isna().sum()
        impute_semantic(raw_df)
        assert raw_df["PoolQC"].isna().sum() == original_nulls

    def test_missing_columns_skipped_gracefully(self):
        df = pd.DataFrame({"GrLivArea": [1500, 1200]})
        out = impute_semantic(df)
        assert out.shape == df.shape


# build_column_transformer

class TestBuildColumnTransformer:

    def test_returns_column_transformer(self, simple_registry):
        ct = build_column_transformer(simple_registry)
        assert isinstance(ct, ColumnTransformer)

    def test_remainder_is_drop(self, simple_registry):
        ct = build_column_transformer(simple_registry)
        assert ct.remainder == "drop"

    def test_unknown_categories_do_not_raise(self):
        df_train = pd.DataFrame({"cat": ["a", "b", "a"]})
        df_test  = pd.DataFrame({"cat": ["a", "c", "b"]})
        reg = TransformationRegistry(df_train)
        reg.set("cat", "one_hot_encoding", "nominal")
        ct = build_column_transformer(reg)
        ct.fit(df_train)
        out = ct.transform(df_test)
        assert out.shape[0] == 3
