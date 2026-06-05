import numpy as np
import pandas as pd
import pytest

from src.features import (
    engineer_base_features,
    engineer_extended_features,
    fit_neighborhood_medians,
)


#  Fixtures

@pytest.fixture
def base_df() -> pd.DataFrame:
    return pd.DataFrame({
        "TotalBsmtSF":  [800.0, np.nan, 1200.0],
        "1stFlrSF":     [1000,  800,    1200],
        "2ndFlrSF":     [500,   0,      600],
        "YrSold":       [2010,  2008,   2009],
        "YearBuilt":    [1990,  2005,   1960],
        "YearRemodAdd": [2005,  2005,   1960],
        "FullBath":     [2,     np.nan, 2],
        "BsmtFullBath": [1,     np.nan, 1],
        "HalfBath":     [1,     np.nan, 0],
        "BsmtHalfBath": [0,     np.nan, 1],
        "OpenPorchSF":  [100,   np.nan, 50],
        "EnclosedPorch":[0,     80,     0],
        "3SsnPorch":    [0,     0,      0],
        "ScreenPorch":  [0,     0,      30],
    })


@pytest.fixture
def extended_df(base_df) -> pd.DataFrame:
    df = engineer_base_features(base_df)
    df["OverallQual"]   = [7,      5,       8]
    df["OverallCond"]   = [5,      6,       4]
    df["GarageCars"]    = [2,      0,       1]
    df["LotArea"]       = [8000,   6000,    10000]
    df["GrLivArea"]     = [1500,   800,     1800]
    df["Neighborhood"]  = ["NAmes", "OldTown", "NAmes"]
    df["SaleCondition"] = ["Normal", "Abnorml", "Normal"]
    df["Functional"]    = ["Typ",  "Min1",  "Mod"]
    df["SalePrice"]     = [180000, 120000,  250000]
    return df


@pytest.fixture
def neighborhood_medians(extended_df) -> dict:
    return fit_neighborhood_medians(extended_df, target="SalePrice")


#  engineer_base_features

class TestEngineerBaseFeatures:

    def test_nan_filled_with_zero_before_arithmetic(self, base_df):
        out = engineer_base_features(base_df)
        assert not out["TotalSF"].isna().any()
        assert not out["TotalBaths"].isna().any()
        assert not out["TotalPorchSF"].isna().any()

    def test_was_remodeled_flag(self, base_df):
        out = engineer_base_features(base_df)
        assert out["WasRemodeled"].tolist() == [1, 0, 0]

    def test_all_base_features_created(self, base_df):
        out = engineer_base_features(base_df)
        for col in ["TotalSF", "HouseAge", "RemodAge", "WasRemodeled", "TotalBaths", "TotalPorchSF"]:
            assert col in out.columns

    def test_does_not_mutate_input(self, base_df):
        original_cols = set(base_df.columns)
        engineer_base_features(base_df)
        assert set(base_df.columns) == original_cols


#  fit_neighborhood_medians

class TestFitNeighborhoodMedians:

    def test_correct_median_per_neighbourhood(self, extended_df):
        result = fit_neighborhood_medians(extended_df)
        assert result["NAmes"] == pytest.approx(215000.0)
        assert result["OldTown"] == pytest.approx(120000.0)

    def test_returns_dict(self, extended_df):
        result = fit_neighborhood_medians(extended_df)
        assert isinstance(result, dict)


#  engineer_extended_features

class TestEngineerExtendedFeatures:

    def test_all_extended_features_created(self, extended_df, neighborhood_medians):
        out = engineer_extended_features(extended_df, neighborhood_medians)
        expected = [
            "QualSF", "QualAge", "HasBasement", "HasGarage",
            "BsmtRatio", "LotDensity", "NeighborhoodTier",
            "QualCond", "IsOldHouse", "IsNormalSale", "FuncQual", "TotalQualSF",
        ]
        for col in expected:
            assert col in out.columns

    def test_unknown_neighbourhood_falls_back_to_global_median(self, extended_df, neighborhood_medians):
        df = extended_df.copy()
        df.loc[0, "Neighborhood"] = "UnknownPlace"
        out = engineer_extended_features(df, neighborhood_medians)
        global_median = float(np.median(list(neighborhood_medians.values())))
        assert out["NeighborhoodTier"].iloc[0] == pytest.approx(global_median)

    def test_empty_medians_falls_back_to_163000(self, extended_df):
        out = engineer_extended_features(extended_df, {})
        assert out["NeighborhoodTier"].iloc[0] == pytest.approx(163_000.0)

    def test_is_normal_sale_binary(self, extended_df, neighborhood_medians):
        out = engineer_extended_features(extended_df, neighborhood_medians)
        assert out["IsNormalSale"].tolist() == [1, 0, 1]

    def test_does_not_mutate_input(self, extended_df, neighborhood_medians):
        original_cols = set(extended_df.columns)
        engineer_extended_features(extended_df, neighborhood_medians)
        assert set(extended_df.columns) == original_cols
