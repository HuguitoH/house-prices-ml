"""
Feature engineering functions for the house prices pipeline.

Two levels of features:
- Base features: created in notebook 01, exported via pipeline_state.pkl
- Extended features: created in notebook 02, require base features to exist

All functions are pure - no side effects, no global state.
Applied identically to train and test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def engineer_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create base features from raw housing columns.

    Features created:
        TotalSF      - total living area (basement + 1F + 2F)
        HouseAge     - years between build year and sale year
        RemodAge     - years since last remodel
        WasRemodeled - binary flag, has the house ever been remodelled
        TotalBaths   - weighted bath count (full=1, half=0.5)
        TotalPorchSF - combined porch area, replaces 4 low-signal columns

    All source columns filled with 0 before arithmetic to handle missing
    values in test set without leaking train statistics.
    """
    out = df.copy()

    out["TotalSF"] = (
        out["TotalBsmtSF"].fillna(0)
        + out["1stFlrSF"].fillna(0)
        + out["2ndFlrSF"].fillna(0)
    )
    out["HouseAge"]     = out["YrSold"] - out["YearBuilt"]
    out["RemodAge"]     = out["YrSold"] - out["YearRemodAdd"]
    out["WasRemodeled"] = (out["YearRemodAdd"] != out["YearBuilt"]).astype(int)
    out["TotalBaths"]   = (
        out["FullBath"].fillna(0)
        + out["BsmtFullBath"].fillna(0)
        + 0.5 * (out["HalfBath"].fillna(0) + out["BsmtHalfBath"].fillna(0))
    )
    out["TotalPorchSF"] = (
        out["OpenPorchSF"].fillna(0)
        + out["EnclosedPorch"].fillna(0)
        + out["3SsnPorch"].fillna(0)
        + out["ScreenPorch"].fillna(0)
    )

    return out


def fit_neighborhood_medians(
    df: pd.DataFrame, target: str = "SalePrice"
) -> dict[str, float]:
    """
    Compute median SalePrice per neighbourhood on the train set.

    Must be called only on train data - applying to the full dataset
    would leak test information into training (data leakage).

    Returns a dict mapping neighbourhood name to median price,
    to be passed into engineer_extended_features().
    """
    return df.groupby("Neighborhood")[target].median().to_dict()


def engineer_extended_features(
    df: pd.DataFrame,
    neighborhood_medians: dict[str, float],
) -> pd.DataFrame:
    """
    Add 12 extended features to a DataFrame that already has base features.

    Requires engineer_base_features() to have been applied first.
    neighborhood_medians must be fitted on train only via
    fit_neighborhood_medians().

    Wave 1 - structure and composition:
        QualSF           - quality x area interaction
        QualAge          - quality relative to age
        HasBasement      - binary flag
        HasGarage        - binary flag
        BsmtRatio        - basement proportion of total area
        LotDensity       - build density (living area / lot area)
        NeighborhoodTier - target-encoded neighbourhood

    Wave 2 - low price range segment (<$100k):
        QualCond         - quality x condition interaction
        IsOldHouse       - binary flag for pre-1970 houses
        IsNormalSale     - binary flag for normal vs distressed sales
        FuncQual         - functional quality interaction
        TotalQualSF      - improved QualSF incorporating condition
    """
    out = df.copy()

    global_median = (
        float(np.median(list(neighborhood_medians.values())))
        if neighborhood_medians
        else 163_000.0
    )

    # Wave 1
    out["QualSF"]           = out["OverallQual"] * out["TotalSF"]
    out["QualAge"]          = out["OverallQual"] / (out["HouseAge"].clip(lower=0) + 1)
    out["HasBasement"]      = (out["TotalBsmtSF"] > 0).astype(int)
    out["HasGarage"]        = (out["GarageCars"] > 0).astype(int)
    out["BsmtRatio"]        = out["TotalBsmtSF"] / (out["TotalSF"] + 1)
    out["LotDensity"]       = out["GrLivArea"] / (out["LotArea"] + 1)
    out["NeighborhoodTier"] = (
        out["Neighborhood"].map(neighborhood_medians).fillna(global_median)
    )

    # Wave 2
    _func_map = {
        "Typ": 7, "Min1": 6, "Min2": 5, "Mod": 4,
        "Maj1": 3, "Maj2": 2, "Sev": 1, "Sal": 0,
    }
    out["QualCond"]     = out["OverallQual"] * out["OverallCond"]
    out["IsOldHouse"]   = (out["HouseAge"] > 50).astype(int)
    out["IsNormalSale"] = (out["SaleCondition"] == "Normal").astype(int)
    out["FuncQual"]     = (
        out["OverallQual"] * out["Functional"].map(_func_map).fillna(4)
    )
    out["TotalQualSF"]  = (
        (out["OverallQual"] + out["OverallCond"]) / 2 * out["TotalSF"]
    )

    return out
