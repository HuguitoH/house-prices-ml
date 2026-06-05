from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OrdinalEncoder, OneHotEncoder,
    RobustScaler, StandardScaler,
)
from sklearn.metrics import (
    r2_score, mean_absolute_error,
    mean_squared_error, mean_absolute_percentage_error,
)


# Constants
class TransformationRegistry:
    """
    Encapsulates the transformation assignment state for all dataset columns.
    Single source of truth for what transformation each column receives.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = self._build(df)

    def _build(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "column":         df.columns,
            "dtype":          df.dtypes.values,
            "is_target":      df.columns.isin([]),
            "n_unique":       [df[c].nunique() for c in df.columns],
            "pct_null":       [round(100 * df[c].isnull().mean(), 2) for c in df.columns],
            "transformation": None,
            "note":           None,
        })

    def set(self, column: str, transformation: str, note: str) -> None:
        """Assign a transformation and note to a single column."""
        mask = self._df["column"] == column
        self._df.loc[mask, "transformation"] = transformation
        self._df.loc[mask, "note"]           = note

    def set_many(self, columns: list[str], transformation: str, note: str) -> None:
        """Assign the same transformation to multiple columns at once."""
        for col in columns:
            self.set(col, transformation, note)

    def sync(self, df: pd.DataFrame) -> None:
        """
        Reconcile registry with current DataFrame columns.
        Removes dropped columns, adds new ones with no transformation assigned.
        """
        self._df = self._df[self._df["column"].isin(df.columns)].copy()
        new_cols = [c for c in df.columns if c not in self._df["column"].values]
        if new_cols:
            new_rows = pd.DataFrame({
                "column":         new_cols,
                "dtype":          [df[c].dtype for c in new_cols],
                "is_target":      [False] * len(new_cols),
                "n_unique":       [df[c].nunique() for c in new_cols],
                "pct_null":       [round(100 * df[c].isnull().mean(), 2) for c in new_cols],
                "transformation": None,
                "note":           None,
            })
            self._df = pd.concat([self._df, new_rows], ignore_index=True)

    def get_columns(self, transformation: str) -> list[str]:
        """Return all columns assigned to a given transformation."""
        return self._df.loc[
            self._df["transformation"] == transformation, "column"
        ].tolist()

    def unassigned(self) -> pd.DataFrame:
        """Return columns with no transformation assigned."""
        return self._df[
            self._df["transformation"].isna() & ~self._df["is_target"]
        ]

    @property
    def data(self) -> pd.DataFrame:
        """Read-only access to the underlying DataFrame."""
        return self._df.copy()

CAT_ABSENT = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
    "BsmtFinType2", "MasVnrType",
]

NUM_ABSENT = [
    "GarageYrBlt", "GarageArea", "GarageCars",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath", "MasVnrArea",
]

MODE_IMPUTE = [
    "Electrical", "MSZoning", "Exterior1st", "Exterior2nd",
    "KitchenQual", "SaleType", "Functional",
]


# Imputation

def impute_semantic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply domain-driven imputation before the sklearn Pipeline.
    Must be called on both train and test before any transformation.
    """
    out = df.copy()
    for col in CAT_ABSENT:
        if col in out.columns:
            out[col] = out[col].fillna("None")
    for col in NUM_ABSENT:
        if col in out.columns:
            out[col] = out[col].fillna(0)
    if "LotFrontage" in out.columns and "Neighborhood" in out.columns:
        out["LotFrontage"] = out.groupby("Neighborhood")["LotFrontage"].transform(
            lambda x: x.fillna(x.median())
        )
    for col in MODE_IMPUTE:
        if col in out.columns:
            out[col] = out[col].fillna(out[col].mode()[0])
    return out


# Feature engineering

def engineer_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create E1 base features from raw housing columns.
    Applied identically to train and test.
    """
    out = df.copy()
    out["TotalSF"]      = out["TotalBsmtSF"].fillna(0) + out["1stFlrSF"].fillna(0) + out["2ndFlrSF"].fillna(0)
    out["HouseAge"]     = out["YrSold"] - out["YearBuilt"]
    out["RemodAge"]     = out["YrSold"] - out["YearRemodAdd"]
    out["WasRemodeled"] = (out["YearRemodAdd"] != out["YearBuilt"]).astype(int)
    out["TotalBaths"]   = (
        out["FullBath"].fillna(0) + out["BsmtFullBath"].fillna(0)
        + 0.5 * (out["HalfBath"].fillna(0) + out["BsmtHalfBath"].fillna(0))
    )
    out["TotalPorchSF"] = (
        out["OpenPorchSF"].fillna(0) + out["EnclosedPorch"].fillna(0)
        + out["3SsnPorch"].fillna(0) + out["ScreenPorch"].fillna(0)
    )
    return out


def fit_neighborhood_medians(df: pd.DataFrame, target: str = "SalePrice") -> dict[str, float]:
    """
    Compute median SalePrice per neighbourhood on train set.
    Must be called only on train - never on test (data leakage).
    """
    return df.groupby("Neighborhood")[target].median().to_dict()


def engineer_extended_features(
    df: pd.DataFrame,
    neighborhood_medians: dict[str, float],
) -> pd.DataFrame:
    """
    Add 12 E3 features to a DataFrame that already has base features.
    neighborhood_medians must be fitted on train only.
    """
    out = df.copy()

    global_median = float(np.median(list(neighborhood_medians.values()))) if neighborhood_medians else 163_000.0

    # Wave 1 - structure and composition
    out["QualSF"]          = out["OverallQual"] * out["TotalSF"]
    out["QualAge"]         = out["OverallQual"] / (out["HouseAge"].clip(lower=0) + 1)
    out["HasBasement"]     = (out["TotalBsmtSF"] > 0).astype(int)
    out["HasGarage"]       = (out["GarageCars"] > 0).astype(int)
    out["BsmtRatio"]       = out["TotalBsmtSF"] / (out["TotalSF"] + 1)
    out["LotDensity"]      = out["GrLivArea"] / (out["LotArea"] + 1)
    out["NeighborhoodTier"]= out["Neighborhood"].map(neighborhood_medians).fillna(global_median)

    # Wave 2 - low price range segment (<$100k)
    func_map = {"Typ": 7, "Min1": 6, "Min2": 5, "Mod": 4, "Maj1": 3, "Maj2": 2, "Sev": 1, "Sal": 0}
    out["QualCond"]     = out["OverallQual"] * out["OverallCond"]
    out["IsOldHouse"]   = (out["HouseAge"] > 50).astype(int)
    out["IsNormalSale"] = (out["SaleCondition"] == "Normal").astype(int)
    out["FuncQual"]     = out["OverallQual"] * out["Functional"].map(func_map).fillna(4)
    out["TotalQualSF"]  = ((out["OverallQual"] + out["OverallCond"]) / 2) * out["TotalSF"]

    return out


# Evaluation

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics on dollar-scale predictions."""
    return {
        "r2":   round(r2_score(y_true, y_pred), 4),
        "mae":  round(mean_absolute_error(y_true, y_pred), 0),
        "rmse": round(np.sqrt(mean_squared_error(y_true, y_pred)), 0),
        "mape": round(mean_absolute_percentage_error(y_true, y_pred) * 100, 2),
    }


def evaluate_model(
    model,
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    y_train: pd.Series,
    y_test:  pd.Series,
    name:    str,
) -> dict:
    """
    Fit model on train, evaluate on both splits.
    Reverts log-scale predictions to dollar scale before computing metrics.
    """
    model.fit(X_train, y_train)

    y_true_train = np.expm1(y_train.values)
    y_true_test  = np.expm1(y_test.values)
    y_pred_train = np.expm1(model.predict(X_train))
    y_pred_test  = np.expm1(model.predict(X_test))

    train = compute_metrics(y_true_train, y_pred_train)
    test  = compute_metrics(y_true_test,  y_pred_test)

    print(
        f"{name:<35} train_r2={train['r2']:.4f} "
        f"test_r2={test['r2']:.4f} "
        f"test_rmse=${test['rmse']:,.0f}"
    )

    return {
        "model":       name,
        "r2_train":    train["r2"],
        "r2_test":     test["r2"],
        "mae_test":    test["mae"],
        "rmse_test":   test["rmse"],
        "mape_test":   test["mape"],
        "overfit_gap": round(train["r2"] - test["r2"], 4),
        "_model":      model,
        "_y_true":     y_true_test,
        "_y_pred":     y_pred_test,
        "_residuals":  y_true_test - y_pred_test,
    }


# Pipeline builder

def build_column_transformer(registry: "TransformationRegistry") -> ColumnTransformer:
    """
    Build a ColumnTransformer from a TransformationRegistry.
    remainder='drop' ensures unregistered columns never silently pass through.
    """
    def sub_pipeline(imputer: SimpleImputer, transformer=None) -> Pipeline:
        steps = [("imputer", imputer)]
        if transformer is not None:
            steps.append(("transformer", transformer))
        return Pipeline(steps)

    mapping = [
        ("passthrough",
         sub_pipeline(SimpleImputer(strategy="median")),
         registry.get_columns("passthrough")),

        ("std_scaler",
         sub_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
         registry.get_columns("standard_scaler")),

        ("rob_scaler",
         sub_pipeline(SimpleImputer(strategy="median"), RobustScaler()),
         registry.get_columns("robust_scaler")),

        ("ordinal",
         sub_pipeline(
             SimpleImputer(strategy="most_frequent"),
             OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
         ),
         registry.get_columns("ordinal_encoder")),

        ("ohe",
         sub_pipeline(
             SimpleImputer(strategy="most_frequent"),
             OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         ),
         registry.get_columns("one_hot_encoding")),
    ]

    transformers = [(name, pipe, cols) for name, pipe, cols in mapping if cols]
    return ColumnTransformer(transformers=transformers, remainder="drop")
