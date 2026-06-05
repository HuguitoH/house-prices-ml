"""
Model evaluation utilities.

compute_metrics() is a pure function that computes regression metrics
on dollar-scale predictions.

evaluate_model() fits a model on train, evaluates on both splits,
and returns a flat dict with metrics and raw arrays for diagnostic plots.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute regression metrics on dollar-scale predictions.

    Returns:
        r2   - R squared
        mae  - Mean Absolute Error
        rmse - Root Mean Squared Error
        mape - Mean Absolute Percentage Error (%)
    """
    return {
        "r2":   round(r2_score(y_true, y_pred), 4),
        "mae":  round(mean_absolute_error(y_true, y_pred), 0),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 0),
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

    Predictions are reverted from log-scale to dollar scale before
    computing metrics - ensures all reported errors are interpretable.

    Returns a flat dict with train/test metrics and raw arrays
    for diagnostic plots (prefixed with underscore).
    """
    model.fit(X_train, y_train)

    y_true_train = np.expm1(y_train.values)
    y_true_test  = np.expm1(y_test.values)
    y_pred_train = np.expm1(model.predict(X_train))
    y_pred_test  = np.expm1(model.predict(X_test))

    train = compute_metrics(y_true_train, y_pred_train)
    test  = compute_metrics(y_true_test,  y_pred_test)

    print(
        f"{name:<35} "
        f"train_r2={train['r2']:.4f} "
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
