import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

from src.evaluate import compute_metrics, evaluate_model


# Fixtures

@pytest.fixture
def perfect_predictions() -> tuple[np.ndarray, np.ndarray]:
    y_true = np.array([100_000, 150_000, 200_000, 250_000], dtype=float)
    return y_true, y_true.copy()


@pytest.fixture
def linear_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    np.random.seed(42)
    n    = 100
    X    = pd.DataFrame({"feature": np.linspace(0, 1, n)})
    y    = pd.Series(np.log1p(X["feature"] * 200_000 + 100_000))
    split = int(n * 0.8)
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


# compute_metrics

class TestComputeMetrics:

    def test_perfect_predictions_give_r2_one_and_zero_errors(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["r2"]   == pytest.approx(1.0)
        assert metrics["rmse"] == pytest.approx(0.0)
        assert metrics["mae"]  == pytest.approx(0.0)
        assert metrics["mape"] == pytest.approx(0.0)

    def test_returns_all_required_keys(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        assert set(compute_metrics(y_true, y_pred).keys()) == {"r2", "mae", "rmse", "mape"}


# evaluate_model

class TestEvaluateModel:

    def test_returns_all_required_keys(self, linear_data):
        X_train, X_test, y_train, y_test = linear_data
        result = evaluate_model(LinearRegression(), X_train, X_test, y_train, y_test, "LR")
        required = {
            "model", "r2_train", "r2_test", "mae_test",
            "rmse_test", "mape_test", "overfit_gap",
            "_model", "_y_true", "_y_pred", "_residuals",
        }
        assert required.issubset(set(result.keys()))

    def test_overfit_gap_is_train_minus_test(self, linear_data):
        X_train, X_test, y_train, y_test = linear_data
        result = evaluate_model(LinearRegression(), X_train, X_test, y_train, y_test, "LR")
        assert result["overfit_gap"] == pytest.approx(
            round(result["r2_train"] - result["r2_test"], 4)
        )

    def test_predictions_in_dollar_scale(self, linear_data):
        X_train, X_test, y_train, y_test = linear_data
        result = evaluate_model(LinearRegression(), X_train, X_test, y_train, y_test, "LR")
        # y is log1p(100k-300k) - expm1 should give dollar-scale values
        assert result["_y_true"].mean() > 1_000
        assert result["_y_pred"].mean() > 1_000

    def test_residuals_equal_true_minus_pred(self, linear_data):
        X_train, X_test, y_train, y_test = linear_data
        result = evaluate_model(LinearRegression(), X_train, X_test, y_train, y_test, "LR")
        np.testing.assert_array_almost_equal(
            result["_residuals"],
            result["_y_true"] - result["_y_pred"],
        )

    def test_model_is_fitted_after_call(self, linear_data):
        X_train, X_test, y_train, y_test = linear_data
        model = LinearRegression()
        evaluate_model(model, X_train, X_test, y_train, y_test, "LR")
        model.predict(X_test)  # raises if not fitted
