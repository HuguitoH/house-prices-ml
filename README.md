# House Prices — Advanced Regression

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-orange?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-red)
![Plotly](https://img.shields.io/badge/Plotly-6.8-purple?logo=plotly&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-12%2C337%20RMSE-20BEFF?logo=kaggle&logoColor=white)
![CI](https://github.com/HuguitoH/house-prices-ml/actions/workflows/ci.yml/badge.svg)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live-FF4B4B?logo=streamlit&logoColor=white)](https://house-prices-ml-kaggle.streamlit.app/)

End-to-end ML project predicting residential property sale prices (Ames, Iowa).
Built across two notebooks with a shared utility module, covering the full pipeline
from raw data exploration to a serialised production-ready model.

**Kaggle Score: $12,337 RMSE** — Home Data for ML Course competition

---

## Results

| Model                             | R2 Test | RMSE Test | CV RMSE | Overfit Gap |
| --------------------------------- | ------- | --------- | ------- | ----------- |
| **Blend Ridge 50% + XGBoost 50%** | 0.9371  | $18,647   | -       | -           |
| XGBoost (baseline)                | 0.9392  | $18,326   | -       | 0.041       |
| Ridge (GridSearch)                | 0.9306  | $19,585   | 0.1118  | 0.021       |
| XGBoost (GridSearch)              | 0.9301  | $19,652   | 0.1150  | 0.043       |
| Random Forest (GridSearch)        | 0.9137  | $21,833   | 0.1273  | 0.064       |
| SVR (GridSearch)                  | 0.8152  | $31,948   | 0.4903  | -           |

Ridge (alpha=10) selected as production model: lowest CV RMSE (0.1118), minimal
overfitting gap (0.021), full interpretability via coefficients.
Blend selected for Kaggle submission: complementary errors between Ridge (linear)
and XGBoost (non-linear) reduce generalisation error.

> [!Note]
> RMSE Test is on the local 80/20 validation split. Kaggle score reflects generalisation on the full held-out test set. The blend achieves $12,337 on Kaggle despite $18,647 on local validation - confirming that CV RMSE is a more reliable estimate than any single split.

---

## Project Structure

```
house-prices-ml/
├── notebooks/
│   ├── 01_eda_and_pipeline.ipynb
│   └── 02_feature_engineering_and_modelling.ipynb
├── src/
│   ├── registry.py
│   ├── features.py
│   ├── pipeline.py
│   └── evaluate.py
├── tests/
│   ├── test_features.py
│   ├── test_registry.py
│   ├── test_pipeline.py
│   └── test_evaluate.py
├── pipeline_utils.py
├── models/
│   ├── pipeline_state.pkl
│   ├── model_ridge.pkl
│   └── model_blend.pkl
├── pyproject.toml
└── README.md
```

### Why this structure?

**`pipeline_utils.py`** contains all shared logic — imputation, feature engineering,
model evaluation, pipeline builder. Both notebooks import from it, eliminating
code duplication and making every function independently testable.

**`pipeline_state.pkl`** exports the full preprocessing state from notebook 01 so
notebook 02 does not reload raw data or redefine functions. This mirrors how a
production system would consume a trained preprocessor.

**Two notebooks, one direction** — notebook 01 explores and establishes the baseline.
Notebook 02 imports that baseline and extends it. Running them out of order or in
isolation would fail — intentionally, because that reflects real pipeline dependencies.

**`src/`** contains the production modules — `registry.py`, `features.py`,
`pipeline.py`, `evaluate.py`. Each module has a single responsibility and is
independently testable. `pipeline_utils.py` re-exports everything from `src/`
for backwards compatibility with the notebooks.

**`tests/`** — 42 unit tests with 100% coverage on `src/`. Pure unit tests
with synthetic fixtures, no dependency on real CSV data.

---

## Notebook 01 — EDA, Preprocessing & Benchmark

**Goal:** understand the dataset and build a reproducible preprocessing pipeline
before any modelling decisions.

**Key decisions:**

**NaN disambiguation** - `LotFrontage` (17.7% missing) is truly missing data,
imputed with neighbourhood median. `PoolQC` (99.5% missing) means the house has
no pool - filled with `"None"`. Imputing the latter with mean/mode would tell the
model that nearly every house has a high-quality pool.

**log1p target transformation** - `SalePrice` skewness ~1.88 reduced to ~0.12,
improving residual normality for linear models. Predictions reverted with `np.expm1()`.

**TransformationRegistry** - a stateful object that tracks the assigned transformation
for every column. Replaces the global-state pattern and makes the pipeline declaration
auditable and testable.

**sklearn ColumnTransformer Pipeline** - separate preprocessing paths per transformation
type (StandardScaler, RobustScaler, OrdinalEncoder, OneHotEncoder), all fitted
exclusively on train data to prevent data leakage.

**Benchmark** - LinearRegression / DecisionTree / XGBoost on identical splits before
any tuning. XGBoost achieves R2=0.9392, RMSE=$18,326 — the target for notebook 02.

---

## Notebook 02 — Feature Engineering, GridSearch & Model Selection

**Goal:** push performance beyond the baseline through systematic feature engineering
and hyperparameter optimisation.

### Feature Engineering (12 new features)

Each feature justified by domain reasoning, not correlation fishing:

| Feature            | Formula                                     | Rationale                                                                |
| ------------------ | ------------------------------------------- | ------------------------------------------------------------------------ |
| `QualSF`           | `OverallQual x TotalSF`                     | Quality-area interaction - high quality area is worth exponentially more |
| `QualAge`          | `OverallQual / (HouseAge + 1)`              | Quality relative to age                                                  |
| `HasBasement`      | `(TotalBsmtSF > 0).astype(int)`             | Structural binary flag                                                   |
| `HasGarage`        | `(GarageCars > 0).astype(int)`              | Presence/absence as its own category                                     |
| `BsmtRatio`        | `TotalBsmtSF / (TotalSF + 1)`               | Basement proportion relative to total area                               |
| `LotDensity`       | `GrLivArea / (LotArea + 1)`                 | Build density signal                                                     |
| `NeighborhoodTier` | Median SalePrice per neighbourhood          | Target encoding - fitted on train only                                   |
| `QualCond`         | `OverallQual x OverallCond`                 | Low quality AND low condition interaction                                |
| `IsOldHouse`       | `(HouseAge > 50).astype(int)`               | Pre-1970 price profile                                                   |
| `IsNormalSale`     | `(SaleCondition == 'Normal').astype(int)`   | Distressed sale flag                                                     |
| `FuncQual`         | `OverallQual x func_score`                  | Functional issues combined with quality                                  |
| `TotalQualSF`      | `(OverallQual + OverallCond) / 2 x TotalSF` | Improved QualSF with condition                                           |

Pipeline: 82 input features - 237 after OHE.

### Ablation Study

RMSE improvement from feature engineering quantified on identical splits:
baseline pipeline vs extended pipeline with Ridge (alpha=10).
Feature engineering accounts for the majority of the 716-point total improvement.

### Model Selection

GridSearch with 5-fold CV across Ridge, XGBoost, Random Forest, SVR.
Selection criteria: CV RMSE + overfitting gap - not raw test score.

### Learning & Validation Curves

Validation curve confirms `alpha=10` is the global optimum.
Learning curve is fully flattened at maximum training size - the model has
reached the dataset's information ceiling. The systematic -22% error on
houses below $100k cannot be fixed with more features or tuning: only more
data in that price range would help.

### Kaggle Submission Progression

| Submission | Model                             | Kaggle Score |
| ---------- | --------------------------------- | ------------ |
| S1         | XGBoost baseline                  | 13,053       |
| S2         | Ridge (GridSearch)                | 12,883       |
| S3         | **Blend Ridge 50% + XGBoost 50%** | **12,337**   |

Total improvement: 716 points (5.5%)

---

## Stack

- Python 3.11
- scikit-learn - Pipeline, ColumnTransformer, GridSearchCV
- XGBoost, Ridge, Random Forest, SVR
- Plotly - interactive visualisations
- pandas, numpy, scipy
- kagglehub - automatic dataset download
- pytest + pytest-cov - 42 unit tests, 100% coverage

---

## Run Locally

```bash
git clone https://github.com/HuguitoH/house-prices-ml
cd house-prices-ml
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Run tests
pytest

# Run notebooks in order
# notebooks/01_eda_and_pipeline.ipynb
# notebooks/02_feature_engineering_and_modelling.ipynb
```

> [!TIP]
> Run notebook 01 completely before opening notebook 02. Notebook 02 loads
> `models/pipeline_state.pkl` generated by notebook 01 - it will fail if
> that file does not exist.

Kaggle credentials required for automatic data download via `kagglehub`.
Alternatively, download manually from the
[competition page](https://www.kaggle.com/competitions/home-data-for-ml-course)
and place `train.csv` and `test.csv` in `data/`.

---

## Known Limitations

> [!WARNING]
> This model was trained on Ames, Iowa housing data from 2006-2010. It is not
> generalisable to other markets, geographies, or time periods. Do not use for
> real estate valuation outside this context.
