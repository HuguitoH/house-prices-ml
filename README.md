# House Prices — Advanced Regression

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red)
![Plotly](https://img.shields.io/badge/Plotly-5.18-purple?logo=plotly&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-12%2C337%20RMSE-20BEFF?logo=kaggle&logoColor=white)

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
├── 01_eda_and_pipeline.ipynb           # EDA, preprocessing, 3-model benchmark
├── 02_feature_engineering_and_modelling.ipynb  # Feature engineering, GridSearch, final model
├── pipeline_utils.py                   # Shared functions — imputation, features, evaluation
├── requirements.txt
├── data/
│   ├── train.csv
│   └── test.csv
├── models/
│   ├── pipeline_state.pkl              # Exported state from notebook 01
│   ├── model_ridge.pkl                 # Ridge (alpha=10) — 20 KB
│   └── model_blend.pkl                 # Blend Ridge + XGBoost — 823 KB
└── submissions/
    ├── submission_ridge.csv            # Kaggle score: 12,883
    └── submission_blend.csv            # Kaggle score: 12,337
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

---

## Run Locally

```bash
git clone https://github.com/HuguitoH/house-prices-ml
cd house-prices-ml
pip install -r requirements.txt

# Run notebook 01 first - generates pipeline_state.pkl
# Then run notebook 02
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
