# House Prices — Advanced Regression

End-to-end ML project predicting residential property sale prices (Ames, Iowa dataset).
Built across two notebooks covering the full pipeline from raw data to Kaggle submission.

**Kaggle Score: $12,481 RMSE** (top leaderboard — Home Data for ML Course competition)

---

## Results

| Model                             | R² Test | RMSE Test   | CV RMSE | Overfitting Gap |
| --------------------------------- | ------- | ----------- | ------- | --------------- |
| **Blend Ridge 50% + XGBoost 50%** | —       | **$12,481** | —       | —               |
| XGBoost E2 baseline               | 0.9392  | $18,326     | —       | 0.041           |
| Ridge (GridSearch)                | 0.9306  | $19,585     | 0.1118  | 0.021           |
| XGBoost (GridSearch)              | 0.9301  | $19,652     | 0.1150  | 0.043           |
| Random Forest (GridSearch)        | 0.9137  | $21,833     | 0.1273  | 0.064           |
| SVR (GridSearch)                  | 0.8152  | $31,948     | 0.4903  | —               |

The final model is a 50/50 blend of Ridge (alpha=10) and XGBoost, selected after systematic
GridSearch and variance analysis. Ridge was chosen as the production-safe model due to minimal
overfitting gap (0.021) and maximum interpretability.

---

## Project Structure

```
house-prices-ml/
├── E2_house_prices.ipynb        # EDA, data preparation, sklearn Pipeline, benchmark
├── E3_house_prices.ipynb        # Feature engineering, GridSearch, SHAP, final model
├── data/
│   ├── train.csv
│   └── test.csv
├── models/
│   ├── model_final_ridge_e3.pkl    # Ridge (alpha=10) — 20 KB
│   └── model_final_blend_e3.pkl    # Blend Ridge+XGBoost — 823 KB
└── requirements.txt
```

---

## Notebook E2 — EDA & Pipeline

**Goal:** Understand the dataset and build a preprocessing pipeline before any modelling.

Key decisions documented:

- **NaN disambiguation** — distinguishing `NaN = missing value` (e.g. `LotFrontage` 17.7%) from
  `NaN = feature absent` (e.g. `PoolQC` 99.5% — the house simply has no pool). Imputing the
  latter with mean/mode would introduce systematic noise.
- **log1p target transformation** — `SalePrice` skewness ~1.88 reduced to ~0.12 after
  transformation, improving residual normality for linear models. Predictions reverted with
  `np.expm1()`.
- **sklearn ColumnTransformer Pipeline** — separate preprocessing paths for numerical
  (StandardScaler + SimpleImputer) and categorical (OrdinalEncoder / OneHotEncoder) features,
  avoiding data leakage.
- **Benchmark** — LinearRegression / DecisionTree / XGBoost comparison on identical splits
  before any tuning.

---

## Notebook E3 — Feature Engineering, GridSearch & SHAP

**Goal:** Push performance beyond the E2 baseline through systematic feature engineering and
hyperparameter optimisation.

### Feature Engineering (12 new features)

Each feature justified by domain reasoning, not correlation fishing:

| Feature                 | Rationale                                                              |
| ----------------------- | ---------------------------------------------------------------------- |
| `TotalSF`               | Total living area (basement + 1F + 2F) — single strongest price signal |
| `TotalBathrooms`        | Combined full + half baths weighted (full=1, half=0.5)                 |
| `HouseAge`              | Years between build and sale — depreciation proxy                      |
| `RemodAge`              | Years since last remodel — renovation signal                           |
| `IsRemodeled`           | Binary flag — remodeled vs original                                    |
| `GarageAge`             | Garage build year relative to sale                                     |
| `TotalPorchSF`          | Sum of all porch areas — outdoor space signal                          |
| `OverallScore`          | Multiplicative quality × condition index                               |
| `LivingAreaRatio`       | Above-ground area / total SF — living space efficiency                 |
| `PricePerRoom`          | GrLivArea / TotRmsAbvGrd — room density                                |
| `NeighborhoodPriceRank` | Median SalePrice rank per neighbourhood (target encoding)              |
| `QualityPerAge`         | Overall quality adjusted by age                                        |

Pipeline extended to 82 input features → **237 features after OHE**.

### Model Selection

GridSearch with 5-fold CV across Ridge, XGBoost, LightGBM, Random Forest, SVR.
Selection criteria: CV RMSE + overfitting gap, not raw test score.

Ridge (alpha=10) selected as final single model — lowest CV RMSE (0.1118), minimal
overfitting (gap 0.021), full interpretability.
Blend (Ridge 50% + XGBoost 50%) selected for Kaggle submission — best generalisation score.

### Ablation Study

Quantified impact of feature engineering by comparing E2 baseline XGBoost ($18,326 RMSE)
against E3 pipeline — **31.9% RMSE reduction** driven primarily by `TotalSF`, `OverallScore`,
and `NeighborhoodPriceRank`.

### SHAP Analysis

Global feature importance via SHAP values — `TotalSF`, `OverallQual`, `NeighborhoodPriceRank`
and `GrLivArea` consistently top contributors across folds.

### Learning & Validation Curves

Diagnosed model behaviour at different training set sizes. Identified that SVR's high variance
in CV RMSE (0.49) stems from sensitivity to feature scaling rather than data volume.

---

## Stack

- Python 3.11
- scikit-learn — Pipeline, ColumnTransformer, GridSearchCV
- XGBoost, LightGBM, Ridge, Random Forest, SVR
- SHAP
- pandas, numpy, matplotlib, seaborn

---

## Run locally

```bash
git clone https://github.com/HuguitoH/house-prices-ml
cd house-prices-ml
pip install -r requirements.txt
# Open E2_house_prices.ipynb first, then E3_house_prices.ipynb
```

Data is downloaded automatically via `kagglehub` on first run (Kaggle credentials required).
