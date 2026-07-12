# ReX-CoDA: Regression-based Flexible-Limit Compositional Data Imputation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

## Overview

**ReX-CoDA** is a model-agnostic iterative framework for imputing left-censored
geochemical compositional data. It combines machine-learning-based predictive
modelling with a Mills-ratio truncated-normal correction to produce
compositionally coherent imputations in additive log-ratio (ALR) space.

This repository accompanies the manuscript:

> Emenaha O, Başarır H, Ellefmo SL (2026) ReX-CoDA: A Compositionally Aware
> Imputation Framework for Censored Geochemical Data.
> *Mathematical Geosciences*.

---

## Repository Contents

| File | Description |
|---|---|
| `rexcoda_implementation.py` | Full research implementation used in the paper — RF and XGB engines, DL and AL threshold variants, all evaluation metrics and diagnostic plots |
| `rexcoda_tutorial.py` | Self-contained tutorial script with synthetic data; no external files required |
| `requirements.txt` | Python dependencies |

---

## Method Summary

ReX-CoDA operates as follows:

1. Censored observations are flagged and temporarily replaced with their detection limits
2. The full dataset is transformed to ALR space using a stable reference element
3. For each censored target element, the top 15 predictors are selected by absolute covariance computed on the uncensored subset
4. A regression model is trained on uncensored observations and used to predict the censored entries
5. A Mills-ratio truncated-normal correction adjusts predictions downward to remain below the censoring threshold
6. Steps 4 and 5 iterate until the Frobenius-norm convergence criterion is satisfied
7. The imputed ALR matrix is back-transformed to compositional ppm space

The framework supports both a strict **Detection Limit (DL)** and a relaxed
**Acceptable Limit (AL)** threshold, where AL = DL × (1 + τ) accounts for
measurement uncertainty near the detection limit.

---

## Installation

```bash
git clone https://github.com/ogonna-emenaha/ReX-CoDA.git
cd ReX-CoDA
pip install -r requirements.txt
```

---

## Quick Start

### Run the tutorial (no data file needed)

```bash
python rexcoda_tutorial.py
```

### Apply to your own data

```python
from rexcoda_tutorial import (load_and_prepare_data,
                               create_censoring_flags,
                               apply_alr_transformation,
                               rexcoda_impute)

# Step 1: Load data
df = load_and_prepare_data('your_geochemical_data.csv')

# Step 2: Define detection limits and flag censored observations
detection_limits = {'Ag': 0.002, 'Hg': 0.005, 'Ta': 0.05}
df = create_censoring_flags(df, detection_limits)

# Step 3: ALR transformation (Fe as reference element)
elements = ['Ag', 'Hg', 'Ta', 'Cu', 'Ni', 'Co']
df = apply_alr_transformation(df, elements, reference_element='Fe')

# Step 4: Run ReX-CoDA with detection limit threshold
result_dl = rexcoda_impute(df, element='Ag', detection_limit=0.002,
                            reference_element='Fe', threshold_type='DL')

# Step 5: Run ReX-CoDA with acceptable limit threshold (50% tolerance)
result_al = rexcoda_impute(df, element='Ag', detection_limit=0.002,
                            reference_element='Fe', threshold_type='AL',
                            tolerance=0.5)

# Step 6: Add imputed values back to the dataframe
df['ALR_Ag_imputed'] = result_dl['imputed_alr']
df['ppmAg_imputed']  = result_dl['imputed_ppm']
```

### Run the full research implementation

```bash
python rexcoda_implementation.py --data your_data.csv --output results/
```

---

## Supported Regression Engines

ReX-CoDA is model-agnostic. Any sklearn-compatible regressor can be used:

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import xgboost as xgb

# Random Forest (used in the paper)
rf_model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)

# XGBoost (used in the paper)
xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)

# Support Vector Regression
svr_model = SVR(kernel='rbf', C=1.0)

# Neural Network
nn_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500)

# Pass any of the above to rexcoda_impute()
result = rexcoda_impute(df, element='Ag', detection_limit=0.002,
                         model=rf_model, threshold_type='DL')
```

---

## Input Data Format

The input CSV should contain elemental concentrations in parts per million (ppm).
Column naming convention:

| Column | Description |
|---|---|
| `ppm{Element}` | Concentration in ppm (e.g. `ppmAg`, `ppmFe`) |
| `ppm{Element}_censored_flag` | Optional: 1 if censored, 0 if observed |

The reference element (default: Fe) must have no censored observations.

If your data does not already have censoring flags, use `create_censoring_flags()`
to generate them automatically from detection limits.

---

## Output

`rexcoda_impute()` returns a dictionary containing:

| Key | Description |
|---|---|
| `imputed_alr` | Full ALR array with imputed values inserted |
| `imputed_ppm` | Full ppm array with imputed values inserted |
| `imputed_censored_alr` | Imputed ALR values for censored entries only |
| `imputed_censored_ppm` | Imputed ppm values for censored entries only |
| `iterations` | Number of iterations until convergence |
| `convergence_history` | Delta at each iteration |
| `n_censored` | Number of censored observations |
| `model` | Fitted regression model object |

---

## Citation

If you use ReX-CoDA in your research, please cite:

```
Emenaha O, Başarır H, Ellefmo SL (2025) ReX-CoDA: A Compositionally Aware
Imputation Framework for Censored Geochemical Data.
Mathematical Geosciences.
```

---

## Authors

**Ogonna Emenaha** — ogonna.t.emenaha@ntnu.no  
**Hakan Başarır**  
**Steinar Løve Ellefmo**  

Department of Geosciences, Norwegian University of Science and Technology
(NTNU), Trondheim, Norway

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file
for details.
