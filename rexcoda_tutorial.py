"""
================================================================================
ReX-CoDA Tutorial Script: Imputation of Censored Geochemical Data
================================================================================

ReX-CoDA: Regression-based Flexible-Limit Compositional Data Imputation

This tutorial demonstrates how to apply the ReX-CoDA framework to impute
left-censored geochemical compositional data using machine learning models
with Mills-ratio truncation correction for compositional coherence.

Authors: Ogonna Emenaha, Hakan Basarir, Steinar Love Ellefmo
Affiliation: Department of Geosciences, Norwegian University of Science
             and Technology (NTNU), Trondheim, Norway
Contact: ogonna.t.emenaha@ntnu.no

Reference: Emenaha O, Basarir H, Ellefmo SL (2025) ReX-CoDA: A
Compositionally Aware Imputation Framework for Censored Geochemical Data.
Mathematical Geosciences.

Requirements:
    numpy, pandas, scikit-learn, scipy
    xgboost (optional, for XGBoost examples)

Run:
    python rexcoda_tutorial.py

================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# SECTION 1: DATA PREPARATION
# ============================================================================

def load_and_prepare_data(filepath, sep=','):
    """
    Load geochemical data from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to CSV file containing ppm concentrations per element.
        Column names should follow the convention: ppm{Element}
        (e.g. ppmFe, ppmAg, ppmCu).
    sep : str
        CSV delimiter (default ','; use ';' for semicolon-separated files).

    Returns
    -------
    df : pandas.DataFrame
        Loaded dataset.
    """
    df = pd.read_csv(filepath, sep=sep)
    print(f"Data loaded: {df.shape[0]} samples, {df.shape[1]} features")
    return df


def create_censoring_flags(df, detection_limits):
    """
    Identify censored observations and replace them with the detection limit.

    Parameters
    ----------
    df : pandas.DataFrame
        Geochemical data with ppm columns.
    detection_limits : dict
        Mapping of element symbol to detection limit in ppm.
        Example: {'Ag': 0.002, 'Hg': 0.005, 'Ta': 0.05}

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with added binary censoring flag columns
        (ppm{Element}_censored_flag: 1 = censored, 0 = observed).
    """
    for element, dl in detection_limits.items():
        col  = f'ppm{element}'
        flag = f'{col}_censored_flag'

        if col not in df.columns:
            print(f"  Warning: column {col} not found. Skipping {element}.")
            continue

        df[flag] = (df[col] <= dl).astype(int)
        df.loc[df[flag] == 1, col] = dl

        n   = df[flag].sum()
        pct = n / len(df) * 100
        print(f"  {element}: {n} censored ({pct:.1f}%)")

    return df


def apply_alr_transformation(df, elements, reference_element='Fe'):
    """
    Apply the Additive Log-Ratio (ALR) transformation.

    Each element x_i is transformed as:
        ALR_i = log(ppm_i / ppm_ref)

    Parameters
    ----------
    df : pandas.DataFrame
        Data containing ppm columns.
    elements : list of str
        Elements to transform (e.g. ['Ag', 'Cu', 'Ni']).
    reference_element : str
        Reference element for the denominator (must have no censored values).

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with added ALR_{element} columns.
    """
    ref_col = f'ppm{reference_element}'

    if ref_col not in df.columns:
        raise ValueError(f"Reference element column '{ref_col}' not found.")

    for element in elements:
        ppm_col = f'ppm{element}'
        alr_col = f'ALR_{element}'

        if ppm_col not in df.columns:
            print(f"  Warning: {ppm_col} not found. Skipping.")
            continue

        df[alr_col] = np.log(df[ppm_col] / df[ref_col])
        print(f"  ALR transformed: {element}")

    return df


# ============================================================================
# SECTION 2: PREDICTOR SELECTION
# ============================================================================

def select_top_predictors(df, target_alr, censored_mask, n_predictors=15):
    """
    Select the top predictor elements by absolute covariance with the target.

    Covariance is computed using only the uncensored subset to avoid
    contaminating predictor rankings with imputed values.

    Parameters
    ----------
    df : pandas.DataFrame
        ALR-transformed data.
    target_alr : str
        Name of the target ALR column (e.g. 'ALR_Ag').
    censored_mask : pandas.Series or ndarray (bool)
        True where observations are censored.
    n_predictors : int
        Number of top predictors to select (default: 15).

    Returns
    -------
    predictor_cols : list of str
        Selected predictor column names.
    """
    alr_cols       = [c for c in df.columns
                      if c.startswith('ALR_') and c != target_alr]
    uncensored_data = df.loc[~censored_mask]

    covariances = {}
    for col in alr_cols:
        cov = uncensored_data[[target_alr, col]].cov().iloc[0, 1]
        covariances[col] = abs(cov)

    sorted_preds  = sorted(covariances.items(), key=lambda x: x[1], reverse=True)
    predictor_cols = [col for col, _ in sorted_preds[:n_predictors]]

    print(f"  Selected {len(predictor_cols)} predictors for {target_alr}")
    return predictor_cols


# ============================================================================
# SECTION 3: ReX-CoDA IMPUTATION ENGINE
# ============================================================================

def rexcoda_impute(df, element, detection_limit, reference_element='Fe',
                   model=None, threshold_type='DL', tolerance=0.5,
                   max_iter=50, convergence_tol=1e-5):
    """
    Core ReX-CoDA imputation function.

    Iteratively trains a regression model on the current complete dataset
    and updates censored estimates using a Mills-ratio truncated-normal
    correction until convergence.

    Parameters
    ----------
    df : pandas.DataFrame
        ALR-transformed data with censoring flag columns.
    element : str
        Element to impute (e.g. 'Ag').
    detection_limit : float
        Detection limit in ppm.
    reference_element : str
        Reference element used in ALR transformation (default: 'Fe').
    model : sklearn-compatible regressor, optional
        Regression model. Defaults to RandomForestRegressor.
    threshold_type : str
        'DL' for strict detection limit or 'AL' for acceptable limit.
    tolerance : float
        Tolerance factor for AL: AL = DL * (1 + tolerance). Default: 0.5.
    max_iter : int
        Maximum number of EM iterations (default: 50).
    convergence_tol : float
        Convergence threshold on mean absolute change in imputed values.

    Returns
    -------
    result : dict
        Keys include:
            'imputed_alr'         -- full ALR array with imputed values
            'imputed_ppm'         -- back-transformed ppm values
            'imputed_censored_alr'-- imputed values only for censored entries
            'imputed_censored_ppm'-- ppm values for censored entries
            'iterations'          -- number of iterations until convergence
            'convergence_history' -- delta at each iteration
            'n_censored'          -- number of censored observations
            'model'               -- fitted regression model
    """

    if model is None:
        model = RandomForestRegressor(
            n_estimators=100, max_depth=8, random_state=42, n_jobs=-1
        )

    alr_col  = f'ALR_{element}'
    ref_col  = f'ppm{reference_element}'
    flag_col = f'ppm{element}_censored_flag'

    for col in [alr_col, ref_col, flag_col]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe.")

    # Set threshold
    if threshold_type == 'AL':
        threshold_value = detection_limit * (1 + tolerance)
    else:
        threshold_value = detection_limit

    print(f"\nReX-CoDA | {element} | {threshold_type} = {threshold_value:.4f} ppm")

    censored_mask = df[flag_col] == 1
    n_censored    = censored_mask.sum()

    if n_censored == 0:
        print(f"  No censored observations found for {element}.")
        return None

    print(f"  Censored: {n_censored} | Uncensored: {(~censored_mask).sum()}")

    # Select predictors
    predictor_cols = select_top_predictors(df, alr_col, censored_mask)

    X               = df[predictor_cols].values
    y               = df[alr_col].values.copy()
    reference_value = df[ref_col].values

    # Train / test split on uncensored observations
    uncensored_idx = np.where(~censored_mask)[0]
    train_idx, test_idx = train_test_split(
        uncensored_idx, test_size=0.2, random_state=42
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Initialise censored values at 0.5 * threshold in ALR space
    y_imputed = y.copy()
    y_imputed[censored_mask] = np.log(
        0.5 * threshold_value / reference_value[censored_mask]
    )

    prev_imputed      = y_imputed.copy()
    convergence_deltas = []
    delta             = np.inf
    iteration         = 0

    while iteration < max_iter and delta > convergence_tol:
        iteration += 1
        m = clone(model)
        m.fit(X, y_imputed)

        mu = m.predict(X[censored_mask])

        # Residual standard deviation (global estimate)
        residuals = y_imputed - m.predict(X)
        sigma     = np.maximum(np.std(residuals), 0.01)
        sigma_arr = np.full_like(mu, sigma)

        # Transform threshold to ALR space
        threshold_alr = np.log(threshold_value / reference_value[censored_mask])

        # Mills-ratio truncated normal correction
        z       = (threshold_alr - mu) / sigma_arr
        phi     = norm.pdf(z)
        Phi     = norm.cdf(z)
        corrected = mu - sigma_arr * (phi / (Phi + 1e-10))

        # Clip to valid operational range
        lower_bound = np.log(1e-6 / reference_value[censored_mask])
        corrected   = np.clip(corrected, lower_bound, threshold_alr - 1e-10)

        y_imputed[censored_mask] = corrected
        delta = np.mean(np.abs(corrected - prev_imputed[censored_mask]))
        convergence_deltas.append(delta)
        prev_imputed = y_imputed.copy()

        if iteration <= 5 or iteration % 10 == 0:
            print(f"    Iteration {iteration:2d} | Delta: {delta:.6f}")

    print(f"  Converged after {iteration} iterations (delta: {delta:.6f})")

    # Back-transform to ppm
    ppm_imputed          = np.exp(y_imputed) * reference_value
    imputed_censored_ppm = ppm_imputed[censored_mask]

    result = {
        'element':             element,
        'threshold_type':      threshold_type,
        'threshold_value':     threshold_value,
        'n_censored':          int(n_censored),
        'iterations':          iteration,
        'convergence_history': convergence_deltas,
        'imputed_alr':         y_imputed,
        'imputed_ppm':         ppm_imputed,
        'imputed_censored_alr': y_imputed[censored_mask],
        'imputed_censored_ppm': imputed_censored_ppm,
        'min_imputed':         imputed_censored_ppm.min(),
        'max_imputed':         imputed_censored_ppm.max(),
        'mean_imputed':        imputed_censored_ppm.mean(),
        'model':               m,
    }

    print(f"  Imputed values — "
          f"min: {result['min_imputed']:.6f}, "
          f"max: {result['max_imputed']:.6f}, "
          f"mean: {result['mean_imputed']:.6f} ppm")

    return result


# ============================================================================
# SECTION 4: BATCH PROCESSING
# ============================================================================

def batch_impute_elements(df, element_configs, reference_element='Fe'):
    """
    Impute multiple elements using ReX-CoDA under both DL and AL thresholds.

    Parameters
    ----------
    df : pandas.DataFrame
        ALR-transformed data with censoring flags.
    element_configs : list of dict
        Each dict must contain:
            'element'         : str  (e.g. 'Ag')
            'detection_limit' : float (ppm)
            'model'           : sklearn regressor (optional)
    reference_element : str
        Reference element for ALR (default: 'Fe').

    Returns
    -------
    results : dict
        Keys are '{element}_DL' and '{element}_AL'.
        Values are result dicts from rexcoda_impute().
    """
    results = {}

    for config in element_configs:
        element = config['element']
        dl      = config['detection_limit']
        model   = config.get('model', None)

        print(f"\n{'=' * 60}")
        print(f"  Processing {element}")
        print(f"{'=' * 60}")

        for threshold_type in ['DL', 'AL']:
            result = rexcoda_impute(
                df=df,
                element=element,
                detection_limit=dl,
                reference_element=reference_element,
                model=model,
                threshold_type=threshold_type,
            )
            if result is not None:
                results[f"{element}_{threshold_type}"] = result

    return results


# ============================================================================
# SECTION 5: EXAMPLE WORKFLOW (runs on synthetic data — no file needed)
# ============================================================================

def example_workflow():
    """
    Demonstrate the full ReX-CoDA pipeline using synthetic geochemical data.

    No external data file is required. This function generates a small
    synthetic dataset and runs the complete imputation workflow so that
    any user can verify the installation and understand the API.
    """

    print("=" * 70)
    print("ReX-CoDA: Example Workflow (synthetic data)")
    print("=" * 70)

    # --- Generate synthetic data -------------------------------------------
    np.random.seed(42)
    n = 500

    df = pd.DataFrame({
        'ppmFe': np.random.lognormal(10.0, 0.5, n),
        'ppmAg': np.random.lognormal(-2.0, 1.0, n),
        'ppmCu': np.random.lognormal(2.0,  0.8, n),
        'ppmNi': np.random.lognormal(3.0,  0.7, n),
        'ppmCo': np.random.lognormal(1.0,  0.6, n),
        'ppmZn': np.random.lognormal(4.0,  0.9, n),
    })
    print(f"\nSynthetic dataset: {df.shape[0]} samples, {df.shape[1]} elements")

    # --- Step 1: Define detection limits and flag censored values -----------
    detection_limits = {'Ag': 0.002}
    df = create_censoring_flags(df, detection_limits)

    # --- Step 2: ALR transformation -----------------------------------------
    elements = ['Ag', 'Cu', 'Ni', 'Co', 'Zn']
    df = apply_alr_transformation(df, elements, reference_element='Fe')

    # --- Step 3: Impute with DL threshold ------------------------------------
    result_dl = rexcoda_impute(
        df=df,
        element='Ag',
        detection_limit=0.002,
        reference_element='Fe',
        threshold_type='DL',
    )

    # --- Step 4: Impute with AL threshold (50% tolerance) -------------------
    result_al = rexcoda_impute(
        df=df,
        element='Ag',
        detection_limit=0.002,
        reference_element='Fe',
        threshold_type='AL',
        tolerance=0.5,
    )

    # --- Step 5: Add imputed columns back to dataframe ----------------------
    if result_dl is not None:
        df['ALR_Ag_imputed_DL'] = result_dl['imputed_alr']
        df['ppmAg_imputed_DL']  = result_dl['imputed_ppm']
        print(f"\nDL imputation summary:")
        print(f"  Censored values imputed : {result_dl['n_censored']}")
        print(f"  Convergence iterations  : {result_dl['iterations']}")
        print(f"  Mean imputed value      : {result_dl['mean_imputed']:.6f} ppm")

    if result_al is not None:
        df['ALR_Ag_imputed_AL'] = result_al['imputed_alr']
        df['ppmAg_imputed_AL']  = result_al['imputed_ppm']
        print(f"\nAL imputation summary:")
        print(f"  Censored values imputed : {result_al['n_censored']}")
        print(f"  Convergence iterations  : {result_al['iterations']}")
        print(f"  Mean imputed value      : {result_al['mean_imputed']:.6f} ppm")

    print("\n" + "=" * 70)
    print("Example workflow completed successfully.")
    print("=" * 70)

    return df


# ============================================================================
# SECTION 6: USING XGBOOST
# ============================================================================

def example_with_xgboost():
    """
    Demonstrate ReX-CoDA with XGBoost as the regression engine.

    Any sklearn-compatible regressor can be passed as the 'model' argument
    to rexcoda_impute(). This example uses XGBoost.
    """
    try:
        import xgboost as xgb
    except ImportError:
        print("XGBoost not installed. Run: pip install xgboost")
        return

    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )

    print("\nXGBoost model configured. Pass it to rexcoda_impute() as:")
    print("  result = rexcoda_impute(df, element='Ag', detection_limit=0.002,")
    print("                          model=xgb_model, threshold_type='DL')")

    return xgb_model


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    df_imputed = example_workflow()

    print("\nTo apply to your own data:")
    print("  df = load_and_prepare_data('your_geochemical_data.csv')")
    print("  df = create_censoring_flags(df, {'Ag': 0.002, 'Hg': 0.005})")
    print("  df = apply_alr_transformation(df, ['Ag', 'Hg'], 'Fe')")
    print("  result = rexcoda_impute(df, 'Ag', 0.002)")
    print("\nFor multiple elements use batch_impute_elements().")
    print("For the full research implementation see rexcoda_implementation.py.")
