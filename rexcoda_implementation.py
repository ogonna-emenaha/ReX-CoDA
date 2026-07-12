"""
ReX-CoDA: Regression-based Flexible-Limit Compositional Data Imputation
========================================================================
Full Implementation Script

Authors: Ogonna Emenaha, Hakan Basarir, Steinar Love Ellefmo
Affiliation: Department of Geosciences, Norwegian University of Science
             and Technology (NTNU), Trondheim, Norway
Contact: ogonna.t.emenaha@ntnu.no

This script implements the complete ReX-CoDA framework for imputing
left-censored geochemical compositional data, including comparison
between Detection Limit (DL) and Acceptable Limit (AL) thresholds.

Usage:
    python rexcoda_implementation.py --data your_data.csv --output results/

Input data format:
    CSV file with columns:
        - ppm{Element}           : concentration in ppm (e.g. ppmAg, ppmFe)
        - ppm{Element}_censored_flag : 1 if censored, 0 if observed

Reference: Emenaha O, Basarir H, Ellefmo SL (2025) ReX-CoDA: A
Compositionally Aware Imputation Framework for Censored Geochemical Data.
Mathematical Geosciences.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.stats import norm, gaussian_kde
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import os
import argparse
import warnings

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

# Element detection limits and reference element
ELEMENT_CONFIG = {
    "Ag": {"dl_value": 0.002, "reference_col": "ppmFe", "censored_flag": "ppmAg_censored_flag"},
    "Hg": {"dl_value": 0.005, "reference_col": "ppmFe", "censored_flag": "ppmHg_censored_flag"},
    "Ta": {"dl_value": 0.05,  "reference_col": "ppmFe", "censored_flag": "ppmTa_censored_flag"},
}

# Acceptable Limit tolerance factor (0.5 = 50% above DL)
AL_TOLERANCE = 0.5

# Model configurations matching the paper
RF_MODEL  = RandomForestRegressor(n_estimators=100, max_depth=8,
                                   random_state=42, n_jobs=-1)
XGB_MODEL = xgb.XGBRegressor(n_estimators=100, max_depth=4,
                               learning_rate=0.1, verbosity=0,
                               random_state=42, n_jobs=1)


# =============================================================================
# SETUP
# =============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="ReX-CoDA: Censored Geochemical Data Imputation"
    )
    parser.add_argument(
        "--data", type=str, default="data/ALR_transformed_data.csv",
        help="Path to input CSV file (semicolon-separated)"
    )
    parser.add_argument(
        "--output", type=str, default="results/",
        help="Path to output folder for results and plots"
    )
    return parser.parse_args()


def create_output_folder(path):
    """Create output folder if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created output folder: {path}")
    return path


def load_data(data_path):
    """Load and validate input dataset."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    df = pd.read_csv(data_path, sep=';')
    df = df.iloc[:-1].copy()
    print(f"Data loaded. Shape: {df.shape}")
    return df


# =============================================================================
# CORE IMPUTATION ENGINE
# =============================================================================

def run_rexcoda_imputation(model, model_name, element, threshold_type,
                            X, y, X_train, X_test, y_train, y_test,
                            censored_mask, threshold_value, reference_value):
    """
    ReX-CoDA iterative imputation with Mills-ratio truncation correction.

    Parameters
    ----------
    model : sklearn-compatible regressor
    model_name : str
        'RF' or 'XGB'
    element : str
        Element being imputed (e.g. 'Ag')
    threshold_type : str
        'DL' for detection limit or 'AL' for acceptable limit
    X : ndarray
        Full feature matrix in ALR space
    y : ndarray
        Target variable (ALR-transformed target element)
    X_train, X_test : ndarray
        Train/test splits of uncensored observations
    y_train, y_test : ndarray
        Train/test targets
    censored_mask : boolean ndarray
        True where observations are censored
    threshold_value : float
        Detection or acceptable limit in ppm
    reference_value : Series
        Reference element concentrations (ppm) for ALR back-transformation

    Returns
    -------
    key : str
        Result identifier string
    result : dict
        Imputation outputs, model metrics, and metadata
    """

    print(f"\nReX-CoDA | {element} | {model_name} | {threshold_type} = {threshold_value}")

    if censored_mask.sum() == 0:
        print(f"  No censored values found for {element}. Skipping.")
        return None, None

    # Initialise censored values at 0.5 * DL in ALR space
    y_imputed = y.copy()
    y_imputed[censored_mask] = np.log(
        0.5 * threshold_value / reference_value[censored_mask].values
    )

    prev_imputed = y_imputed.copy()
    max_iter = 50
    tol = 1e-5
    delta = np.inf
    iteration = 0
    convergence_deltas = []

    print(f"  Censored: {censored_mask.sum()} | Uncensored: {(~censored_mask).sum()}")

    while iteration < max_iter and delta > tol:
        iteration += 1
        m = clone(model)
        m.fit(X, y_imputed)
        mu = m.predict(X[censored_mask])

        # Estimate residual standard deviation
        sigma = np.full_like(
            mu, np.maximum(np.std(y_imputed - m.predict(X)), 0.01)
        )

        # Transform threshold to ALR space
        threshold_alr = np.log(
            threshold_value / reference_value[censored_mask].values
        )

        # Mills-ratio truncated normal correction
        z = (threshold_alr - mu) / sigma
        phi = norm.pdf(z)
        Phi = norm.cdf(z)
        corrected = mu - sigma * (phi / (Phi + 1e-10))

        # Clip to valid operational range
        lower_bound = np.log(1e-6 / reference_value[censored_mask].values)
        corrected = np.clip(corrected, lower_bound, threshold_alr - 1e-10)

        y_imputed[censored_mask] = corrected
        delta = np.mean(np.abs(corrected - prev_imputed[censored_mask]))
        convergence_deltas.append(delta)
        prev_imputed = y_imputed.copy()

        if iteration % 10 == 0 or iteration <= 5:
            print(f"    Iteration {iteration:2d} | Delta: {delta:.6f}")

    print(f"  Converged after {iteration} iterations (delta: {delta:.6f})")

    # Back-transform to ppm
    ppm_imputed = np.exp(y_imputed) * reference_value
    imputed_censored_ppm = ppm_imputed[censored_mask]

    # Model performance on uncensored data
    y_train_pred = m.predict(X_train)
    y_test_pred  = m.predict(X_test)

    key = f"{element}_{model_name}_{threshold_type}"
    result = {
        "element":            element,
        "model":              model_name,
        "threshold_type":     threshold_type,
        "threshold_value":    threshold_value,
        "iterations":         iteration,
        "n_censored":         int(censored_mask.sum()),
        "n_uncensored":       int((~censored_mask).sum()),
        "censoring_rate":     round(censored_mask.sum() / len(censored_mask) * 100, 2),
        "r2_train":           round(r2_score(y_train, y_train_pred), 4),
        "r2_test":            round(r2_score(y_test, y_test_pred), 4),
        "mae_train":          round(mean_absolute_error(y_train, y_train_pred), 4),
        "mae_test":           round(mean_absolute_error(y_test, y_test_pred), 4),
        "rmse_train":         round(np.sqrt(mean_squared_error(y_train, y_train_pred)), 4),
        "rmse_test":          round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 4),
        "min_imputed_ppm":    round(imputed_censored_ppm.min(), 6),
        "max_imputed_ppm":    round(imputed_censored_ppm.max(), 6),
        "mean_imputed_ppm":   round(imputed_censored_ppm.mean(), 6),
        "imputed_alr_all":    y_imputed.copy(),
        "imputed_censored_alr":  y_imputed[censored_mask],
        "imputed_censored_ppm":  imputed_censored_ppm,
        "convergence":        convergence_deltas,
        "y_train_true":       y_train,
        "y_test_true":        y_test,
        "y_train_pred":       y_train_pred,
        "y_test_pred":        y_test_pred,
        "model_obj":          m,
    }

    return key, result


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def calculate_rdcm(S_original, S_imputed):
    """
    Calculate Relative Difference in Covariance Matrix (RDCM).

    RDCM = ||S_orig - S_imp||_F / ||S_orig||_F

    Lower values indicate better preservation of covariance structure.
    """
    diff = S_original - S_imputed
    return np.linalg.norm(diff, 'fro') / np.linalg.norm(S_original, 'fro')


# =============================================================================
# VISUALISATION
# =============================================================================

def create_performance_plots(results, element, save_path):
    """Predicted vs observed scatter plots for train and test sets."""

    element_results = {k: v for k, v in results.items() if v['element'] == element}
    if not element_results:
        return

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    colors = {'RF': '#2ca02c', 'XGB': '#1f77b4'}
    plot_idx = 0

    for threshold_type in ['DL', 'AL']:
        for model_name in ['RF', 'XGB']:
            key = f"{element}_{model_name}_{threshold_type}"
            if key not in element_results:
                continue
            result = element_results[key]

            for row, split in enumerate(['train', 'test']):
                ax = axes[row, plot_idx]
                y_true = result[f"y_{split}_true"]
                y_pred = result[f"y_{split}_pred"]
                r2    = result[f"r2_{split}"]

                ax.scatter(y_true, y_pred, alpha=0.6,
                           color=colors[model_name], s=30,
                           edgecolor='black', linewidth=0.5)
                lims = [min(y_true.min(), y_pred.min()),
                        max(y_true.max(), y_pred.max())]
                ax.plot(lims, lims, 'r--', linewidth=2)
                ax.set_xlabel(f'Observed ALR_{element}', fontsize=10)
                ax.set_ylabel(f'Predicted ALR_{element}', fontsize=10)
                ax.set_title(
                    f'{model_name}-{threshold_type} ({split.capitalize()})\n'
                    f'R² = {r2:.4f}',
                    fontweight='bold', fontsize=11
                )
                ax.grid(True, alpha=0.3)

            plot_idx += 1

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{element}_Performance_Plots.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Performance plots saved for {element}")


def create_histograms_and_boxplots(results, element, original_data,
                                    censored_mask, alr_col, save_path):
    """Distribution histograms with KDE and comparative boxplot."""

    element_results = {k: v for k, v in results.items() if v['element'] == element}
    if not element_results:
        return

    datasets = {'Original': original_data.copy()}
    for key, result in element_results.items():
        label = f"{result['model']}_{result['threshold_type']}"
        imputed_full = original_data.copy()
        imputed_full[censored_mask] = result['imputed_censored_alr']
        datasets[label] = imputed_full

    all_vals = np.concatenate(list(datasets.values()))
    x_min, x_max = all_vals.min() - 0.2, all_vals.max() + 0.2

    n = len(datasets)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    colors = ['#d62728', '#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd']

    fig = plt.figure(figsize=(6 * cols, 4 * rows + 4))

    for i, (label, data) in enumerate(datasets.items()):
        ax = plt.subplot(rows + 1, cols, i + 1)
        ax.hist(data, bins=30, alpha=0.7, color=colors[i % len(colors)],
                density=True, range=(x_min, x_max),
                edgecolor='black', linewidth=0.5)
        try:
            kde = gaussian_kde(data)
            xs = np.linspace(x_min, x_max, 200)
            ax.plot(xs, kde(xs), color='black', linewidth=2)
        except Exception:
            pass
        ax.set_title(
            f'{element} — {label}\n'
            f'mean={data.mean():.3f}, var={data.var():.3f}',
            fontweight='bold', fontsize=10
        )
        ax.set_xlabel(f'ALR_{element}', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_xlim(x_min, x_max)
        ax.grid(True, alpha=0.3)

    ax_box = plt.subplot(rows + 1, 1, rows + 1)
    bp = ax_box.boxplot(list(datasets.values()),
                        tick_labels=list(datasets.keys()),
                        patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax_box.set_title(f'{element} Distribution Comparison',
                     fontsize=14, fontweight='bold', pad=20)
    ax_box.set_ylabel(f'ALR_{element}', fontsize=12)
    ax_box.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{element}_Histograms_Boxplots.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Histograms and boxplots saved for {element}")


def create_rdcm_analysis(results, element, df, censored_mask,
                          alr_col, save_path):
    """RDCM bar chart and imputation statistics table."""

    element_results = {k: v for k, v in results.items() if v['element'] == element}
    if not element_results:
        return {}

    alr_columns  = [c for c in df.columns if c.startswith('ALR_')]
    S_original   = df[alr_columns].cov()
    rdcm_results = {}
    imputation_stats = {}

    for key, result in element_results.items():
        imputed_data = df[alr_columns].copy()
        imputed_data.loc[censored_mask, alr_col] = result['imputed_censored_alr']
        S_imputed = imputed_data.cov()
        rdcm      = calculate_rdcm(S_original, S_imputed)

        label = f"{result['model']}_{result['threshold_type']}"
        rdcm_results[label] = rdcm

        ppm     = result['imputed_censored_ppm']
        thr_val = result['threshold_value']
        n_below = int(np.sum(ppm < thr_val))
        n_total = len(ppm)
        imputation_stats[label] = {
            'total':         n_total,
            'below':         n_below,
            'pct_below':     round(n_below / n_total * 100, 1) if n_total else 0,
            'threshold_type': result['threshold_type'],
        }

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6),
                                    gridspec_kw={'width_ratios': [3, 2]})
    methods = list(rdcm_results.keys())
    values  = list(rdcm_results.values())
    colors  = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6']
    y_pos   = np.arange(len(methods))

    bars = ax1.barh(y_pos, values, color=colors, alpha=0.8,
                    edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars, values):
        ax1.text(bar.get_width() + max(values) * 0.01,
                 bar.get_y() + bar.get_height() / 2,
                 f'{val:.6f}', ha='left', va='center',
                 fontweight='bold', fontsize=11)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(methods, fontweight='bold', fontsize=11)
    ax1.set_xlabel('RDCM Value', fontsize=12, fontweight='bold')
    ax1.set_title(f'{element} — RDCM Comparison\n(Lower = Better)',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')

    ax2.axis('off')
    table_data = [
        [m, str(imputation_stats[m]['total']),
         str(imputation_stats[m]['below']),
         f"{imputation_stats[m]['pct_below']}%"]
        for m in methods
    ]
    tbl = ax2.table(cellText=table_data,
                    colLabels=['Method', 'Total', 'Below Threshold', '% Below'],
                    cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.2)
    ax2.set_title('Imputation Statistics', fontweight='bold',
                  fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{element}_RDCM_Analysis.png"),
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n  {element} RDCM results:")
    for label, rdcm in rdcm_results.items():
        s = imputation_stats[label]
        print(f"    {label}: RDCM = {rdcm:.6f} | "
              f"Below threshold: {s['below']}/{s['total']} ({s['pct_below']}%)")

    return rdcm_results


def create_rdcm_comparison_plots(all_results, df, save_path):
    """Grouped bar chart and line plot comparing RDCM across all elements."""

    element_config = ELEMENT_CONFIG
    rdcm_data = {}

    for element in element_config:
        alr_col      = f"ALR_{element}"
        censored_flag = element_config[element]["censored_flag"]
        alr_columns  = [c for c in df.columns if c.startswith('ALR_')]

        if censored_flag not in df.columns:
            continue

        censored_mask = df[censored_flag] == 1
        S_original    = df[alr_columns].cov()
        element_rdcm  = {}

        for method in ['RF_DL', 'RF_AL', 'XGB_DL', 'XGB_AL']:
            result_key = f"{element}_{method}"
            if result_key not in all_results:
                continue
            result = all_results[result_key]
            imputed_data = df[alr_columns].copy()
            imputed_data.loc[censored_mask, alr_col] = result['imputed_censored_alr']
            element_rdcm[method] = calculate_rdcm(S_original, imputed_data.cov())

        rdcm_data[element] = element_rdcm

    elements = list(rdcm_data.keys())
    methods  = ['RF_DL', 'RF_AL', 'XGB_DL', 'XGB_AL']
    labels   = ['RF-DL', 'RF-AL', 'XGB-DL', 'XGB-AL']
    colors   = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6']
    x        = np.arange(len(elements))
    width    = 0.2

    # Grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    for i, (method, label, color) in enumerate(zip(methods, labels, colors)):
        vals = [rdcm_data[e].get(method, 0) for e in elements]
        bars = ax.bar(x + i * width, vals, width,
                      label=label, color=color, alpha=0.8,
                      edgecolor='black', linewidth=1)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f'{val:.6f}', ha='center', va='bottom',
                        fontweight='bold', fontsize=9)
    ax.set_xlabel('Elements', fontsize=14, fontweight='bold')
    ax.set_ylabel('RDCM Value', fontsize=14, fontweight='bold')
    ax.set_title('RDCM Comparison Across Elements and Methods\n(Lower = Better)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(elements, fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "RDCM_Comparison_Grouped_Bars.png"),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Line plot
    fig, ax = plt.subplots(figsize=(12, 8))
    styles = {
        'Ag': {'color': '#e74c3c', 'marker': 'o', 'linestyle': '-'},
        'Hg': {'color': '#3498db', 'marker': 's', 'linestyle': '--'},
        'Ta': {'color': '#2ecc71', 'marker': '^', 'linestyle': '-.'},
    }
    x_pos = np.arange(len(methods))
    for element, element_rdcm in rdcm_data.items():
        vals  = [element_rdcm.get(m, 0) for m in methods]
        style = styles.get(element, {'color': 'grey', 'marker': 'o', 'linestyle': '-'})
        ax.plot(x_pos, vals, label=element,
                color=style['color'], marker=style['marker'],
                linestyle=style['linestyle'], linewidth=3,
                markersize=10, markeredgecolor='black', markeredgewidth=1.5)
    ax.set_xlabel('Imputation Method', fontsize=14, fontweight='bold')
    ax.set_ylabel('RDCM Value', fontsize=14, fontweight='bold')
    ax.set_title('RDCM Trends Across Elements and Methods\n(Lower = Better)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "RDCM_Comparison_Lines.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  RDCM comparison plots saved.")
    return rdcm_data


def create_performance_summary(results, element):
    """Return a DataFrame summarising model performance metrics."""
    element_results = {k: v for k, v in results.items() if v['element'] == element}
    if not element_results:
        return pd.DataFrame()

    rows = []
    for key, result in element_results.items():
        rows.append({
            'Element':          result['element'],
            'Model':            result['model'],
            'Threshold':        result['threshold_type'],
            'Threshold_Value':  result['threshold_value'],
            'N_Censored':       result['n_censored'],
            'Censoring_Rate_%': result['censoring_rate'],
            'Iterations':       result['iterations'],
            'R2_Train':         result['r2_train'],
            'R2_Test':          result['r2_test'],
            'MAE_Train':        result['mae_train'],
            'MAE_Test':         result['mae_test'],
            'RMSE_Train':       result['rmse_train'],
            'RMSE_Test':        result['rmse_test'],
            'Min_Imputed_ppm':  result['min_imputed_ppm'],
            'Max_Imputed_ppm':  result['max_imputed_ppm'],
            'Mean_Imputed_ppm': result['mean_imputed_ppm'],
        })
    return pd.DataFrame(rows)


def create_comprehensive_imputation_summary(all_results, save_path):
    """Create and save a comprehensive CSV summary of all results."""
    rows = []
    for key, result in all_results.items():
        ppm     = result['imputed_censored_ppm']
        thr_val = result['threshold_value']
        n_below = int(np.sum(ppm < thr_val))
        n_above = int(np.sum(ppm >= thr_val))
        n_total = len(ppm)
        rows.append({
            'Element':            result['element'],
            'Model':              result['model'],
            'Threshold_Type':     result['threshold_type'],
            'Threshold_Value_ppm': thr_val,
            'Total_Censored':     n_total,
            'Below_Threshold':    n_below,
            'Above_Threshold':    n_above,
            'Percent_Below':      round(n_below / n_total * 100, 2) if n_total else 0,
            'Percent_Above':      round(n_above / n_total * 100, 2) if n_total else 0,
            'Min_Imputed_ppm':    round(ppm.min(), 6),
            'Max_Imputed_ppm':    round(ppm.max(), 6),
            'Mean_Imputed_ppm':   round(ppm.mean(), 6),
            'Std_Imputed_ppm':    round(ppm.std(), 6),
            'Iterations':         result['iterations'],
            'R2_Test':            result['r2_test'],
        })
    summary_df = pd.DataFrame(rows)
    out = os.path.join(save_path, "Comprehensive_Imputation_Summary.csv")
    summary_df.to_csv(out, index=False)
    print(f"  Comprehensive summary saved: {out}")
    return summary_df


def create_comprehensive_csv(all_results, df, save_path):
    """Append all imputed ALR columns to the original dataset and save."""
    output_df = df.copy()

    for element in ELEMENT_CONFIG:
        alr_col      = f"ALR_{element}"
        censored_flag = ELEMENT_CONFIG[element]["censored_flag"]

        if alr_col not in output_df.columns or censored_flag not in output_df.columns:
            continue

        censored_mask = output_df[censored_flag] == 1

        for method in ['RF_DL', 'RF_AL', 'XGB_DL', 'XGB_AL']:
            result_key = f"{element}_{method}"
            new_col    = f"ALR_{element}_{method}_imputed"
            output_df[new_col] = output_df[alr_col].copy()

            if result_key in all_results:
                result = all_results[result_key]
                output_df.loc[censored_mask, new_col] = result['imputed_censored_alr']

    out = os.path.join(save_path, "Comprehensive_Imputation_Results.csv")
    output_df.to_csv(out, index=False, sep=';')
    print(f"  Full imputed dataset saved: {out}")
    return out


# =============================================================================
# ELEMENT ANALYSIS WORKFLOW
# =============================================================================

def analyze_element(element, df, results_folder):
    """
    Run the complete ReX-CoDA workflow for a single element.

    Includes predictor selection, imputation under DL and AL thresholds
    using both RF and XGB, and generation of all diagnostic outputs.
    """

    print(f"\n{'*' * 60}")
    print(f"  Analysing {element}")
    print(f"{'*' * 60}")

    config        = ELEMENT_CONFIG[element]
    alr_col       = f"ALR_{element}"
    censored_flag = config["censored_flag"]
    reference_col = config["reference_col"]

    if alr_col not in df.columns or censored_flag not in df.columns:
        print(f"  Required columns not found for {element}. Skipping.")
        return {}

    censored_mask = df[censored_flag] == 1

    # Predictor selection by covariance (uncensored subset only)
    alr_columns     = [c for c in df.columns if c.startswith('ALR_')]
    uncensored_data = df.loc[~censored_mask, alr_columns]

    covariances = {}
    for col in alr_columns:
        if col != alr_col:
            cov = uncensored_data[[alr_col, col]].cov().iloc[0, 1]
            covariances[col] = abs(cov)

    top_predictors = sorted(covariances.items(),
                            key=lambda x: x[1], reverse=True)[:15]
    predictor_cols = [col for col, _ in top_predictors]
    print(f"  Top 5 predictors: {predictor_cols[:5]}")

    X               = df[predictor_cols].values
    y               = df[alr_col].values
    reference_value = df[reference_col]

    # Train/test split on uncensored observations only
    uncensored_idx = np.where(~censored_mask)[0]
    train_idx, test_idx = train_test_split(
        uncensored_idx, test_size=0.2, random_state=42
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    results = {}

    for threshold_type in ['DL', 'AL']:
        thr_val = (config['dl_value'] if threshold_type == 'DL'
                   else config['dl_value'] * (1 + AL_TOLERANCE))

        for model, model_name in [(RF_MODEL, 'RF'), (XGB_MODEL, 'XGB')]:
            key, result = run_rexcoda_imputation(
                model, model_name, element, threshold_type,
                X, y, X_train, X_test, y_train, y_test,
                censored_mask, thr_val, reference_value
            )
            if result is not None:
                results[key] = result

    if not results:
        print(f"  No results generated for {element}.")
        return {}

    # Diagnostic outputs
    try:
        create_performance_plots(results, element, results_folder)
        create_histograms_and_boxplots(
            results, element, df[alr_col], censored_mask, alr_col, results_folder
        )
        create_rdcm_analysis(
            results, element, df, censored_mask, alr_col, results_folder
        )

        summary_df = create_performance_summary(results, element)
        if not summary_df.empty:
            out = os.path.join(results_folder, f"{element}_Performance_Summary.csv")
            summary_df.to_csv(out, index=False)

        print(f"\n  {element} analysis complete.")
    except Exception as e:
        print(f"  Error during visualisation for {element}: {e}")

    return results


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_complete_analysis(data_path, results_folder):
    """Run ReX-CoDA for all configured elements."""

    print("=" * 80)
    print("ReX-CoDA: Regression-based Flexible-Limit Compositional Data Imputation")
    print("Elements: Ag (Silver), Hg (Mercury), Ta (Tantalum)")
    print("Thresholds: Detection Limit (DL) and Acceptable Limit (AL)")
    print("Models: Random Forest (RF) and Extreme Gradient Boosting (XGB)")
    print("=" * 80)

    df = load_data(data_path)
    create_output_folder(results_folder)

    # Print element configuration
    print("\nElement configuration:")
    for element, config in ELEMENT_CONFIG.items():
        al_val = config['dl_value'] * (1 + AL_TOLERANCE)
        print(f"  {element}: DL = {config['dl_value']} ppm | "
              f"AL = {al_val:.4f} ppm")

    all_results = {}

    for element in ELEMENT_CONFIG:
        try:
            element_results = analyze_element(element, df, results_folder)
            if element_results:
                all_results.update(element_results)
        except Exception as e:
            print(f"  Error processing {element}: {e}")
            continue

    if all_results:
        print(f"\n{'=' * 80}")
        print("CREATING COMPREHENSIVE OUTPUTS")
        print(f"{'=' * 80}")

        create_comprehensive_csv(all_results, df, results_folder)
        create_comprehensive_imputation_summary(all_results, results_folder)
        create_rdcm_comparison_plots(all_results, df, results_folder)

        all_summaries = []
        for element in ELEMENT_CONFIG:
            el_results = {k: v for k, v in all_results.items()
                          if v.get('element') == element}
            if el_results:
                all_summaries.append(create_performance_summary(el_results, element))
        if all_summaries:
            combined = pd.concat(all_summaries, ignore_index=True)
            combined.to_csv(
                os.path.join(results_folder, "Performance_Summary_All_Elements.csv"),
                index=False
            )

        print(f"\n{'=' * 80}")
        print("ANALYSIS COMPLETE")
        print(f"{'=' * 80}")
        for element in ELEMENT_CONFIG:
            el_results = {k: v for k, v in all_results.items()
                          if v.get('element') == element}
            n   = len(el_results)
            avg = (np.mean([v['censoring_rate'] for v in el_results.values()])
                   if el_results else 0)
            print(f"  {element}: {n}/4 methods successful | "
                  f"Avg censoring rate: {avg:.1f}%")

        print(f"\nAll outputs saved to: {results_folder}")
        print("ReX-CoDA analysis complete.")
    else:
        print("No results generated for any element.")


if __name__ == "__main__":
    args = parse_arguments()
    run_complete_analysis(
        data_path      = args.data,
        results_folder = args.output,
    )
