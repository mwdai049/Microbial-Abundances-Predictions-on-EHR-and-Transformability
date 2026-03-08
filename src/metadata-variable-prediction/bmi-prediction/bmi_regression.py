#!/usr/bin/env python3
"""
BMI Regression Pipeline

Purpose
-------
Predict continuous BMI from microbiome features using two abundance tables:
    1. Absolute abundance
    2. Relative abundance

This pipeline compares predictive performance between the two data
representations and evaluates whether differences are statistically
significant using bootstrap testing.

Pipeline Steps
--------------
1. Load train / validation / test datasets.

2. Preprocess microbiome features:
      - keep columns starting with "G"
      - apply prevalence filter on TRAIN only
      - apply log10(x + 1) transform

3. Train regression models on Absolute and Relative data:
      - RandomForest
      - HistGradientBoosting (with grid search)
      - SVM (RBF kernel with grid search)

4. Hyperparameter tuning
      - GridSearchCV is applied to HGB and SVM models
      - best parameters are recorded and printed

5. Evaluate models using:
      - R² (coefficient of determination)
      - Mean Absolute Error (MAE)

6. Generate prediction plots:
      - True BMI vs Predicted BMI
      - saved as PNG figures

7. Statistical validation:
      - Bootstrap significance testing compares Absolute vs Relative models
      - metrics evaluated:
            ΔMAE
            ΔR²
      - reports mean difference, 95% confidence interval, and p-value

8. Save outputs:
      - trained models
      - evaluation plots
      - metrics tables

Output
------
The pipeline saves results to the directory specified by PLOT_DIR
(or the default location if not provided).

Files generated:

    PLOT_DIR/
        bmi_Absolute_<model>.png
        bmi_Relative_<model>.png
        reg_bmi_results_<model>.csv

    MODEL_DIR/
        bmi_Absolute_<model>.joblib
        bmi_Relative_<model>.joblib

Additionally printed to console:
    - bootstrap significance test results
    - best hyperparameters for each trained model

Usage
-----
Run all models:
    python bmi_regression.py

Run a single model:
    python bmi_regression.py --model RandomForest
    python bmi_regression.py --model HGB
    python bmi_regression.py --model SVM_RBF

Optional environment variables
    DATA_DIR    Path to input CSV files.
    PLOT_DIR    Directory where regression plots and metrics will be saved.
    MODEL_DIR   Directory where trained models will be saved.

Example
-------
DATA_DIR="/ddn_scratch/k5zhao/data/classifier_training" \
PLOT_DIR="/home/zhw074/bmi/regression_plots" \
MODEL_DIR="/home/zhw074/bmi/models" \
python bmi_regression.py --model HGB
"""

import os
import argparse
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.svm import SVR, SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder

DEFAULT_DATA_DIR = "/ddn_scratch/k5zhao/data/classifier_training"
DEFAULT_PLOT_DIR = "/home/zhw074/bmi/regression_plots"
DEFAULT_MODEL_DIR = "/home/zhw074/bmi/models"

def load_data(data_dir=None):
    """
    Load abs/rel train/val/test CSVs.

    Expects:
      abs_train.csv, abs_val.csv, abs_test.csv,
      rel_train.csv, rel_val.csv, rel_test.csv
    under data_dir.
    """
    data_dir = data_dir or os.environ.get("DATA_DIR", DEFAULT_DATA_DIR)

    abs_train = pd.read_csv(os.path.join(data_dir, "abs_train.csv"), low_memory=False)
    abs_val   = pd.read_csv(os.path.join(data_dir, "abs_val.csv"), low_memory=False)
    abs_test  = pd.read_csv(os.path.join(data_dir, "abs_test.csv"), low_memory=False)

    rel_train = pd.read_csv(os.path.join(data_dir, "rel_train.csv"), low_memory=False)
    rel_val   = pd.read_csv(os.path.join(data_dir, "rel_val.csv"), low_memory=False)
    rel_test  = pd.read_csv(os.path.join(data_dir, "rel_test.csv"), low_memory=False)

    return abs_train, abs_val, abs_test, rel_train, rel_val, rel_test

def preprocess_representation(train_df, val_df, test_df, prevalence_thresh):
    """
    Matches your pipeline behavior:
    - Use only columns starting with "G"
    - Prevalence filter computed ONLY on train
    - log10(x + 1)
    """
    feature_cols = [c for c in train_df.columns if c.startswith("G")]

    X_train_raw = train_df[feature_cols]
    X_val_raw   = val_df[feature_cols]
    X_test_raw  = test_df[feature_cols]

    keep_mask = (X_train_raw > 0).mean(axis=0) >= prevalence_thresh

    X_train = np.log10(X_train_raw.loc[:, keep_mask] + 1)
    X_val   = np.log10(X_val_raw.loc[:, keep_mask] + 1)
    X_test  = np.log10(X_test_raw.loc[:, keep_mask] + 1)

    return X_train, X_val, X_test


def run_regression_experiments(
    target_col,
    abs_train, abs_val, abs_test,
    rel_train, rel_val, rel_test,
    prevalence_thresh=0.4,
    selected_model="all",
    model_dir=None,
):
    """
    Run regression models on absolute vs relative representations.

    - Trains RandomForest, HGB, and SVM_RBF (or a single chosen model)
    - Caches X_te and y_te for later plotting or SHAP
    - Does NOT generate or save any plots

    Returns
    -------
    results_df : pd.DataFrame
        Metrics for each (Representation, Model)

    best_models : dict
        (rep_name, model_name) -> fitted model
        (rep_name, model_name, "X_te") -> test features
        (rep_name, model_name, "y_te") -> test targets
    """

    regressors = {
        "RandomForest": {
            "model": RandomForestRegressor(
                n_estimators=300,
                max_depth=20,
                random_state=42,
                n_jobs=-1,
                verbose=1
            ),
            "param_grid": None
        },
        "HGB": {
            "model": HistGradientBoostingRegressor(
                max_depth=8,
                learning_rate=0.05,
                max_iter=300,
                random_state=42,
                verbose=1
            ),
            "param_grid": {
                "max_depth": [6, 8, 10],
                "learning_rate": [0.05, 0.1],
            }
        },
        "SVM_RBF": {
            "model": SVR(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                verbose=True
            ),
            "param_grid": {
                "C": [1.0, 10.0, 100.0],
                "gamma": ["scale", "auto"],
            }
        },
    }

    # ---- filter to a single model if requested ----
    if selected_model != "all":
        if selected_model not in regressors:
            raise ValueError(f"selected_model must be one of {['all'] + list(regressors.keys())}")
        regressors = {selected_model: regressors[selected_model]}

    # Extract targets
    y_abs_train = abs_train[target_col].values
    y_abs_val   = abs_val[target_col].values
    y_abs_test  = abs_test[target_col].values

    y_rel_train = rel_train[target_col].values
    y_rel_val   = rel_val[target_col].values
    y_rel_test  = rel_test[target_col].values

    reps = {
        "Absolute": (abs_train, abs_val, abs_test, y_abs_train, y_abs_val, y_abs_test),
        "Relative": (rel_train, rel_val, rel_test, y_rel_train, y_rel_val, y_rel_test),
    }

    all_results = []
    best_models = {}
    best_params_summary = {}

    if model_dir is not None:
        os.makedirs(model_dir, exist_ok=True)

    for rep_name, (train_df, val_df, test_df, y_tr, y_va, y_te) in reps.items():

        print(f"\n=== Representation: {rep_name} ===")

        # Preprocess once per representation
        X_tr, X_va, X_te = preprocess_representation(
            train_df, val_df, test_df,
            prevalence_thresh=prevalence_thresh
        )

        for model_name, info in regressors.items():

            base_model = info["model"]
            param_grid = info["param_grid"]

            print(f"\n>>> Training {model_name} on {rep_name} ...")

            if param_grid is not None:
                gs = GridSearchCV(
                    estimator=clone(base_model),
                    param_grid=param_grid,
                    cv=3,
                    scoring="neg_mean_absolute_error",
                    n_jobs=-1,
                    verbose=2
                )
                gs.fit(X_tr, y_tr)
                model = gs.best_estimator_
                print(f"Best parameters: {gs.best_params_}")
                best_params_summary[(rep_name, model_name)] = gs.best_params_
            else:
                model = clone(base_model)
                model.fit(X_tr, y_tr)
                best_params_summary[(rep_name, model_name)] = "default parameters"

            # Cache model + test data
            best_models[(rep_name, model_name)] = model
            best_models[(rep_name, model_name, "X_te")] = X_te
            best_models[(rep_name, model_name, "y_te")] = y_te

            if model_dir is not None:
                model_path = os.path.join(model_dir, f"{target_col}_{rep_name}_{model_name}.joblib")
                joblib.dump(model, model_path)

            # Validation metrics
            y_va_pred = model.predict(X_va)
            val_r2  = r2_score(y_va, y_va_pred)
            val_mae = mean_absolute_error(y_va, y_va_pred)

            # Test metrics
            y_te_pred = model.predict(X_te)
            test_r2  = r2_score(y_te, y_te_pred)
            test_mae = mean_absolute_error(y_te, y_te_pred)

            all_results.append({
                "Task": "Regression",
                "Target": target_col,
                "Representation": rep_name,
                "Model": model_name,
                "Prevalence": prevalence_thresh,
                "Val_R2": val_r2,
                "Test_R2": test_r2,
                "Val_MAE": val_mae,
                "Test_MAE": test_mae,
            })

            print(
                f"[{rep_name} - {model_name}] "
                f"Val_R2={val_r2:.3f}, Test_R2={test_r2:.3f}, "
                f"Val_MAE={val_mae:.3f}, Test_MAE={test_mae:.3f}"
            )

    results_df = pd.DataFrame(all_results)
    return results_df, best_models, best_params_summary

# plotting
def plot_cached_regression_results(
    best_models,
    target_col,
    reps=("Absolute", "Relative"),
    models=("RandomForest", "HGB", "SVM_RBF"),
    save_dir=None,
):
    """
    Plot regression results directly from cached models and test data.

    Parameters
    ----------
    save_dir : str or Path or None
        If provided, save PNG files to this directory instead of showing inline.
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for rep_name in reps:
        for model_name in models:
            key_model = (rep_name, model_name)
            key_X = (rep_name, model_name, "X_te")
            key_y = (rep_name, model_name, "y_te")

            if key_model not in best_models:
                print(f"[skip] No model found for {rep_name} - {model_name}")
                continue

            model = best_models[key_model]
            X_te = best_models[key_X]
            y_te = best_models[key_y]

            y_pred = model.predict(X_te)
            test_r2 = r2_score(y_te, y_pred)
            test_mae = mean_absolute_error(y_te, y_pred)

            plt.figure(figsize=(4, 4))
            plt.scatter(y_te, y_pred, alpha=0.4)

            lim_min = min(y_te.min(), y_pred.min())
            lim_max = max(y_te.max(), y_pred.max())
            plt.plot([lim_min, lim_max], [lim_min, lim_max], "r--", linewidth=1)

            plt.xlabel(f"True {target_col}")
            plt.ylabel(f"Predicted {target_col}")
            plt.title(
                f"{target_col} | {rep_name} - {model_name}\n"
                f"R² = {test_r2:.3f}, MAE = {test_mae:.2f}"
            )
            plt.gca().set_aspect("equal", adjustable="box")
            plt.tight_layout()

            if save_dir is None:
                plt.show()
            else:
                fname = f"{target_col}_{rep_name}_{model_name}.png"
                plt.savefig(os.path.join(save_dir, fname), dpi=200, bbox_inches="tight")
                plt.close()


def bootstrap_regression_test(y_true, pred_abs, pred_rel, n_boot=10000):

    rng = np.random.default_rng(42)
    n = len(y_true)

    mae_diffs = []
    r2_diffs = []

    for _ in range(n_boot):

        idx = rng.choice(n, n, replace=True)

        y_sample = y_true[idx]
        abs_sample = pred_abs[idx]
        rel_sample = pred_rel[idx]

        mae_abs = mean_absolute_error(y_sample, abs_sample)
        mae_rel = mean_absolute_error(y_sample, rel_sample)
        mae_diffs.append(mae_rel - mae_abs)

        r2_abs = r2_score(y_sample, abs_sample)
        r2_rel = r2_score(y_sample, rel_sample)
        r2_diffs.append(r2_rel - r2_abs)

    mae_diffs = np.array(mae_diffs)
    r2_diffs = np.array(r2_diffs)

    print("\nBOOTSTRAP RESULTS")
    print("------------------")
    print("ΔMAE:", mae_diffs.mean())
    print("95% CI:", np.percentile(mae_diffs, [2.5, 97.5]))
    print("p-value:", 2 * min(np.mean(mae_diffs <= 0), np.mean(mae_diffs >= 0)))

    print("\nΔR²:", r2_diffs.mean())
    print("95% CI:", np.percentile(r2_diffs, [2.5, 97.5]))
    print("p-value:", 2 * min(np.mean(r2_diffs <= 0), np.mean(r2_diffs >= 0)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="all",
        choices=["all", "RandomForest", "HGB", "SVM_RBF"],
        help="Choose which model(s) to run."
    )
    parser.add_argument(
        "--prevalence",
        type=float,
        default=0.4,
        help="Prevalence threshold for feature filtering (default: 0.4)."
    )
    args = parser.parse_args(args=[])

    # Load data
    abs_train, abs_val, abs_test, rel_train, rel_val, rel_test = load_data()

    # Run experiments
    reg_bmi_results, reg_bmi_models, best_params_summary = run_regression_experiments(
        target_col="bmi",
        abs_train=abs_train,
        abs_val=abs_val,
        abs_test=abs_test,
        rel_train=rel_train,
        rel_val=rel_val,
        rel_test=rel_test,
        prevalence_thresh=args.prevalence,
        selected_model=args.model,
        model_dir=os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR),
    )

    # Plot directory
    plot_dir = os.environ.get("PLOT_DIR", DEFAULT_PLOT_DIR)
    model_dir = os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR)

    # Decide which plots to generate (all vs one)
    if args.model == "all":
        plot_models = ("RandomForest", "HGB", "SVM_RBF")
    else:
        plot_models = (args.model,)

    plot_cached_regression_results(
        best_models=reg_bmi_models,
        target_col="bmi",
        models=plot_models,
        save_dir=plot_dir
    )

    # Save results table (helpful for batch runs)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    out_csv = os.path.join(plot_dir, f"reg_bmi_results_{args.model}.csv")
    reg_bmi_results.to_csv(out_csv, index=False)

    print("\nRunning bootstrap significance tests...")

    models = ("RandomForest", "HGB", "SVM_RBF")

    for m in models:

        if ("Absolute", m) not in reg_bmi_models or ("Relative", m) not in reg_bmi_models:
            print(f"Skipping {m} (model not run)")
            continue

        model_abs = reg_bmi_models[("Absolute", m)]
        model_rel = reg_bmi_models[("Relative", m)]

        X_abs = reg_bmi_models[("Absolute", m, "X_te")]
        X_rel = reg_bmi_models[("Relative", m, "X_te")]

        y = reg_bmi_models[("Absolute", m, "y_te")]

        pred_abs = model_abs.predict(X_abs)
        pred_rel = model_rel.predict(X_rel)

        print(f"\n=== Bootstrap test: {m} ===")

        bootstrap_regression_test(
            y_true=np.array(y),
            pred_abs=np.array(pred_abs),
            pred_rel=np.array(pred_rel)
        )

    print("\nSaved plots to:", plot_dir)
    print("Saved models to:", model_dir)
    print("Saved metrics to:", out_csv)

    print("\nBest hyperparameters for each model")
    print("------------------------------------")
    for (rep, model), params in best_params_summary.items():
        print(f"{rep} - {model}: {params}")

    print("\nResults preview:")
    print(reg_bmi_results)


if __name__ == "__main__":
    main()