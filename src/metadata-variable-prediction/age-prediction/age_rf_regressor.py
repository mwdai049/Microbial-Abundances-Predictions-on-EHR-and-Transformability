# Filepath imports
# Making sure filepath runs smoothly
import sys
import os

print(sys.executable)

env_root = sys.prefix

env_bin = os.path.join(env_root, "bin")
os.environ["PATH"] = env_bin + os.pathsep + os.environ.get("PATH", "")

import rpy2.robjects as ro
from rpy2.robjects.packages import isinstalled
env_r_lib = os.path.join(env_root, "lib", "R", "library")
ro.r(f'.libPaths("{env_r_lib}")')

print("R libPaths now:")
print(ro.r(".libPaths()"))
print("phyloseq installed?", isinstalled("phyloseq"))

# imports for data analysis

# data analysis
import pandas as pd
import numpy as np
import json

# qiime and related
import qiime2
from qiime2 import Metadata, Artifact
import biom

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# machine learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score, roc_curve, auc, roc_auc_score

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from scipy.interpolate import UnivariateSpline


DATA_PATHS = {
    "abs": {
        "train": "/ddn_scratch/k5zhao/data/classifier_training/abs_train.csv",
        "test": "/ddn_scratch/k5zhao/data/classifier_training/abs_test.csv",
        "val": "/ddn_scratch/k5zhao/data/classifier_training/abs_val.csv",
    },
    "rel": {
        "train": "/ddn_scratch/k5zhao/data/classifier_training/rel_train.csv",
        "test": "/ddn_scratch/k5zhao/data/classifier_training/rel_test.csv",
        "val": "/ddn_scratch/k5zhao/data/classifier_training/rel_val.csv",
    },
}

AGE_MIN = 20
AGE_MAX = 69
FEATURE_COL_COUNT = 1148
ID_COL = "original_SampleID"
TARGET_COL = "age"
RANDOM_STATE = 42
N_ESTIMATORS = 300


def load_split_data(paths_by_split):
    datasets = {}
    for split_name, path in paths_by_split.items():
        datasets[split_name] = pd.read_csv(path)
    return datasets



def filter_age_range(df, age_col=TARGET_COL, min_age=AGE_MIN, max_age=AGE_MAX):
    return df.loc[(df[age_col] >= min_age) & (df[age_col] <= max_age)].copy()



def select_feature_target(df, feature_col_count=FEATURE_COL_COUNT, id_col=ID_COL, target_col=TARGET_COL):
    X = df[df.columns[:feature_col_count]].drop(columns=[id_col]).copy()
    y = df[target_col].copy()
    return X, y



def log_transform_splits(feature_splits):
    transformed = {}
    for split_name, X in feature_splits.items():
        transformed[split_name] = np.log1p(X.copy())
    return transformed



def prepare_dataset_group(paths_by_split, apply_log_transform=True):
    raw_splits = load_split_data(paths_by_split)
    filtered_splits = {}
    feature_splits = {}
    target_splits = {}

    for split_name, df in raw_splits.items():
        filtered_df = filter_age_range(df)
        filtered_splits[split_name] = filtered_df
        X, y = select_feature_target(filtered_df)
        feature_splits[split_name] = X
        target_splits[split_name] = y

    if apply_log_transform:
        feature_splits = log_transform_splits(feature_splits)

    return {
        "dataframes": filtered_splits,
        "X": feature_splits,
        "y": target_splits,
    }



def build_random_forest_regressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE):
    return RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )



def evaluate_regression(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred, squared=False),
        "r2": r2_score(y_true, y_pred),
    }



def print_metrics(metrics, dataset_label):
    print(f"{dataset_label} MAE (years):", metrics["mae"])
    print(f"{dataset_label} RMSE (years):", metrics["rmse"])
    print(f"{dataset_label} R²:", metrics["r2"])



def plot_predicted_vs_true(y_true, y_pred, title):
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.5)

    min_age = min(y_true.min(), y_pred.min())
    max_age = max(y_true.max(), y_pred.max())
    plt.plot([min_age, max_age], [min_age, max_age], linestyle="--")
    plt.xlabel("True Age")
    plt.ylabel("Predicted Age")
    plt.title(title)
    plt.tight_layout()
    plt.show()



def fit_relative_age_spline(y_true, y_pred):
    order = np.argsort(y_true)
    x_sorted = np.array(y_true)[order]
    y_sorted = np.array(y_pred)[order]

    spline = UnivariateSpline(
        x_sorted,
        y_sorted,
        s=len(x_sorted) * np.var(y_sorted),
    )
    return spline, x_sorted, y_sorted



def build_relative_age_results(y_true, y_pred):
    spline, x_sorted, y_sorted = fit_relative_age_spline(y_true, y_pred)
    expected_microbiota_age = spline(y_true)

    results = pd.DataFrame({
        "chronological_age": y_true,
        "microbiota_age": y_pred,
        "expected_microbiota_age": expected_microbiota_age,
    })
    results["relative_microbiota_age"] = (
        results["microbiota_age"] - results["expected_microbiota_age"]
    )

    return results, spline, x_sorted, y_sorted



def plot_relative_age_spline(y_true, y_pred, spline, x_sorted):
    plt.scatter(y_true, y_pred, alpha=0.4, label="Samples")
    plt.plot(x_sorted, spline(x_sorted), color="black", linewidth=2, label="Spline")
    plt.xlabel("Chronological age")
    plt.ylabel("Microbiota age")
    plt.legend()
    plt.show()



def run_regression_pipeline(dataset_label, dataset_group):
    X_train = dataset_group["X"]["train"]
    y_train = dataset_group["y"]["train"]
    X_val = dataset_group["X"]["val"]
    y_val = dataset_group["y"]["val"]

    model = build_random_forest_regressor()
    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    metrics = evaluate_regression(y_val, val_preds)
    print_metrics(metrics, dataset_label)

    plot_predicted_vs_true(
        y_val,
        val_preds,
        f"Predicted vs True Age ({dataset_label} Validation Set)",
    )

    val_results, spline, x_sorted, y_sorted = build_relative_age_results(y_val, val_preds)
    plot_relative_age_spline(y_val, val_preds, spline, x_sorted)

    return {
        "model": model,
        "metrics": metrics,
        "predictions": val_preds,
        "val_results": val_results,
        "spline": spline,
        "x_sorted": x_sorted,
        "y_sorted": y_sorted,
    }



def main():
    abs_dataset = prepare_dataset_group(DATA_PATHS["abs"], apply_log_transform=True)
    rel_dataset = prepare_dataset_group(DATA_PATHS["rel"], apply_log_transform=True)

    abs_results = run_regression_pipeline("Absolute Quant", abs_dataset)
    rel_results = run_regression_pipeline("Relative Abundance", rel_dataset)

    return {
        "abs": abs_results,
        "rel": rel_results,
    }


if __name__ == "__main__":
    pipeline_results = main()
