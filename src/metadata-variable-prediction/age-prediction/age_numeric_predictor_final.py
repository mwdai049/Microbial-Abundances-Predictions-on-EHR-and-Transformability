# Filepath imports
# Making sure filepath runs smoothly
import sys
import os

print(sys.executable)

env_root = sys.prefix
env_bin = os.path.join(env_root, "bin")
os.environ["PATH"] = env_bin + os.pathsep + os.environ.get("PATH", "")

# imports for data analysis
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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score, roc_curve, auc, roc_auc_score
from sklearn.svm import SVR

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score

from scipy.interpolate import UnivariateSpline


# ---------------------------
# Constants
# ---------------------------
DATA_DIR = "/ddn_scratch/k5zhao/data/classifier_training"
TAXONOMY_PATH = "/ddn_scratch/miter/nph-tables/wolr2-taxonomy.tsv"
RANDOM_STATE = 42
TARGET_N = 329
AGE_BINS = [18, 30, 40, 50, 60, 70, 100]
VALID_SEXES = ["male", "female"]
PRIMARY_FEATURE_END = 1148
TARGET_COL = "age"
SEX_COL = "sex"
ID_COL = "original_SampleID"
AGE_BIN_COL = "age_bin"


# ---------------------------
# Data loading and preprocessing
# ---------------------------
def load_split_datasets():
    datasets = {}
    for representation in ["abs", "rel"]:
        for split in ["train", "test", "val"]:
            key = f"{representation}_{split}"
            path = os.path.join(DATA_DIR, f"{key}.csv")
            datasets[key] = pd.read_csv(path, low_memory=False)
    return datasets



def filter_to_binary_sex(df):
    return df[df[SEX_COL].isin(VALID_SEXES)].copy()



def add_age_bin(df, bins):
    out = df.copy()
    out[AGE_BIN_COL] = pd.cut(
        out[TARGET_COL],
        bins=bins,
        right=False,
        include_lowest=True
    ).astype(str)
    return out



def prepare_base_datasets(datasets):
    prepared = {}
    for name, df in datasets.items():
        prepared[name] = add_age_bin(filter_to_binary_sex(df), AGE_BINS)
    return prepared



def balance_training_set(df, target_n):
    return (
        df.groupby(AGE_BIN_COL, group_keys=False)
        .apply(lambda group: group.sample(n=min(len(group), target_n), random_state=RANDOM_STATE))
        .reset_index(drop=True)
    )



def get_primary_feature_columns(reference_df):
    feature_cols = list(reference_df.columns[:PRIMARY_FEATURE_END])
    return [col for col in feature_cols if col != ID_COL]



def get_taxa_feature_columns():
    tax = pd.read_csv(TAXONOMY_PATH, sep="\t")
    taxonomy_col = "d__Bacteria; p__Firmicutes; c__Bacilli; o__Bacillales_H; f__Bacillaceae_D; g__Bacillus_S; s__Bacillus_S pseudofirmus"
    mapping_col = "G000005825"
    age_bac = [
        "Haemophilus_D", "Sutterella", "Akkermansia", "Phascolarctobacterium",
        "Ruminiclostridium_E", "Cloacibacillus", "Pseudomonas", "UBA1685",
        "UBA10677", "CAG-314", "CAG-313", "QAKW01"
    ]

    tax = tax.copy()
    tax["genus_raw"] = tax[taxonomy_col].str.extract(r"g__([^;]+)")
    return tax.loc[tax["genus_raw"].isin(age_bac), mapping_col].dropna().tolist()



def fit_sex_encoder(train_df):
    encoder = OneHotEncoder(
        drop="first",
        handle_unknown="ignore",
        sparse_output=False
    )
    encoder.fit(train_df[[SEX_COL]])
    return encoder



def build_feature_matrix(df, feature_columns, sex_encoder):
    x_numeric = np.log1p(df.loc[:, feature_columns].copy())
    x_sex = sex_encoder.transform(df[[SEX_COL]])
    x = np.hstack([x_numeric.values, x_sex])
    y = df[TARGET_COL]
    return x, y



def prepare_representation_data(prepared_datasets, representation, feature_columns, sex_encoder):
    train_df = prepared_datasets[f"{representation}_train"]
    test_df = prepared_datasets[f"{representation}_test"]
    val_df = prepared_datasets[f"{representation}_val"]

    if representation == "abs":
        balanced_train_df = balance_training_set(train_df, TARGET_N)
    else:
        balanced_train_df = balance_training_set(train_df, TARGET_N)

    available_feature_columns = [col for col in feature_columns if col in balanced_train_df.columns]

    x_train, y_train = build_feature_matrix(balanced_train_df, available_feature_columns, sex_encoder)
    x_test, y_test = build_feature_matrix(test_df, available_feature_columns, sex_encoder)
    x_val, y_val = build_feature_matrix(val_df, available_feature_columns, sex_encoder)

    return {
        "train_df": balanced_train_df,
        "test_df": test_df,
        "val_df": val_df,
        "feature_columns": available_feature_columns,
        "X_train": x_train,
        "Y_train": y_train,
        "X_test": x_test,
        "Y_test": y_test,
        "X_val": x_val,
        "Y_val": y_val,
    }


# ---------------------------
# Modeling and evaluation
# ---------------------------
def compute_regression_metrics(y_true, y_pred):
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
    }



def print_metrics(prefix, metrics):
    print(f"{prefix} R²:", metrics["r2"])
    print(f"{prefix} RMSE:", metrics["rmse"])
    print(f"{prefix} MAE:", metrics["mae"])



def train_random_forest(x_train, y_train):
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(x_train, y_train)
    return model



def train_gradient_boosting(x_train, y_train):
    model = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE
    )
    model.fit(x_train, y_train)
    return model



def train_svr_with_grid_search(x_train, y_train):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf"))
    ])

    param_grid = {
        "svr__C": [0.1, 1, 10, 100],
        "svr__gamma": ["scale", 0.01, 0.1, 1],
        "svr__epsilon": [0.01, 0.1, 0.5]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(x_train, y_train)
    print("Best parameters:", grid_search.best_params_)
    print("Best CV R²:", grid_search.best_score_)
    return grid_search.best_estimator_



def evaluate_model(model, dataset_bundle, label):
    val_pred = model.predict(dataset_bundle["X_val"])
    test_pred = model.predict(dataset_bundle["X_test"])

    val_metrics = compute_regression_metrics(dataset_bundle["Y_val"], val_pred)
    test_metrics = compute_regression_metrics(dataset_bundle["Y_test"], test_pred)

    print_metrics(f"{label} Validation", val_metrics)
    print_metrics(f"{label} Test", test_metrics)

    return {
        "model": model,
        "val_pred": val_pred,
        "test_pred": test_pred,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }



def plot_prediction_comparison(y_true_abs, pred_abs, y_true_rel, pred_rel, title, abs_title, rel_title, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    min_val = min(y_true_abs.min(), y_true_rel.min())
    max_val = max(y_true_abs.max(), y_true_rel.max())
    x_line = np.linspace(min_val, max_val, 100)

    axes[0].scatter(y_true_abs, pred_abs, alpha=0.5)
    axes[0].plot(x_line, x_line)
    axes[0].set_title(f"{abs_title}\nR² = {r2_score(y_true_abs, pred_abs):.3f}")
    axes[0].set_xlabel("True Age")
    axes[0].set_ylabel("Predicted Age")

    axes[1].scatter(y_true_rel, pred_rel, alpha=0.5)
    axes[1].plot(x_line, x_line)
    axes[1].set_title(f"{rel_title}\nR² = {r2_score(y_true_rel, pred_rel):.3f}")
    axes[1].set_xlabel("True Age")
    axes[1].set_ylabel("Predicted Age")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, format="png")
    plt.show()



def bootstrap_r2_difference(y_true, pred_abs, pred_rel, n_boot=5000):
    rng = np.random.default_rng(RANDOM_STATE)
    diffs = []
    n = len(y_true)

    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        y_sample = y_true.iloc[idx]
        abs_sample = pred_abs[idx]
        rel_sample = pred_rel[idx]
        diffs.append(r2_score(y_sample, rel_sample) - r2_score(y_sample, abs_sample))

    diffs = np.array(diffs)
    mean_diff = diffs.mean()
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    p_value = np.mean(diffs <= 0)

    print("Mean ΔR²:", mean_diff)
    print("95% CI:", ci_lower, "to", ci_upper)
    print("P-value:", p_value)

    return {
        "mean_diff": mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
    }



def run_paired_experiment(model_name, trainer, abs_bundle, rel_bundle, plot_config):
    print(f"\n===== {model_name}: Absolute =====")
    abs_model = trainer(abs_bundle["X_train"], abs_bundle["Y_train"])
    abs_result = evaluate_model(abs_model, abs_bundle, f"{model_name} Absolute")

    print(f"\n===== {model_name}: Relative =====")
    rel_model = trainer(rel_bundle["X_train"], rel_bundle["Y_train"])
    rel_result = evaluate_model(rel_model, rel_bundle, f"{model_name} Relative")

    plot_prediction_comparison(
        abs_bundle["Y_test"],
        abs_result["test_pred"],
        rel_bundle["Y_test"],
        rel_result["test_pred"],
        plot_config["title"],
        plot_config["abs_title"],
        plot_config["rel_title"],
        plot_config["output_path"],
    )

    bootstrap_stats = bootstrap_r2_difference(
        abs_bundle["Y_test"],
        abs_result["test_pred"],
        rel_result["test_pred"],
    )

    return {
        "abs": abs_result,
        "rel": rel_result,
        "bootstrap": bootstrap_stats,
    }



def summarize_results_row(representation, model, result):
    return {
        "Task": "Regression",
        "Target": TARGET_COL,
        "Representation": representation,
        "Model": model,
        "Val_MAE": result["val_metrics"]["mae"],
        "Test_MAE": result["test_metrics"]["mae"],
        "Val_R2": result["val_metrics"]["r2"],
        "Test_R2": result["test_metrics"]["r2"],
        "Val_RMSE": result["val_metrics"]["rmse"],
        "Test_RMSE": result["test_metrics"]["rmse"],
    }



def build_results_table(experiment_outputs):
    rows = []
    for experiment_name, output in experiment_outputs.items():
        rows.append(summarize_results_row("Absolute", experiment_name, output["abs"]))
        rows.append(summarize_results_row("Relative", experiment_name, output["rel"]))
    results_df = pd.DataFrame(rows).round(6)
    return results_df


# ---------------------------
# Main workflow
# ---------------------------
def main():
    datasets = load_split_datasets()
    prepared_datasets = prepare_base_datasets(datasets)

    sex_encoder = fit_sex_encoder(prepared_datasets["abs_train"])

    primary_feature_columns = get_primary_feature_columns(prepared_datasets["abs_train"])
    taxa_feature_columns = get_taxa_feature_columns()

    abs_primary = prepare_representation_data(prepared_datasets, "abs", primary_feature_columns, sex_encoder)
    rel_primary = prepare_representation_data(prepared_datasets, "rel", primary_feature_columns, sex_encoder)

    abs_taxa = prepare_representation_data(prepared_datasets, "abs", taxa_feature_columns, sex_encoder)
    rel_taxa = prepare_representation_data(prepared_datasets, "rel", taxa_feature_columns, sex_encoder)

    experiment_outputs = {}

    experiment_outputs["RandomForest"] = run_paired_experiment(
        model_name="RandomForest",
        trainer=train_random_forest,
        abs_bundle=abs_primary,
        rel_bundle=rel_primary,
        plot_config={
            "title": "Absolute vs. Relative Abundance Random Forest Regressor Results",
            "abs_title": "Absolute Abundance RF",
            "rel_title": "Relative Abundance RF",
            "output_path": "rf_reg_comparison.png",
        },
    )

    experiment_outputs["GBR"] = run_paired_experiment(
        model_name="GBR",
        trainer=train_gradient_boosting,
        abs_bundle=abs_primary,
        rel_bundle=rel_primary,
        plot_config={
            "title": "Absolute vs. Relative Abundance Gradient Boosted Regressor Results",
            "abs_title": "Absolute Abundance",
            "rel_title": "Relative Abundance",
            "output_path": "gbr_comparison.png",
        },
    )

    experiment_outputs["SVM_RBF"] = run_paired_experiment(
        model_name="SVM_RBF",
        trainer=train_svr_with_grid_search,
        abs_bundle=abs_primary,
        rel_bundle=rel_primary,
        plot_config={
            "title": "Absolute vs. Relative Abundance RBF SVM Results",
            "abs_title": "Absolute Abundance RF",
            "rel_title": "Relative Abundance RF",
            "output_path": "rbf_comparison.png",
        },
    )

    experiment_outputs["RandomForest_Taxa"] = run_paired_experiment(
        model_name="RandomForest_Taxa",
        trainer=train_random_forest,
        abs_bundle=abs_taxa,
        rel_bundle=rel_taxa,
        plot_config={
            "title": "Absolute vs. Relative Abundance with Taxa as Features",
            "abs_title": "Absolute Abundance",
            "rel_title": "Relative Abundance",
            "output_path": "rf_taxa_comparison.png",
        },
    )

    results_df = build_results_table(experiment_outputs)
    print(results_df)
    results_df.to_csv("age_reg_model_comparisons.csv", index=False)


if __name__ == "__main__":
    main()
