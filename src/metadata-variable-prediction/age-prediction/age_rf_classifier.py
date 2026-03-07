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
import pandas as pd
import numpy as np
import json
import shap

# qiime and related
import qiime2
from qiime2 import Metadata, Artifact
import biom

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# machine learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_curve,
    auc,
    roc_auc_score,
    f1_score,
)

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression


FEATURE_END_IDX = 1148
ID_COL = "original_SampleID"
AGE_COL = "age"
SEX_COL = "sex"
AGE_BIN_COL = "age_bin"
RANDOM_STATE = 42
TARGET_N = 300

DATA_PATHS = {
    "abs_train": "/ddn_scratch/k5zhao/data/classifier_training/abs_train.csv",
    "abs_test": "/ddn_scratch/k5zhao/data/classifier_training/abs_test.csv",
    "abs_val": "/ddn_scratch/k5zhao/data/classifier_training/abs_val.csv",
    "rel_train": "/ddn_scratch/k5zhao/data/classifier_training/rel_train.csv",
    "rel_test": "/ddn_scratch/k5zhao/data/classifier_training/rel_test.csv",
    "rel_val": "/ddn_scratch/k5zhao/data/classifier_training/rel_val.csv",
}

ARTIFACT_PATH = "/ddn_scratch/k5zhao/data/metaG-absquant-clean.qza"
METADATA_PATH = "/ddn_scratch/k5zhao/data/metadata_absquant_clean.tsv"
TAXONOMY_PATH = "/ddn_scratch/miter/nph-tables/wolr2-taxonomy.tsv"

FIRST_PASS_BINS = [20, 30, 40, 50, 60, 70]
RETUNED_BINS = [0, 20, 35, 50, 65, 80, np.inf]
BALANCED_BINS = [18, 30, 40, 50, 60, 70, 100]
ALLOWED_SEX = ["male", "female"]

AGE_BAC = [
    "Haemophilus_D",
    "Sutterella",
    "Akkermansia",
    "Phascolarctobacterium",
    "Ruminiclostridium_E",
    "Cloacibacillus",
    "Pseudomonas",
    "UBA1685",
    "UBA10677",
    "CAG-314",
    "CAG-313",
    "QAKW01",
]

BASE_RF_PARAMS = {
    "n_estimators": 300,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

BALANCED_RF_PARAMS = {
    "n_estimators": 500,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

GRID_RF_BASE_PARAMS = {
    "class_weight": "balanced_subsample",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

GRID_PARAM_GRID = {
    "n_estimators": [500, 800, 1000],
    "max_depth": [None, 10, 20, 40],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}


def load_reference_objects():
    artifact = qiime2.Artifact.load(ARTIFACT_PATH)
    df = artifact.view(pd.DataFrame)
    metadata = pd.read_csv(METADATA_PATH, sep="\t")

    print(df.head())
    print(df.columns[:FEATURE_END_IDX])
    print(df.index)
    print(metadata.head())

    return df, metadata


def load_datasets():
    return {name: pd.read_csv(path) for name, path in DATA_PATHS.items()}


def get_feature_columns(df):
    cols = list(df.columns[:FEATURE_END_IDX])
    return [col for col in cols if col != ID_COL]


def filter_age_range(df, min_age=20, max_age=69):
    return df.loc[(df[AGE_COL] >= min_age) & (df[AGE_COL] <= max_age)].copy()


def add_age_bin(df, bins, age_col=AGE_COL, bin_col=AGE_BIN_COL):
    out = df.copy()
    out[bin_col] = pd.cut(
        out[age_col],
        bins=bins,
        right=False,
        include_lowest=True,
    ).astype(str)
    return out


def prepare_feature_matrix(df, feature_columns, log_transform=True):
    X = df.loc[:, feature_columns].copy()
    if log_transform:
        X = np.log1p(X)
    return X


def prepare_target(df, target_col):
    return df[target_col].copy()


def prepare_split_dict(train_df, test_df, val_df, feature_columns, target_col, log_transform=True):
    return {
        "X_train": prepare_feature_matrix(train_df, feature_columns, log_transform=log_transform),
        "X_test": prepare_feature_matrix(test_df, feature_columns, log_transform=log_transform),
        "X_val": prepare_feature_matrix(val_df, feature_columns, log_transform=log_transform),
        "y_train": prepare_target(train_df, target_col),
        "y_test": prepare_target(test_df, target_col),
        "y_val": prepare_target(val_df, target_col),
    }


def fit_random_forest(X_train, y_train, params=None):
    model = RandomForestClassifier(**(params or BASE_RF_PARAMS))
    model.fit(X_train, y_train)
    return model


def fit_random_forest_grid(X_train, y_train, param_grid=None):
    estimator = RandomForestClassifier(**GRID_RF_BASE_PARAMS)
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid or GRID_PARAM_GRID,
        cv=5,
        scoring="roc_auc_ovr",
        n_jobs=-1,
        verbose=2,
    )
    grid_search.fit(X_train, y_train)
    print("Best parameters:", grid_search.best_params_)
    print("Best CV score:", grid_search.best_score_)
    return grid_search, grid_search.best_estimator_


def normalize_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return cm, cm / row_sums


def plot_confusion_heatmap(y_true, y_pred, labels, title, filename, cmap="Blues"):
    _, cm_norm = normalize_confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted age bin")
    plt.ylabel("True age bin")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename, format="png")
    plt.show()


def compute_multiclass_roc(y_true, y_pred_prob):
    y_cat = y_true.astype("category")
    y_codes = y_cat.cat.codes
    classes = list(range(len(y_cat.cat.categories)))
    y_bin = label_binarize(y_codes, classes=classes)

    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    macro_auc = roc_auc_score(y_bin, y_pred_prob, average="macro")
    return {
        "y_bin": y_bin,
        "classes": classes,
        "bin_labels": y_cat.cat.categories,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
        "macro_auc": macro_auc,
    }


def plot_roc_curves(roc_data, title, filename):
    plt.figure(figsize=(8, 6))
    for i in range(len(roc_data["classes"])):
        plt.plot(
            roc_data["fpr"][i],
            roc_data["tpr"][i],
            lw=2,
            label=f"{roc_data['bin_labels'][i]} (AUC = {roc_data['roc_auc'][i]:.2f})",
        )
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(filename, format="png")
    plt.show()


def evaluate_classifier(model, X_val, y_val, X_test, y_test, name, cm_file=None, roc_file=None, cm_title=None, roc_title=None, cm_cmap="Blues"):
    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)

    val_acc = accuracy_score(y_val, val_preds)
    test_acc = accuracy_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds, average="macro")
    roc_data = compute_multiclass_roc(y_test, test_probs)

    print(f"[{name}] Validation accuracy:", val_acc)
    print(f"[{name}] Test accuracy:", test_acc)
    print(f"[{name}] Test macro-F1:", test_f1)
    print(f"[{name}] Macro-average AUC:", roc_data["macro_auc"])
    for i in range(len(roc_data["classes"])):
        print(roc_data["bin_labels"][i], roc_data["roc_auc"][i])

    if cm_file:
        plot_confusion_heatmap(
            y_true=y_test,
            y_pred=test_preds,
            labels=model.classes_,
            title=cm_title or f"Normalized Confusion Matrix — {name}",
            filename=cm_file,
            cmap=cm_cmap,
        )

    if roc_file:
        plot_roc_curves(
            roc_data,
            title=roc_title or f"ROC Curve — {name}",
            filename=roc_file,
        )

    return {
        "model": model,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "test_f1": test_f1,
        "test_pred": test_preds,
        "test_prob": test_probs,
        "roc_data": roc_data,
    }


def compare_models_roc(abs_result, rel_result, y_abs_test, y_rel_test, filename):
    abs_codes = y_abs_test.astype("category").cat.codes
    rel_codes = y_rel_test.astype("category").cat.codes
    classes = range(len(y_abs_test.astype("category").cat.categories))

    abs_y_bin = label_binarize(abs_codes, classes=classes)
    rel_y_bin = label_binarize(rel_codes, classes=classes)

    plt.figure(figsize=(10, 6))
    bin_labels = y_abs_test.astype("category").cat.categories

    for i in classes:
        fpr_abs, tpr_abs, _ = roc_curve(abs_y_bin[:, i], abs_result["test_prob"][:, i])
        fpr_rel, tpr_rel, _ = roc_curve(rel_y_bin[:, i], rel_result["test_prob"][:, i])
        plt.plot(fpr_abs, tpr_abs, lw=2, label=f"Abs {bin_labels[i]}")
        plt.plot(fpr_rel, tpr_rel, lw=2, linestyle="--", label=f"Rel {bin_labels[i]}")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Absolute vs Relative Abundance")
    plt.legend(loc="lower right")
    plt.savefig(filename, format="png")
    plt.show()


def plot_confusion_comparison(y_abs_test, abs_pred, y_rel_test, rel_pred, filename):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_abs_test,
        abs_pred,
        normalize="true",
        cmap="Blues",
        ax=axes[0],
    )
    axes[0].set_title("Absolute Abundance")

    ConfusionMatrixDisplay.from_predictions(
        y_rel_test,
        rel_pred,
        normalize="true",
        cmap="Greens",
        ax=axes[1],
    )
    axes[1].set_title("Relative Abundance")

    plt.suptitle("Confusion Matrices for RF Classifiers")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


def stratified_bootstrap_indices(y):
    indices = []
    for c in np.unique(y):
        class_indices = np.where(y == c)[0]
        resampled = resample(class_indices, replace=True, n_samples=len(class_indices))
        indices.extend(resampled)
    return np.array(indices)


def bootstrap_metric_difference(y_abs_test, y_rel_test, abs_pred, rel_pred, abs_prob, rel_prob, n_boot=1000):
    acc_diffs = []
    auc_diffs = []

    abs_codes = y_abs_test.astype("category").cat.codes
    rel_codes = y_rel_test.astype("category").cat.codes
    classes = range(len(y_abs_test.astype("category").cat.categories))

    abs_y_bin = label_binarize(abs_codes, classes=classes)
    rel_y_bin = label_binarize(rel_codes, classes=classes)

    for _ in range(n_boot):
        indices = stratified_bootstrap_indices(abs_codes)
        acc_abs_i = accuracy_score(y_abs_test.iloc[indices], abs_pred[indices])
        acc_rel_i = accuracy_score(y_rel_test.iloc[indices], rel_pred[indices])
        acc_diffs.append(acc_abs_i - acc_rel_i)

        auc_abs_i = roc_auc_score(abs_y_bin[indices], abs_prob[indices], average="macro")
        auc_rel_i = roc_auc_score(rel_y_bin[indices], rel_prob[indices], average="macro")
        auc_diffs.append(auc_abs_i - auc_rel_i)

    acc_ci = np.percentile(acc_diffs, [2.5, 97.5])
    auc_ci = np.percentile(auc_diffs, [2.5, 97.5])

    print("Accuracy difference 95% CI (Abs - Rel):", acc_ci)
    print("Macro-AUC difference 95% CI (Abs - Rel):", auc_ci)

    if acc_ci[0] > 0 or acc_ci[1] < 0:
        print("Accuracy difference is statistically significant.")
    else:
        print("Accuracy difference is NOT statistically significant.")

    if auc_ci[0] > 0 or auc_ci[1] < 0:
        print("AUC difference is statistically significant.")
    else:
        print("AUC difference is NOT statistically significant!")

    return {"acc_ci": acc_ci, "auc_ci": auc_ci}


def get_feature_importance_df(model, feature_names):
    return pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values(by="importance", ascending=False)


def plot_top_features(feature_df, title, filename, top_n=20):
    plt.figure(figsize=(8, 6))
    plt.barh(
        feature_df["feature"].head(top_n)[::-1],
        feature_df["importance"].head(top_n)[::-1],
    )
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(filename, format="png")
    plt.show()


def print_feature_overlap(abs_feat_importance, rel_feat_importance, top_n=20):
    abs_top = set(abs_feat_importance["feature"].head(top_n))
    rel_top = set(rel_feat_importance["feature"].head(top_n))
    overlap = abs_top.intersection(rel_top)

    print("Overlap:", overlap)
    print("Number overlapping:", len(overlap))
    print("Unique to Absolute:", abs_top - rel_top)
    print("Unique to Relative:", rel_top - abs_top)


def balance_training_set(df, bin_col=AGE_BIN_COL, target_n=TARGET_N):
    return (
        df.groupby(bin_col, group_keys=False)
        .apply(lambda x: x.sample(n=min(len(x), target_n), random_state=RANDOM_STATE))
        .reset_index(drop=True)
    )


def get_taxonomy_gene_columns(taxonomy_path=TAXONOMY_PATH, age_bac=None):
    tax = pd.read_csv(taxonomy_path, sep="\t")
    taxonomy_col = "d__Bacteria; p__Firmicutes; c__Bacilli; o__Bacillales_H; f__Bacillaceae_D; g__Bacillus_S; s__Bacillus_S pseudofirmus"
    feature_id_col = "G000005825"

    tax["genus_raw"] = tax[taxonomy_col].str.extract(r"g__([^;]+)")
    tax_rel_bac = tax[tax["genus_raw"].isin(age_bac or AGE_BAC)].copy()
    col_names = list(tax_rel_bac[feature_id_col].values)
    print(col_names)
    return col_names


def prepare_augmented_features(train_df, test_df, val_df, feature_columns, fit_encoder_df, include_sex=True):
    X_train = prepare_feature_matrix(train_df, feature_columns, log_transform=True)
    X_test = prepare_feature_matrix(test_df, feature_columns, log_transform=True)
    X_val = prepare_feature_matrix(val_df, feature_columns, log_transform=True)

    if not include_sex:
        return X_train, X_test, X_val

    sex_encoder = OneHotEncoder(
        drop="first",
        handle_unknown="ignore",
        sparse_output=False,
    )
    sex_encoder.fit(fit_encoder_df[[SEX_COL]])

    train_sex = sex_encoder.transform(train_df[[SEX_COL]])
    test_sex = sex_encoder.transform(test_df[[SEX_COL]])
    val_sex = sex_encoder.transform(val_df[[SEX_COL]])

    X_train_aug = np.hstack([X_train.values, train_sex])
    X_test_aug = np.hstack([X_test.values, test_sex])
    X_val_aug = np.hstack([X_val.values, val_sex])
    return X_train_aug, X_test_aug, X_val_aug


def multiclass_macro_roc(y_true, y_proba, classes):
    y_true_bin = label_binarize(y_true, classes=classes)
    n_classes = len(classes)

    fpr = {}
    tpr = {}
    auc_per_class = []

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        auc_per_class.append(roc_auc_score(y_true_bin[:, i], y_proba[:, i]))

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes
    auc_macro = float(np.mean(auc_per_class))
    return all_fpr, mean_tpr, auc_macro


def plot_macro_auc_comparison(abs_model, rel_model, abs_X_test, rel_X_test, abs_y_test, rel_y_test, filename):
    abs_probs = abs_model.predict_proba(abs_X_test)
    rel_probs = rel_model.predict_proba(rel_X_test)
    classes = abs_model.classes_

    abs_macro_auc = roc_auc_score(abs_y_test, abs_probs, multi_class="ovr", average="macro")
    rel_macro_auc = roc_auc_score(rel_y_test, rel_probs, multi_class="ovr", average="macro")

    abs_fpr_macro, abs_tpr_macro, _ = multiclass_macro_roc(abs_y_test, abs_probs, classes)
    rel_fpr_macro, rel_tpr_macro, _ = multiclass_macro_roc(rel_y_test, rel_probs, classes)

    plt.figure(figsize=(7, 6))
    plt.plot(abs_fpr_macro, abs_tpr_macro, label=f"Absolute (Macro AUC = {abs_macro_auc:.4f})")
    plt.plot(rel_fpr_macro, rel_tpr_macro, label=f"Relative (Macro AUC = {rel_macro_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Macro ROC Curve (OvR): Absolute vs Relative (Age Bins)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

    return abs_macro_auc, rel_macro_auc


def run_first_pass(abs_train, abs_test, abs_val, rel_train, rel_test, rel_val, feature_columns):
    c_abs_train = filter_age_range(abs_train)
    c_abs_test = filter_age_range(abs_test)
    c_abs_val = filter_age_range(abs_val)
    c_rel_train = filter_age_range(rel_train)
    c_rel_test = filter_age_range(rel_test)
    c_rel_val = filter_age_range(rel_val)

    for df_name, df in [
        ("c_abs_train", c_abs_train),
        ("c_abs_test", c_abs_test),
        ("c_abs_val", c_abs_val),
        ("c_rel_train", c_rel_train),
        ("c_rel_test", c_rel_test),
        ("c_rel_val", c_rel_val),
    ]:
        print(df_name, df.shape)

    c_abs_train = add_age_bin(c_abs_train, FIRST_PASS_BINS)
    c_abs_test = add_age_bin(c_abs_test, FIRST_PASS_BINS)
    c_abs_val = add_age_bin(c_abs_val, FIRST_PASS_BINS)
    c_rel_train = add_age_bin(c_rel_train, FIRST_PASS_BINS)
    c_rel_test = add_age_bin(c_rel_test, FIRST_PASS_BINS)
    c_rel_val = add_age_bin(c_rel_val, FIRST_PASS_BINS)

    abs_data = prepare_split_dict(c_abs_train, c_abs_test, c_abs_val, feature_columns, AGE_BIN_COL)
    rel_data = prepare_split_dict(c_rel_train, c_rel_test, c_rel_val, feature_columns, AGE_BIN_COL)

    abs_rf = fit_random_forest(abs_data["X_train"], abs_data["y_train"], BASE_RF_PARAMS)
    rel_rf = fit_random_forest(rel_data["X_train"], rel_data["y_train"], BASE_RF_PARAMS)

    abs_result = evaluate_classifier(
        abs_rf,
        abs_data["X_val"],
        abs_data["y_val"],
        abs_data["X_test"],
        abs_data["y_test"],
        name="First-pass Absolute RF",
        cm_file="first_pass_abs_rf_cm.png",
        roc_file="first_pass_abs_rf_roc.png",
        roc_title="ROC Curve — Absolute Abundance Model",
    )

    rel_result = evaluate_classifier(
        rel_rf,
        rel_data["X_val"],
        rel_data["y_val"],
        rel_data["X_test"],
        rel_data["y_test"],
        name="First-pass Relative RF",
        cm_file="first_pass_rel_rf_cm.png",
        roc_file="first_pass_rel_rf_roc.png",
        roc_title="ROC Curve — Relative Abundance Model",
    )

    return abs_result, rel_result


def run_retuned_pass(abs_train, abs_test, abs_val, rel_train, rel_test, rel_val, feature_columns):
    abs_train_bin = add_age_bin(abs_train, RETUNED_BINS)
    abs_test_bin = add_age_bin(abs_test, RETUNED_BINS)
    abs_val_bin = add_age_bin(abs_val, RETUNED_BINS)
    rel_train_bin = add_age_bin(rel_train, RETUNED_BINS)
    rel_test_bin = add_age_bin(rel_test, RETUNED_BINS)
    rel_val_bin = add_age_bin(rel_val, RETUNED_BINS)

    abs_data = prepare_split_dict(abs_train_bin, abs_test_bin, abs_val_bin, feature_columns, AGE_BIN_COL)
    rel_data = prepare_split_dict(rel_train_bin, rel_test_bin, rel_val_bin, feature_columns, AGE_BIN_COL)

    abs_rf = fit_random_forest(abs_data["X_train"], abs_data["y_train"], BASE_RF_PARAMS)
    rel_rf = fit_random_forest(rel_data["X_train"], rel_data["y_train"], BASE_RF_PARAMS)

    abs_result = evaluate_classifier(
        abs_rf,
        abs_data["X_val"],
        abs_data["y_val"],
        abs_data["X_test"],
        abs_data["y_test"],
        name="Retuned Absolute RF",
        cm_file="retuned_abs_rf_cm.png",
        roc_file="retuned_abs_rf_roc.png",
        roc_title="ROC Curve — Absolute Abundance Model",
    )

    rel_result = evaluate_classifier(
        rel_rf,
        rel_data["X_val"],
        rel_data["y_val"],
        rel_data["X_test"],
        rel_data["y_test"],
        name="Retuned Relative RF",
        cm_file="retuned_rel_rf_cm.png",
        roc_file="retuned_rel_rf_roc.png",
        roc_title="ROC Curve — Relative Abundance Model",
    )

    return abs_result, rel_result


def run_train_plus_val_comparison(abs_train, abs_test, abs_val, rel_train, rel_test, rel_val, feature_columns):
    abs_train_all = pd.concat([abs_train, abs_val], ignore_index=True)
    rel_train_all = pd.concat([rel_train, rel_val], ignore_index=True)

    abs_train_all = add_age_bin(abs_train_all, RETUNED_BINS)
    abs_test_bin = add_age_bin(abs_test, RETUNED_BINS)
    rel_train_all = add_age_bin(rel_train_all, RETUNED_BINS)
    rel_test_bin = add_age_bin(rel_test, RETUNED_BINS)

    abs_X_train = prepare_feature_matrix(abs_train_all, feature_columns, log_transform=True)
    abs_y_train = prepare_target(abs_train_all, AGE_BIN_COL)
    abs_X_test = prepare_feature_matrix(abs_test_bin, feature_columns, log_transform=True)
    abs_y_test = prepare_target(abs_test_bin, AGE_BIN_COL)

    rel_X_train = prepare_feature_matrix(rel_train_all, feature_columns, log_transform=True)
    rel_y_train = prepare_target(rel_train_all, AGE_BIN_COL)
    rel_X_test = prepare_feature_matrix(rel_test_bin, feature_columns, log_transform=True)
    rel_y_test = prepare_target(rel_test_bin, AGE_BIN_COL)

    abs_rf = fit_random_forest(abs_X_train, abs_y_train, BASE_RF_PARAMS)
    rel_rf = fit_random_forest(rel_X_train, rel_y_train, BASE_RF_PARAMS)

    abs_result = evaluate_classifier(abs_rf, abs_X_test, abs_y_test, abs_X_test, abs_y_test, "Final Absolute RF")
    rel_result = evaluate_classifier(rel_rf, rel_X_test, rel_y_test, rel_X_test, rel_y_test, "Final Relative RF")

    compare_models_roc(abs_result, rel_result, abs_y_test, rel_y_test, "final_rf_roc.png")

    ConfusionMatrixDisplay.from_predictions(abs_y_test, abs_result["test_pred"], normalize="true", cmap="Blues")
    plt.xticks(rotation=45)
    plt.show()

    ConfusionMatrixDisplay.from_predictions(rel_y_test, rel_result["test_pred"], normalize="true", cmap="Greens")
    plt.xticks(rotation=45)
    plt.savefig("final_rf_cm.png", format="png")
    plt.show()

    bootstrap_metric_difference(
        abs_y_test,
        rel_y_test,
        abs_result["test_pred"],
        rel_result["test_pred"],
        abs_result["test_prob"],
        rel_result["test_prob"],
    )

    abs_feat_importance = get_feature_importance_df(abs_rf, feature_columns)
    rel_feat_importance = get_feature_importance_df(rel_rf, feature_columns)
    plot_top_features(abs_feat_importance, "Top 20 Absolute Abundance Features", "final_abs_rf_top20_features.png")
    plot_top_features(rel_feat_importance, "Top 20 Relative Abundance Features", "final_rel_rf_top20_features.png")
    print_feature_overlap(abs_feat_importance, rel_feat_importance)

    return {
        "abs_rf": abs_rf,
        "rel_rf": rel_rf,
        "abs_y_test": abs_y_test,
        "rel_y_test": rel_y_test,
        "abs_X_test": abs_X_test,
        "rel_X_test": rel_X_test,
        "abs_feature_importance": abs_feat_importance,
        "rel_feature_importance": rel_feat_importance,
    }


def run_balanced_experiments(abs_train, abs_test, abs_val, rel_train, rel_test, rel_val, feature_columns):
    abs_train_bin = add_age_bin(abs_train, BALANCED_BINS)
    abs_test_bin = add_age_bin(abs_test, BALANCED_BINS)
    abs_val_bin = add_age_bin(abs_val, BALANCED_BINS)
    rel_train_bin = add_age_bin(rel_train, BALANCED_BINS)
    rel_test_bin = add_age_bin(rel_test, BALANCED_BINS)
    rel_val_bin = add_age_bin(rel_val, BALANCED_BINS)

    abs_balanced_train = balance_training_set(abs_train_bin)
    rel_balanced_train = balance_training_set(rel_train_bin)

    abs_data = prepare_split_dict(abs_balanced_train, abs_test_bin, abs_val_bin, feature_columns, AGE_BIN_COL)
    rel_data = prepare_split_dict(rel_balanced_train, rel_test_bin, rel_val_bin, feature_columns, AGE_BIN_COL)

    abs_grid, abs_best_rf = fit_random_forest_grid(abs_data["X_train"], abs_data["y_train"])
    rel_grid, rel_best_rf = fit_random_forest_grid(rel_data["X_train"], rel_data["y_train"])

    abs_grid_result = evaluate_classifier(abs_best_rf, abs_data["X_val"], abs_data["y_val"], abs_data["X_test"], abs_data["y_test"], "Balanced Absolute RF (grid)")
    rel_grid_result = evaluate_classifier(rel_best_rf, rel_data["X_val"], rel_data["y_val"], rel_data["X_test"], rel_data["y_test"], "Balanced Relative RF (grid)")

    abs_rf_nogrid = fit_random_forest(abs_data["X_train"], abs_data["y_train"], BALANCED_RF_PARAMS)
    rel_rf_nogrid = fit_random_forest(rel_data["X_train"], rel_data["y_train"], BALANCED_RF_PARAMS)

    abs_nogrid_result = evaluate_classifier(abs_rf_nogrid, abs_data["X_val"], abs_data["y_val"], abs_data["X_test"], abs_data["y_test"], "Balanced Absolute RF (no grid)")
    rel_nogrid_result = evaluate_classifier(rel_rf_nogrid, rel_data["X_val"], rel_data["y_val"], rel_data["X_test"], rel_data["y_test"], "Balanced Relative RF (no grid)")

    return {
        "abs_balanced_train": abs_balanced_train,
        "rel_balanced_train": rel_balanced_train,
        "abs_test_bin": abs_test_bin,
        "rel_test_bin": rel_test_bin,
        "abs_val_bin": abs_val_bin,
        "rel_val_bin": rel_val_bin,
        "abs_best_rf": abs_best_rf,
        "rel_best_rf": rel_best_rf,
        "abs_rf_nogrid": abs_rf_nogrid,
        "rel_rf_nogrid": rel_rf_nogrid,
        "abs_grid_result": abs_grid_result,
        "rel_grid_result": rel_grid_result,
        "abs_nogrid_result": abs_nogrid_result,
        "rel_nogrid_result": rel_nogrid_result,
    }


def run_gene_and_sex_experiments(abs_train, abs_test, abs_val, rel_train, rel_test, rel_val):
    abs_train_bin = add_age_bin(abs_train, BALANCED_BINS)
    abs_test_bin = add_age_bin(abs_test, BALANCED_BINS)
    abs_val_bin = add_age_bin(abs_val, BALANCED_BINS)
    rel_train_bin = add_age_bin(rel_train, BALANCED_BINS)
    rel_test_bin = add_age_bin(rel_test, BALANCED_BINS)
    rel_val_bin = add_age_bin(rel_val, BALANCED_BINS)

    abs_balanced_train = balance_training_set(abs_train_bin)
    rel_balanced_train = balance_training_set(rel_train_bin)

    gene_columns = get_taxonomy_gene_columns()

    abs_gene_columns = [col for col in gene_columns if col in abs_balanced_train.columns]
    rel_gene_columns = [col for col in gene_columns if col in rel_balanced_train.columns]

    abs_X_train_aug, abs_X_test_aug, abs_X_val_aug = prepare_augmented_features(
        abs_balanced_train, abs_test_bin, abs_val_bin, abs_gene_columns, abs_train_bin, include_sex=True
    )
    rel_X_train_aug, rel_X_test_aug, rel_X_val_aug = prepare_augmented_features(
        rel_balanced_train, rel_test_bin, rel_val_bin, rel_gene_columns, abs_train_bin, include_sex=True
    )

    abs_y_train = abs_balanced_train[AGE_BIN_COL]
    abs_y_test = abs_test_bin[AGE_BIN_COL]
    abs_y_val = abs_val_bin[AGE_BIN_COL]
    rel_y_train = rel_balanced_train[AGE_BIN_COL]
    rel_y_test = rel_test_bin[AGE_BIN_COL]
    rel_y_val = rel_val_bin[AGE_BIN_COL]

    rel_rf_sex_gene = fit_random_forest(rel_X_train_aug, rel_y_train, BALANCED_RF_PARAMS)
    rel_rf_sex_gene_result = evaluate_classifier(
        rel_rf_sex_gene,
        rel_X_val_aug,
        rel_y_val,
        rel_X_test_aug,
        rel_y_test,
        name="Relative RF + Sex + Gene",
        cm_file="age_abs_rf_sex_gene_cm.png",
        cm_title="Normalized Confusion Matrix (Row-wise) for Relative Abundance RF Using Sex and Genes as Features",
    )

    abs_rf_sex_gene = fit_random_forest(abs_X_train_aug, abs_y_train, BALANCED_RF_PARAMS)
    abs_rf_sex_gene_result = evaluate_classifier(
        abs_rf_sex_gene,
        abs_X_val_aug,
        abs_y_val,
        abs_X_test_aug,
        abs_y_test,
        name="Absolute RF + Sex + Gene",
        cm_file="age_rel_rf_sex_gene_cm.png",
        cm_title="Normalized Confusion Matrix (Row-wise) for Absolute Abundance RF Using Sex and Genes as Features",
    )

    return {
        "abs_rf_sex_gene_result": abs_rf_sex_gene_result,
        "rel_rf_sex_gene_result": rel_rf_sex_gene_result,
        "gene_columns": gene_columns,
    }


def run_sex_stratified_experiments(abs_train, abs_test, abs_val, rel_train, rel_test, rel_val, gene_columns):
    abs_train = abs_train[abs_train[SEX_COL].isin(ALLOWED_SEX)].copy()
    abs_test = abs_test[abs_test[SEX_COL].isin(ALLOWED_SEX)].copy()
    abs_val = abs_val[abs_val[SEX_COL].isin(ALLOWED_SEX)].copy()
    rel_train = rel_train[rel_train[SEX_COL].isin(ALLOWED_SEX)].copy()
    rel_test = rel_test[rel_test[SEX_COL].isin(ALLOWED_SEX)].copy()
    rel_val = rel_val[rel_val[SEX_COL].isin(ALLOWED_SEX)].copy()

    abs_train_bin = add_age_bin(abs_train, BALANCED_BINS)
    abs_test_bin = add_age_bin(abs_test, BALANCED_BINS)
    abs_val_bin = add_age_bin(abs_val, BALANCED_BINS)
    rel_train_bin = add_age_bin(rel_train, BALANCED_BINS)
    rel_test_bin = add_age_bin(rel_test, BALANCED_BINS)
    rel_val_bin = add_age_bin(rel_val, BALANCED_BINS)

    abs_balanced_train = balance_training_set(abs_train_bin)
    rel_balanced_train = balance_training_set(rel_train_bin)

    abs_gene_columns = [col for col in gene_columns if col in abs_balanced_train.columns]
    rel_gene_columns = [col for col in gene_columns if col in rel_balanced_train.columns]

    abs_X_train_aug, abs_X_test_aug, abs_X_val_aug = prepare_augmented_features(
        abs_balanced_train, abs_test_bin, abs_val_bin, abs_gene_columns, abs_train_bin, include_sex=True
    )
    rel_X_train_aug, rel_X_test_aug, rel_X_val_aug = prepare_augmented_features(
        rel_balanced_train, rel_test_bin, rel_val_bin, rel_gene_columns, abs_train_bin, include_sex=True
    )

    abs_y_train = abs_balanced_train[AGE_BIN_COL]
    abs_y_test = abs_test_bin[AGE_BIN_COL]
    abs_y_val = abs_val_bin[AGE_BIN_COL]
    rel_y_train = rel_balanced_train[AGE_BIN_COL]
    rel_y_test = rel_test_bin[AGE_BIN_COL]
    rel_y_val = rel_val_bin[AGE_BIN_COL]

    abs_rf_sex = fit_random_forest(abs_X_train_aug, abs_y_train, BALANCED_RF_PARAMS)
    abs_rf_sex_result = evaluate_classifier(
        abs_rf_sex,
        abs_X_val_aug,
        abs_y_val,
        abs_X_test_aug,
        abs_y_test,
        name="Absolute RF + Sex",
        cm_file="age_abs_rf_sex_cm.png",
        cm_title="Normalized Confusion Matrix (Row-wise) for Absolute Abundance RF with Sex as a Feature",
    )

    rel_rf_sex = fit_random_forest(rel_X_train_aug, rel_y_train, BALANCED_RF_PARAMS)
    rel_rf_sex_result = evaluate_classifier(
        rel_rf_sex,
        rel_X_val_aug,
        rel_y_val,
        rel_X_test_aug,
        rel_y_test,
        name="Relative RF + Sex",
        cm_file="age_rel_rf_sex_cm.png",
        cm_title="Normalized Confusion Matrix (Row-wise) for Relative Abundance RF with Sex as a Feature",
    )

    return {
        "abs_rf_sex_result": abs_rf_sex_result,
        "rel_rf_sex_result": rel_rf_sex_result,
        "abs_y_test": abs_y_test,
        "rel_y_test": rel_y_test,
        "abs_X_test_aug": abs_X_test_aug,
        "rel_X_test_aug": rel_X_test_aug,
        "abs_rf_sex": abs_rf_sex,
        "rel_rf_sex": rel_rf_sex,
    }


def run_final_balanced_comparison(abs_train, abs_test, abs_val, rel_train, rel_test, rel_val, feature_columns):
    abs_train_bin = add_age_bin(abs_train, BALANCED_BINS)
    abs_test_bin = add_age_bin(abs_test, BALANCED_BINS)
    abs_val_bin = add_age_bin(abs_val, BALANCED_BINS)
    rel_train_bin = add_age_bin(rel_train, BALANCED_BINS)
    rel_test_bin = add_age_bin(rel_test, BALANCED_BINS)
    rel_val_bin = add_age_bin(rel_val, BALANCED_BINS)

    abs_balanced_train = balance_training_set(abs_train_bin)
    rel_balanced_train = balance_training_set(rel_train_bin)

    abs_data = prepare_split_dict(abs_balanced_train, abs_test_bin, abs_val_bin, feature_columns, AGE_BIN_COL)
    rel_data = prepare_split_dict(rel_balanced_train, rel_test_bin, rel_val_bin, feature_columns, AGE_BIN_COL)

    abs_rf_nogrid = fit_random_forest(abs_data["X_train"], abs_data["y_train"], BALANCED_RF_PARAMS)
    rel_rf_nogrid = fit_random_forest(rel_data["X_train"], rel_data["y_train"], BALANCED_RF_PARAMS)

    abs_pred = abs_rf_nogrid.predict(abs_data["X_test"])
    rel_pred = rel_rf_nogrid.predict(rel_data["X_test"])

    plot_confusion_comparison(abs_data["y_test"], abs_pred, rel_data["y_test"], rel_pred, "balanced_rf_cm_comparison.png")
    plot_macro_auc_comparison(
        abs_rf_nogrid,
        rel_rf_nogrid,
        abs_data["X_test"],
        rel_data["X_test"],
        abs_data["y_test"],
        rel_data["y_test"],
        "rf_macro_auc_comparison.png",
    )

    bootstrap_metric_difference(
        abs_data["y_test"],
        rel_data["y_test"],
        abs_pred,
        rel_pred,
        abs_rf_nogrid.predict_proba(abs_data["X_test"]),
        rel_rf_nogrid.predict_proba(rel_data["X_test"]),
    )

    abs_feat_importance = get_feature_importance_df(abs_rf_nogrid, feature_columns)
    rel_feat_importance = get_feature_importance_df(rel_rf_nogrid, feature_columns)

    plot_top_features(abs_feat_importance, "Top 20 Absolute Abundance Features", "abs_rf_nogrid_top20_features.png")
    plot_top_features(rel_feat_importance, "Top 20 Relative Abundance Features", "rel_rf_nogrid_top20_features.png")
    print_feature_overlap(abs_feat_importance, rel_feat_importance)

    return {
        "abs_rf_nogrid": abs_rf_nogrid,
        "rel_rf_nogrid": rel_rf_nogrid,
        "abs_X_test_log": abs_data["X_test"],
        "rel_X_test_log": rel_data["X_test"],
    }


def run_shap_analysis(model, X_test, title, filename):
    X = X_test.copy()
    if hasattr(model, "feature_names_in_"):
        X = X[model.feature_names_in_]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    sv = shap_values.values
    if sv.ndim == 3:
        class_idx = 1
        sv = sv[:, :, class_idx]

    plt.figure(figsize=(8, 6))
    shap.summary_plot(
        sv,
        X,
        plot_type="bar",
        max_display=20,
        show=False,
    )
    ax = plt.gca()
    ax.set_title(title, fontsize=14, pad=12)
    plt.tight_layout()
    plt.savefig(filename, format="png")
    plt.show()


def print_final_comparison_table(abs_grid_result, rel_grid_result, abs_nogrid_result, rel_nogrid_result, abs_rf_sex_result, rel_rf_sex_result, abs_rf_sex_gene_result, rel_rf_sex_gene_result):
    print("TEST ACCURACY COMPARISON")
    print(f"{'Model':<35}{'Absolute':>12}{'Relative':>12}")
    print("-" * 60)
    print(f"{'Balanced RF (grid)':<35}{abs_grid_result['test_accuracy']:>12.4f}{rel_grid_result['test_accuracy']:>12.4f}")
    print(f"{'Balanced RF (no grid)':<35}{abs_nogrid_result['test_accuracy']:>12.4f}{rel_nogrid_result['test_accuracy']:>12.4f}")
    print(f"{'RF + Sex':<35}{abs_rf_sex_result['test_accuracy']:>12.4f}{rel_rf_sex_result['test_accuracy']:>12.4f}")
    print(f"{'RF + Sex + Gene':<35}{abs_rf_sex_gene_result['test_accuracy']:>12.4f}{rel_rf_sex_gene_result['test_accuracy']:>12.4f}")
    print("\n=====================================================\n")


def main():
    load_reference_objects()
    datasets = load_datasets()

    abs_train = datasets["abs_train"]
    abs_test = datasets["abs_test"]
    abs_val = datasets["abs_val"]
    rel_train = datasets["rel_train"]
    rel_test = datasets["rel_test"]
    rel_val = datasets["rel_val"]

    feature_columns = get_feature_columns(abs_train)
    print(abs_train.head())
    print(abs_train.columns[:FEATURE_END_IDX])

    run_first_pass(abs_train, abs_test, abs_val, rel_train, rel_test, rel_val, feature_columns)
    run_retuned_pass(abs_train, abs_test, abs_val, rel_train, rel_test, rel_val, feature_columns)
    run_train_plus_val_comparison(abs_train, abs_test, abs_val, rel_train, rel_test, rel_val, feature_columns)

    balanced_results = run_balanced_experiments(abs_train, abs_test, abs_val, rel_train, rel_test, rel_val, feature_columns)
    gene_results = run_gene_and_sex_experiments(abs_train, abs_test, abs_val, rel_train, rel_test, rel_val)
    sex_results = run_sex_stratified_experiments(
        abs_train,
        abs_test,
        abs_val,
        rel_train,
        rel_test,
        rel_val,
        gene_results["gene_columns"],
    )

    print_final_comparison_table(
        balanced_results["abs_grid_result"],
        balanced_results["rel_grid_result"],
        balanced_results["abs_nogrid_result"],
        balanced_results["rel_nogrid_result"],
        sex_results["abs_rf_sex_result"],
        sex_results["rel_rf_sex_result"],
        gene_results["abs_rf_sex_gene_result"],
        gene_results["rel_rf_sex_gene_result"],
    )

    shap_results = run_final_balanced_comparison(abs_train, abs_test, abs_val, rel_train, rel_test, rel_val, feature_columns)
    run_shap_analysis(shap_results["abs_rf_nogrid"], shap_results["abs_X_test_log"], "SHAP Values - Absolute Abundance", "abs_rf_nogrid_SHAP.png")
    run_shap_analysis(shap_results["rel_rf_nogrid"], shap_results["rel_X_test_log"], "SHAP Values - Relative Abundance", "rel_rf_nogrid_SHAP.png")


if __name__ == "__main__":
    main()
