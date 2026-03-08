#!/usr/bin/env python3
"""
SHAP Analysis Pipeline for HGB Models

Runs SHAP on saved HistGradientBoosting (HGB) models for:
    1. BMI regression
    2. BMI classification
    3. Stool quality (Bowel Movement) classification

Assumes the three training pipelines have already been run and that
their trained models were saved to disk.

Saved outputs:
    SHAP_DIR/
        regression_bmi_Absolute_HGB_beeswarm.png
        regression_bmi_Absolute_HGB_bar.png
        regression_bmi_Relative_HGB_beeswarm.png
        regression_bmi_Relative_HGB_bar.png

        classification_bmi_bin_Absolute_HGB_beeswarm_class_<label>.png
        classification_bmi_bin_Absolute_HGB_bar.png
        classification_bmi_bin_Relative_HGB_beeswarm_class_<label>.png
        classification_bmi_bin_Relative_HGB_bar.png

        classification_bowel_movement_clean_Absolute_HGB_beeswarm_class_<label>.png
        classification_bowel_movement_clean_Absolute_HGB_bar.png
        classification_bowel_movement_clean_Relative_HGB_beeswarm_class_<label>.png
        classification_bowel_movement_clean_Relative_HGB_bar.png
"""

import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_DATA_DIR = "/ddn_scratch/k5zhao/data/classifier_training"
DEFAULT_BMI_REG_MODEL_DIR = "/home/zhw074/bmi/models"
DEFAULT_BMI_CLS_MODEL_DIR = "/home/zhw074/bmi_bin/models"
DEFAULT_STOOL_MODEL_DIR = "/home/zhw074/stool/models"
DEFAULT_SHAP_DIR = "/home/zhw074/shap_outputs"

PREVALENCE_THRESH = 0.4
MAX_BACKGROUND = 100
MAX_EXPLAIN = 200
RANDOM_STATE = 42

# SHAP for multiclass models is class-specific; we visualize one representative class.
BMI_CLASS_TO_SHOW = "obese"      # clinically meaningful BMI category
STOOL_CLASS_TO_SHOW = "normal"   # baseline healthy stool state

def load_raw_data(data_dir):
    abs_train = pd.read_csv(os.path.join(data_dir, "abs_train.csv"), low_memory=False)
    abs_val   = pd.read_csv(os.path.join(data_dir, "abs_val.csv"), low_memory=False)
    abs_test  = pd.read_csv(os.path.join(data_dir, "abs_test.csv"), low_memory=False)

    rel_train = pd.read_csv(os.path.join(data_dir, "rel_train.csv"), low_memory=False)
    rel_val   = pd.read_csv(os.path.join(data_dir, "rel_val.csv"), low_memory=False)
    rel_test  = pd.read_csv(os.path.join(data_dir, "rel_test.csv"), low_memory=False)

    return {
        "Absolute": {"train": abs_train, "val": abs_val, "test": abs_test},
        "Relative": {"train": rel_train, "val": rel_val, "test": rel_test},
    }


def add_bmi_bin(df):
    df = df.copy()
    df["bmi_bin"] = pd.cut(
        df["bmi"],
        bins=[0, 18.5, 25, 30, 40, 1000],
        right=False,
        labels=["under", "normal", "over", "obese", "severe_obese"]
    )
    return df


def clean_bowel_movement(df):
    df = df.copy()
    raw = df["bowel_movement"].astype(str).str.strip().str.strip('"')

    mapping = {
        "I had normal formed stool, and my stool looks like Type 3 and/or 4": "normal",
        "I had diarrhea (watery stool), and my stool looks like Type 5, 6, and/or 7": "diarrhea",
        "I was constipated (had difficulty passing stool), and my stool looks like Type 1 and/or 2": "constipated",
    }
    df["bowel_movement_clean"] = raw.map(mapping)
    return df


def preprocess_representation(train_df, test_df, prevalence_thresh=PREVALENCE_THRESH):
    feature_cols = [c for c in train_df.columns if c.startswith("G")]

    X_train_raw = train_df[feature_cols]
    X_test_raw  = test_df[feature_cols]

    keep_mask = (X_train_raw > 0).mean(axis=0) >= prevalence_thresh
    kept_features = list(X_train_raw.columns[keep_mask])

    X_test = np.log10(X_test_raw.loc[:, keep_mask] + 1)
    return X_test, kept_features


def sample_rows(X_df, max_n, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    n = min(len(X_df), max_n)
    idx = rng.choice(len(X_df), size=n, replace=False)
    return X_df.iloc[idx].copy()


def get_hgb_model(model_dir, target_col, rep_name):
    model_path = os.path.join(model_dir, f"{target_col}_{rep_name}_HGB.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model: {model_path}")
    return joblib.load(model_path)


def make_explainer(model):
    try:
        return shap.TreeExplainer(model)
    except Exception:
        return shap.Explainer(model.predict, masker=None)


def save_beeswarm(shap_explanation, title, save_path):
    plt.figure()
    shap.plots.beeswarm(shap_explanation, max_display=20, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_bar_from_values(values_2d, feature_names, title, save_path, top_n=20):
    mean_abs = np.abs(values_2d).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_n]

    top_features = [feature_names[i] for i in order][::-1]
    top_vals = mean_abs[order][::-1]

    plt.figure(figsize=(8,6))
    plt.barh(top_features, top_vals)
    plt.xlabel("Mean |SHAP value|")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    
def build_bmi_reg_test_data(raw_data, rep_name):
    train_df = raw_data[rep_name]["train"]
    test_df = raw_data[rep_name]["test"]
    X_test, kept_features = preprocess_representation(train_df, test_df)
    return X_test, kept_features


def build_bmi_cls_test_data(raw_data, rep_name):
    train_df = add_bmi_bin(raw_data[rep_name]["train"])
    test_df  = add_bmi_bin(raw_data[rep_name]["test"])
    X_test, kept_features = preprocess_representation(train_df, test_df)
    return X_test, kept_features


def build_stool_cls_test_data(raw_data, rep_name):
    train_df = clean_bowel_movement(raw_data[rep_name]["train"])
    test_df  = clean_bowel_movement(raw_data[rep_name]["test"])

    train_df = train_df.dropna(subset=["bowel_movement_clean"]).copy()
    test_df  = test_df.dropna(subset=["bowel_movement_clean"]).copy()

    X_test, kept_features = preprocess_representation(train_df, test_df)
    return X_test, kept_features


def run_regression_shap(model, X_test_df, feature_names, rep_name, shap_dir):
    X_bg = sample_rows(X_test_df, MAX_BACKGROUND)
    X_ex = sample_rows(X_test_df, MAX_EXPLAIN)

    explainer = make_explainer(model)
    shap_values = explainer(X_ex)

    beeswarm_path = os.path.join(
        shap_dir, f"regression_bmi_{rep_name}_HGB_beeswarm.png"
    )
    bar_path = os.path.join(
        shap_dir, f"regression_bmi_{rep_name}_HGB_bar.png"
    )

    save_beeswarm(
        shap_values,
        f"BMI Regression | {rep_name} | HGB",
        beeswarm_path
    )

    values_2d = shap_values.values
    save_bar_from_values(
        values_2d,
        feature_names,
        f"BMI Regression | {rep_name} | HGB",
        bar_path
    )


def run_multiclass_shap(model, X_test_df, feature_names, rep_name, shap_dir, prefix, class_to_show):
    X_bg = sample_rows(X_test_df, MAX_BACKGROUND)
    X_ex = sample_rows(X_test_df, MAX_EXPLAIN)

    explainer = make_explainer(model)
    shap_values = explainer(X_ex)

    class_names = list(model.classes_)
    if class_to_show not in class_names:
        class_to_show = class_names[0]

    class_idx = class_names.index(class_to_show)

    # multiclass TreeExplainer often returns (n_samples, n_features, n_classes)
    values = shap_values.values
    if values.ndim != 3:
        raise ValueError(f"Expected multiclass SHAP values with 3 dims, got {values.shape}")

    selected_values = values[:, :, class_idx]

    beeswarm_explanation = shap.Explanation(
        values=selected_values,
        base_values=shap_values.base_values[:, class_idx] if np.ndim(shap_values.base_values) > 1 else shap_values.base_values,
        data=X_ex.values,
        feature_names=feature_names
    )

    beeswarm_path = os.path.join(
        shap_dir, f"{prefix}_{rep_name}_HGB_beeswarm_class_{class_to_show}.png"
    )
    bar_path = os.path.join(
        shap_dir, f"{prefix}_{rep_name}_HGB_bar.png"
    )

    save_beeswarm(
        beeswarm_explanation,
        f"{prefix} | {rep_name} | HGB | class={class_to_show}",
        beeswarm_path
    )

    # aggregate over all classes for overall importance bar plot
    overall_values = np.abs(values).mean(axis=2)
    save_bar_from_values(
        overall_values,
        feature_names,
        f"{prefix} | {rep_name} | HGB",
        bar_path
    )


def main():
    data_dir = os.environ.get("DATA_DIR", DEFAULT_DATA_DIR)
    bmi_reg_model_dir = os.environ.get("BMI_REG_MODEL_DIR", DEFAULT_BMI_REG_MODEL_DIR)
    bmi_cls_model_dir = os.environ.get("BMI_CLS_MODEL_DIR", DEFAULT_BMI_CLS_MODEL_DIR)
    stool_model_dir = os.environ.get("STOOL_MODEL_DIR", DEFAULT_STOOL_MODEL_DIR)
    shap_dir = os.environ.get("SHAP_DIR", DEFAULT_SHAP_DIR)

    os.makedirs(shap_dir, exist_ok=True)

    raw_data = load_raw_data(data_dir)

    for rep_name in ["Absolute", "Relative"]:
        # BMI regression
        reg_model = get_hgb_model(bmi_reg_model_dir, "bmi", rep_name)
        X_reg_test, reg_features = build_bmi_reg_test_data(raw_data, rep_name)
        run_regression_shap(
            reg_model,
            X_reg_test,
            reg_features,
            rep_name,
            shap_dir
        )

        # BMI classification
        bmi_cls_model = get_hgb_model(bmi_cls_model_dir, "bmi_bin", rep_name)
        X_bmi_cls_test, bmi_cls_features = build_bmi_cls_test_data(raw_data, rep_name)
        run_multiclass_shap(
            bmi_cls_model,
            X_bmi_cls_test,
            bmi_cls_features,
            rep_name,
            shap_dir,
            prefix="classification_bmi_bin",
            class_to_show=BMI_CLASS_TO_SHOW
        )

        # Stool classification
        stool_cls_model = get_hgb_model(stool_model_dir, "bowel_movement_clean", rep_name)
        X_stool_test, stool_features = build_stool_cls_test_data(raw_data, rep_name)
        run_multiclass_shap(
            stool_cls_model,
            X_stool_test,
            stool_features,
            rep_name,
            shap_dir,
            prefix="classification_bowel_movement_clean",
            class_to_show=STOOL_CLASS_TO_SHOW
        )

    print(f"\nSaved all SHAP outputs to: {shap_dir}")


if __name__ == "__main__":
    main()