#!/usr/bin/env python3
"""
BMI Classification Pipeline

Purpose
-------
Train classifiers to predict BMI category (bmi_bin) from microbiome features
using two abundance data:
    1. Absolute abundance
    2. Relative abundance

Pipeline Steps
--------------
1. Load train / validation / test datasets.
2. Create `bmi_bin` from continuous BMI using predefined bins.
3. Preprocess microbiome features:
      - keep columns starting with "G"
      - apply prevalence filter on TRAIN only
      - apply log10(x + 1) transform
4. Train models on Absolute and Relative abundance data:
      - RandomForest
      - HistGradientBoosting
      - SVM (RBF kernel)
5. Evaluate models using:
      - Accuracy
      - Macro / Weighted F1
      - R² (computed on encoded labels)
6. Generate plots:
      - Confusion matrices
      - Multiclass ROC curves
7. Perform statistical validation:
      - Bootstrap significance testing comparing
        Absolute vs Relative models
8. Save metrics, plots, and trained models to disk.

Output
------
The pipeline saves results to the directories specified by environment variables
(or their default locations).

Files generated:
    CONF_DIR/
        bmi_bin_Absolute_<model>_confusion.png
        bmi_bin_Relative_<model>_confusion.png
        cls_bmi_results_<model>.csv

    ROC_DIR/
        bmi_bin_Absolute_<model>_roc.png
        bmi_bin_Relative_<model>_roc.png

    MODEL_DIR/
        bmi_bin_Absolute_<model>.joblib
        bmi_bin_Relative_<model>.joblib

Usage
-----
Run all models:
    python bmi_classification.py

Run a single model:
    python bmi_classification.py --model RandomForest
    python bmi_classification.py --model HGB
    python bmi_classification.py --model SVM_RBF

Optional environment variables:
    DATA_DIR    path to input CSV files
    CONF_DIR    folder for confusion matrices
    ROC_DIR     folder for ROC plots
    MODEL_DIR   folder for saved trained models

Example with all variables:
    DATA_DIR="/ddn_scratch/k5zhao/data/classifier_training" \
    CONF_DIR="/home/zhw074/bmi_bin/confusion" \
    ROC_DIR="/home/zhw074/bmi_bin/roc_plots" \
    MODEL_DIR="/home/zhw074/bmi_bin/models" \
    python bmi_classification.py --model HGB
"""

import os
import argparse
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder, label_binarize

DEFAULT_DATA_DIR = "/ddn_scratch/k5zhao/data/classifier_training"
DEFAULT_CONF_DIR = "/home/zhw074/bmi_bin/confusion"
DEFAULT_ROC_DIR  = "/home/zhw074/bmi_bin/roc_plots"
DEFAULT_MODEL_DIR = "/home/zhw074/bmi_bin/models"

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


def add_bmi_bin(df):
    df = df.copy()
    df["bmi_bin"] = pd.cut(
        df["bmi"],
        bins=[0, 18.5, 25, 30, 40, 1000],
        right=False,
        labels=["under", "normal", "over", "obese", "severe_obese"]
    )
    return df


def run_classification_experiments(
    target_col,
    abs_train, abs_val, abs_test,
    rel_train, rel_val, rel_test,
    prevalence_thresh=0.4,
    selected_model="all",
    model_dir=None,
):
    classifiers = {
        "RandomForest": {
            "model": RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                random_state=42,
                n_jobs=-1,
                verbose=1
            ),
            "param_grid": None,
        },
        "HGB": {
            "model": HistGradientBoostingClassifier(
                max_depth=8,
                learning_rate=0.05,
                max_iter=300,
                random_state=42,
                verbose=1
            ),
            "param_grid": {
                "max_depth": [6, 8, 10],
                "learning_rate": [0.05, 0.1],
            },
        },
        "SVM_RBF": {
            "model": SVC(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                probability=True,
                verbose=True
            ),
            "param_grid": {
                "C": [1.0, 10.0, 100.0],
                "gamma": ["scale", "auto"],
            },
        },
    }

    # ---- model switch (all vs single) ----
    if selected_model != "all":
        if selected_model not in classifiers:
            raise ValueError(f"selected_model must be one of {['all'] + list(classifiers.keys())}")
        classifiers = {selected_model: classifiers[selected_model]}

    reps = {
        "Absolute": (abs_train, abs_val, abs_test),
        "Relative": (rel_train, rel_val, rel_test),
    }

    all_results = []
    best_models = {}
    best_params_summary = {}

    if model_dir is not None:
        os.makedirs(model_dir, exist_ok=True)

    for rep_name, (train_df, val_df, test_df) in reps.items():

        print(f"\n=== Representation: {rep_name} ===")

        X_tr, X_va, X_te = preprocess_representation(
            train_df, val_df, test_df,
            prevalence_thresh=prevalence_thresh
        )

        y_tr = train_df[target_col].values
        y_va = val_df[target_col].values
        y_te = test_df[target_col].values

        # Encode labels numerically for R²
        le = LabelEncoder()
        y_tr_enc = le.fit_transform(y_tr)
        y_va_enc = le.transform(y_va)
        y_te_enc = le.transform(y_te)

        for model_name, info in classifiers.items():
            base_model = info["model"]
            param_grid = info["param_grid"]

            print(f">>> Training {model_name} on {rep_name} ...")

            if param_grid is not None:
                gs = GridSearchCV(
                    estimator=clone(base_model),
                    param_grid=param_grid,
                    cv=3,
                    scoring="accuracy",
                    n_jobs=-1,
                    verbose=1
                )
                gs.fit(X_tr, y_tr)
                model = gs.best_estimator_
                print(f"Best parameters: {gs.best_params_}")
                best_params_summary[(rep_name, model_name)] = gs.best_params_
            else:
                model = clone(base_model)
                model.fit(X_tr, y_tr)
                best_params_summary[(rep_name, model_name)] = "default parameters"

            best_models[(rep_name, model_name)] = model
            best_models[(rep_name, model_name, "X_te")] = X_te
            best_models[(rep_name, model_name, "y_te")] = y_te

            if model_dir is not None:
                model_path = os.path.join(model_dir, f"{target_col}_{rep_name}_{model_name}.joblib")
                joblib.dump(model, model_path)

            # Validation predictions
            y_va_pred = model.predict(X_va)
            val_acc = accuracy_score(y_va, y_va_pred)
            val_f1_macro = f1_score(y_va, y_va_pred, average="macro")
            val_f1_weighted = f1_score(y_va, y_va_pred, average="weighted")

            # Encode predictions for R²
            y_va_pred_enc = le.transform(y_va_pred)
            val_r2 = r2_score(y_va_enc, y_va_pred_enc)

            # Test predictions
            y_te_pred = model.predict(X_te)
            test_acc = accuracy_score(y_te, y_te_pred)
            test_f1_macro = f1_score(y_te, y_te_pred, average="macro")
            test_f1_weighted = f1_score(y_te, y_te_pred, average="weighted")

            y_te_pred_enc = le.transform(y_te_pred)
            test_r2 = r2_score(y_te_enc, y_te_pred_enc)

            all_results.append({
                "Task": "Classification",
                "Target": target_col,
                "Representation": rep_name,
                "Model": model_name,
                "Prevalence": prevalence_thresh,
                "Val_Accuracy": val_acc,
                "Test_Accuracy": test_acc,
                "Val_MacroF1": val_f1_macro,
                "Test_MacroF1": test_f1_macro,
                "Val_WeightedF1": val_f1_weighted,
                "Test_WeightedF1": test_f1_weighted,
                "Val_R2": val_r2,
                "Test_R2": test_r2,
            })

    results_df = pd.DataFrame(all_results)
    return results_df, best_models, best_params_summary

# plotting
def plot_cached_classification_results(
    best_models,
    target_col,
    reps=("Absolute", "Relative"),
    models=("RandomForest", "HGB", "SVM_RBF"),
    normalize="true",
    save_dir=None,
):
    """Plot confusion matrices for cached classification models."""
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for rep_name in reps:
        for model_name in models:
            key_model = (rep_name, model_name)
            key_X = (rep_name, model_name, "X_te")
            key_y = (rep_name, model_name, "y_te")

            model = best_models.get(key_model)
            X_te = best_models.get(key_X)
            y_te = best_models.get(key_y)

            if model is None or X_te is None or y_te is None:
                print(f"[skip] No model or test data for {rep_name} - {model_name}")
                continue

            y_pred = model.predict(X_te)

            acc = accuracy_score(y_te, y_pred)
            macro_f1 = f1_score(y_te, y_pred, average="macro")

            fig, ax = plt.subplots(figsize=(4, 4))
            ConfusionMatrixDisplay.from_predictions(
                y_te,
                y_pred,
                normalize=normalize,
                cmap="Blues",
                colorbar=False,
                ax=ax,
            )
            ax.set_title(
                f"{target_col} | {rep_name} - {model_name}\n"
                f"Acc = {acc:.3f}, Macro F1 = {macro_f1:.3f}"
            )
            plt.tight_layout()

            if save_dir is None:
                plt.show()
            else:
                fname = f"{target_col}_{rep_name}_{model_name}_confusion.png"
                plt.savefig(os.path.join(save_dir, fname), dpi=200, bbox_inches="tight")
                plt.close()


def plot_cached_multiclass_roc(
    best_models,
    target_col,
    reps=("Absolute", "Relative"),
    models=("RandomForest", "HGB", "SVM_RBF"),
    save_dir=None,
):
    """Plot one-vs-rest ROC curves and macro AUC for cached classification models."""
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for rep_name in reps:
        for model_name in models:
            key_model = (rep_name, model_name)
            key_X = (rep_name, model_name, "X_te")
            key_y = (rep_name, model_name, "y_te")

            model = best_models.get(key_model)
            X_te = best_models.get(key_X)
            y_te = best_models.get(key_y)

            if model is None or X_te is None or y_te is None:
                print(f"[skip] No model or test data for {rep_name} - {model_name}")
                continue

            # Need probability estimates for ROC
            if not hasattr(model, "predict_proba"):
                print(f"[skip] {rep_name} - {model_name} (no predict_proba)")
                continue

            probs = model.predict_proba(X_te)
            classes = model.classes_
            y_bin = label_binarize(y_te, classes=classes)

            plt.figure(figsize=(4.5, 4))
            aucs = []

            for i, cls in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)
                plt.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.2f})", linewidth=1.2)

            plt.plot([0, 1], [0, 1], "k--", linewidth=1)

            macro_auc = float(np.mean(aucs))
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(
                f"{target_col} ROC | {rep_name} - {model_name}\n"
                f"Macro AUC = {macro_auc:.3f}"
            )
            plt.legend(fontsize=7)
            plt.tight_layout()

            if save_dir is None:
                plt.show()
            else:
                fname = f"{target_col}_{rep_name}_{model_name}_roc.png"
                plt.savefig(os.path.join(save_dir, fname), dpi=200, bbox_inches="tight")
                plt.close()


def bootstrap_classification_test(y_true, pred_abs, pred_rel,
                                  prob_abs=None, prob_rel=None,
                                  n_boot=10000):

    rng = np.random.default_rng(42)
    n = len(y_true)

    acc_diffs = []
    auc_diffs = []

    for _ in range(n_boot):

        idx = rng.choice(n, n, replace=True)

        y_sample = y_true[idx]
        abs_sample = pred_abs[idx]
        rel_sample = pred_rel[idx]

        acc_abs = accuracy_score(y_sample, abs_sample)
        acc_rel = accuracy_score(y_sample, rel_sample)

        acc_diffs.append(acc_rel - acc_abs)

        if prob_abs is not None:
            try:
                auc_abs = roc_auc_score(y_sample, prob_abs[idx], multi_class="ovr")
                auc_rel = roc_auc_score(y_sample, prob_rel[idx], multi_class="ovr")

                auc_diffs.append(auc_rel - auc_abs)

            except ValueError:
                # happens if bootstrap sample missing classes
                continue

    acc_diffs = np.array(acc_diffs)

    print("\nBOOTSTRAP RESULTS")
    print("------------------")
    print("ΔAccuracy:", acc_diffs.mean())
    print("95% CI:", np.percentile(acc_diffs, [2.5,97.5]))
    print("p-value:", 2*min(np.mean(acc_diffs<=0),np.mean(acc_diffs>=0)))

    if len(auc_diffs) > 0:
        auc_diffs = np.array(auc_diffs)
        print("\nΔROC-AUC:", auc_diffs.mean())
        print("95% CI:", np.percentile(auc_diffs,[2.5,97.5]))
        print("p-value:",2*min(np.mean(auc_diffs<=0),np.mean(auc_diffs>=0)))                

        
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
    parser.add_argument(
        "--normalize",
        default="true",
        choices=["true", "pred", "all", None],
        help="Confusion matrix normalization (sklearn normalize=)."
    )
    args = parser.parse_args(args=[])
    
    # Load abs/rel
    abs_train, abs_val, abs_test, rel_train, rel_val, rel_test = load_data()

    # Add bmi_bin exactly like notebook
    abs_train = add_bmi_bin(abs_train)
    abs_val   = add_bmi_bin(abs_val)
    abs_test  = add_bmi_bin(abs_test)

    rel_train = add_bmi_bin(rel_train)
    rel_val   = add_bmi_bin(rel_val)
    rel_test  = add_bmi_bin(rel_test)

    # Print counts like your notebook (helps sanity check on cluster logs)
    print("\n[abs_train bmi_bin counts]")
    print(abs_train["bmi_bin"].value_counts(dropna=False))

    # Output dirs (defaults match your notebook)
    conf_dir = os.environ.get("CONF_DIR", DEFAULT_CONF_DIR)
    roc_dir  = os.environ.get("ROC_DIR", DEFAULT_ROC_DIR)
    model_dir = os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR)

    # Run experiments
    cls_bmi_results, cls_bmi_models, best_params_summary = run_classification_experiments(
        target_col="bmi_bin",
        abs_train=abs_train,
        abs_val=abs_val,
        abs_test=abs_test,
        rel_train=rel_train,
        rel_val=rel_val,
        rel_test=rel_test,
        prevalence_thresh=args.prevalence,
        selected_model=args.model,
        model_dir=model_dir
    )

    # Plot only the requested model(s)
    if args.model == "all":
        plot_models = ("RandomForest", "HGB", "SVM_RBF")
    else:
        plot_models = (args.model,)

    plot_cached_classification_results(
        best_models=cls_bmi_models,
        target_col="bmi_bin",
        models=plot_models,
        normalize=args.normalize,
        save_dir=conf_dir
    )

    plot_cached_multiclass_roc(
        best_models=cls_bmi_models,
        target_col="bmi_bin",
        models=plot_models,
        save_dir=roc_dir
    )

    # Save metrics table (handy for batch runs)
    os.makedirs(conf_dir, exist_ok=True)
    os.makedirs(roc_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    out_csv = os.path.join(conf_dir, f"cls_bmi_results_{args.model}.csv")
    cls_bmi_results.to_csv(out_csv, index=False)

    # Bootstrap significance tests
    print("\nRunning bootstrap significance tests...")

    models = ("RandomForest","HGB","SVM_RBF")

    for m in models:

        try:

            model_abs = cls_bmi_models[("Absolute",m)]
            model_rel = cls_bmi_models[("Relative",m)]

            X_abs = cls_bmi_models[("Absolute",m,"X_te")]
            X_rel = cls_bmi_models[("Relative",m,"X_te")]

            y = cls_bmi_models[("Absolute",m,"y_te")]

            pred_abs = model_abs.predict(X_abs)
            pred_rel = model_rel.predict(X_rel)

            prob_abs = None
            prob_rel = None

            if hasattr(model_abs,"predict_proba"):
                prob_abs = model_abs.predict_proba(X_abs)
                prob_rel = model_rel.predict_proba(X_rel)

            print(f"\n=== Bootstrap test: {m} ===")

            bootstrap_classification_test(
                y_true=np.array(y),
                pred_abs=np.array(pred_abs),
                pred_rel=np.array(pred_rel),
                prob_abs=prob_abs,
                prob_rel=prob_rel
            )

        except KeyError:
            print(f"Skipping {m} (model not run)")
    
    print("\nSaved confusion matrices to:", conf_dir)
    print("Saved ROC plots to:", roc_dir)
    print("Saved models to:", model_dir)
    print("Saved metrics to:", out_csv)

    print("\nBest hyperparameters for each model")
    print("------------------------------------")
    for (rep, model), params in best_params_summary.items():
        print(f"{rep} - {model}: {params}")

    print("\nResults preview:")
    print(cls_bmi_results)


if __name__ == "__main__":
    main()