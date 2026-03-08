import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os

RANDOM_STATE = 42
LABEL_COL = "total"
EPS = 1e-6
RANDOM_STATE = 42

### LOADING THE DATA ###

rel_ft = pd.read_csv('/ddn_scratch/miter/pacbio-tables/pacbio-pergenome-feature-table.tsv', sep='\t', index_col=0)
abs_ft = pd.read_csv('/ddn_scratch/miter/pacbio-tables/pacbio-absquant-feature-table.tsv', sep='\t', index_col=0)

abs_ft['total'] = abs_ft.sum(axis=1)

count_df = rel_ft.merge(abs_ft[['total']], left_index=True, right_index=True, how='inner')

row_sums = rel_ft.sum(axis=1)
rel_ft = rel_ft.div(row_sums, axis=0)

comp_df = rel_ft.merge(abs_ft[['total']], left_index=True, right_index=True, how='inner')

TEST_SIZE = 0.20         
VAL_SIZE = 0.20
val_relative = VAL_SIZE / (1 - TEST_SIZE)

count_train_val_df, count_test_df = train_test_split(
    count_df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True
)

count_train_df, count_val_df = train_test_split(
    count_train_val_df,
    test_size=val_relative,
    random_state=RANDOM_STATE,
    shuffle=True
)

count_train_df.to_csv("/ddn_scratch/k5zhao/data/pacbio/train.csv", index=True)
count_val_df.to_csv("/ddn_scratch/k5zhao/data/pacbio/val.csv", index=True)
count_test_df.to_csv("/ddn_scratch/k5zhao/data/pacbio/test.csv", index=True)

comp_train_val_df, comp_test_df = train_test_split(
    comp_df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True
)

comp_train_df, comp_val_df = train_test_split(
    comp_train_val_df,
    test_size=val_relative,
    random_state=RANDOM_STATE,
    shuffle=True
)

comp_train_df.to_csv("/ddn_scratch/k5zhao/data/pacbio/compositional/train.csv", index=True)
comp_val_df.to_csv("/ddn_scratch/k5zhao/data/pacbio/compositional/val.csv", index=True)
comp_test_df.to_csv("/ddn_scratch/k5zhao/data/pacbio/compositional/test.csv", index=True)

### MODEL TRAINING ###

X_train_count = pd.read_csv('/ddn_scratch/k5zhao/data/pacbio/train.csv', index_col=0)
X_val_count = pd.read_csv('/ddn_scratch/k5zhao/data/pacbio/val.csv', index_col=0)
X_test_count = pd.read_csv('/ddn_scratch/k5zhao/data/pacbio/test.csv', index_col=0)

y_train = X_train_count['total'].values
y_val = X_val_count['total'].values
y_test = X_test_count['total'].values

X_train_count = X_train_count.drop(columns=['total'])
X_val_count = X_val_count.drop(columns=['total'])
X_test_count = X_test_count.drop(columns=['total'])

X_train_comp = pd.read_csv('/ddn_scratch/k5zhao/data/pacbio/compositional/train.csv', index_col=0)
X_val_comp = pd.read_csv('/ddn_scratch/k5zhao/data/pacbio/compositional/val.csv', index_col=0)
X_test_comp = pd.read_csv('/ddn_scratch/k5zhao/data/pacbio/compositional/test.csv', index_col=0)

X_train_comp = X_train_comp.drop(columns=['total'])
X_val_comp = X_val_comp.drop(columns=['total'])
X_test_comp = X_test_comp.drop(columns=['total'])

y_train_log = np.log1p(y_train)
y_val_log   = np.log1p(y_val)
y_test_log  = np.log1p(y_test)

def eval_split_xgboost(name, dmat, y_true):
    pred = bst_best.predict(dmat)
    r2 = r2_score(y_true, pred)
    rmse = root_mean_squared_error(y_true, pred)
    mae = mean_absolute_error(y_true, pred)
    return r2, rmse, mae
    
### BEST MODEL ###

LABEL_COL = "total"
EPS = 1e-6
RANDOM_STATE = 42

def clr_transform(rel, eps=1e-6):
    X = np.asarray(rel, dtype=np.float64)
    X = np.clip(X, eps, None)
    logX = np.log(X)
    return logX - logX.mean(axis=1, keepdims=True)

def build_X(comp_df, count_df, feature_cols, eps=1e-6):
    rel = comp_df.reindex(columns=feature_cols, fill_value=0.0).to_numpy(np.float64)
    counts = count_df.reindex(columns=feature_cols, fill_value=0.0).to_numpy(np.float64)

    lib = counts.sum(axis=1)
    X1 = clr_transform(rel, eps=eps)
    X2 = np.log1p(counts) - np.log1p(lib).reshape(-1, 1)
    X3 = np.log1p(lib).reshape(-1, 1)

    X4 = (counts > 0).astype(np.float32)
    X5 = np.log1p(counts) 

    return np.hstack([X1, X2, X3, X4, X5])

feature_cols = X_train_comp.columns

X_train = build_X(X_train_comp, X_train_count, feature_cols, eps=EPS)
X_val   = build_X(X_val_comp, X_val_count,   feature_cols, eps=EPS)
X_test  = build_X(X_test_comp, X_test_count,  feature_cols, eps=EPS)

dtrain = xgb.DMatrix(X_train, label=y_train_log)
dval   = xgb.DMatrix(X_val,   label=y_val_log)
dtest  = xgb.DMatrix(X_test,  label=y_test_log)

params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "seed": RANDOM_STATE,
    "verbosity": 1,

    "max_depth": 5,
    "min_child_weight": 10.0,
    "subsample": 0.8,
    "colsample_bytree": 0.5,
    "lambda": 10.0,
    "subsample": 0.7,

    "eta": 0.02,
    "gamma": 0.01,
    "alpha": 0.1,
}

bst = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=30000,
    evals=[(dval, "val")],
    early_stopping_rounds=300,
    verbose_eval=False,
)

best_ntree = bst.best_iteration + 1
bst_best = bst[:best_ntree]

pacbio_train_r2, pacbio_train_rmse, pacbio_train_mae = eval_split_xgboost("Train", dtrain, y_train_log)
pacbio_val_r2, pacbio_val_rmse, pacbio_val_mae = eval_split_xgboost("Val  ", dval,   y_val_log)
pacbio_test_r2, pacbio_test_rmse, pacbio_test_mae = eval_split_xgboost("Test ", dtest,  y_test_log)

print("Model Training on Pacbio Data Finished")

### VISUALIZING THE RESULTS ###

def plot_residuals_3way(model,
                        X_train, y_train,
                        X_val, y_val,
                        X_test, y_test,
                        title="Residuals vs Predicted (log1p space)",
                        savepath="residuals_3way.png",
                        dpi=300,
                        show=False):
    splits = [
        ("Train", X_train, y_train),
        ("Validation", X_val, y_val),
        ("Test", X_test, y_test),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    for ax, (split_name, X, y) in zip(axes, splits):
        y_pred = model.predict(X, iteration_range=(0, model.best_iteration + 1))
        resid = y - y_pred

        ax.scatter(y_pred, resid, alpha=0.4)
        ax.axhline(0, linewidth=1)
        ax.set_xlabel("Predicted log1p(total)")
        ax.set_ylabel("Residual")
        ax.set_title(split_name)

    fig.suptitle(title)
    fig.tight_layout()

    os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
    plt.savefig(savepath, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def scatter_true_vs_pred_3way(model,
                              X_train, y_train,
                              X_val, y_val,
                              X_test, y_test,
                              title="True vs Predicted (log1p space)",
                              savepath="true_vs_pred_3way.png",
                              dpi=300,
                              show=False):
    splits = [
        ("Train", X_train, y_train),
        ("Validation", X_val, y_val),
        ("Test", X_test, y_test),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    for ax, (split_name, X, y) in zip(axes, splits):
        y_pred = model.predict(X, iteration_range=(0, model.best_iteration + 1))

        ax.scatter(y, y_pred, alpha=0.4)
        lo = min(y.min(), y_pred.min())
        hi = max(y.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi])
        ax.set_xlabel("True log1p(total)")
        ax.set_ylabel("Predicted log1p(total)")
        ax.set_title(split_name)

    fig.suptitle(title)
    fig.tight_layout()

    os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
    plt.savefig(savepath, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    
plot_residuals_3way(
    bst,
    dtrain, y_train_log,
    dval, y_val_log,
    dtest, y_test_log,
    savepath="/home/k5zhao/model_figs/pacbio_residuals.png"
)

scatter_true_vs_pred_3way(
    bst,
    dtrain, y_train_log,
    dval, y_val_log,
    dtest, y_test_log,
    savepath="/home/k5zhao/model_figs/pacbio_true_vs_pred.png"
)

print("Pacbio Model figures saved at: /home/k5zhao/model_figs/pacbio_residuals.png and /home/k5zhao/model_figs/pacbio_true_vs_pred.png")

### TRAINING THE MODEL ON SUBSAMPLED DATA ###

X_train_count = pd.read_csv('/ddn_scratch/k5zhao/data/model_training/train.csv', index_col='original_SampleID')
X_val_count = pd.read_csv('/ddn_scratch/k5zhao/data/model_training/val.csv', index_col='original_SampleID')
X_test_count = pd.read_csv('/ddn_scratch/k5zhao/data/model_training/test.csv', index_col='original_SampleID')

y_train = X_train_count['total'].values
y_val = X_val_count['total'].values
y_test = X_test_count['total'].values

X_train_count = X_train_count.drop(columns=['total'])
X_val_count = X_val_count.drop(columns=['total'])
X_test_count = X_test_count.drop(columns=['total'])

X_train_comp = pd.read_csv('/ddn_scratch/k5zhao/data/model_training/compositional/train.csv', index_col='original_SampleID')
X_val_comp = pd.read_csv('/ddn_scratch/k5zhao/data/model_training/compositional/val.csv', index_col='original_SampleID')
X_test_comp = pd.read_csv('/ddn_scratch/k5zhao/data/model_training/compositional/test.csv', index_col='original_SampleID')

X_train_comp = X_train_comp.drop(columns=['total'])
X_val_comp = X_val_comp.drop(columns=['total'])
X_test_comp = X_test_comp.drop(columns=['total'])

y_train_log = np.log1p(y_train)
y_val_log   = np.log1p(y_val)
y_test_log  = np.log1p(y_test)

### HELPER FUNCTIONS ###

def downsample_df_and_y(X_comp_split, X_count_split, y_log_split, n, seed=42):
    common_idx = X_comp_split.index.intersection(X_count_split.index)
    X_comp_split = X_comp_split.loc[common_idx]
    X_count_split = X_count_split.loc[common_idx]

    if isinstance(y_log_split, (pd.Series, pd.DataFrame)):
        y_log_split = y_log_split.loc[common_idx]
    else:
        y_log_split = pd.Series(y_log_split, index=common_idx)

    if len(common_idx) < n:
        raise ValueError(f"Split has {len(common_idx)} samples, need {n}.")

    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.array(common_idx), size=n, replace=False)

    return (X_comp_split.loc[chosen],
            X_count_split.loc[chosen],
            y_log_split.loc[chosen].to_numpy())

### BEST MODEL ###

X_train_comp, X_train_count, y_train_log = downsample_df_and_y(X_train_comp, X_train_count, y_train_log, 87, seed=RANDOM_STATE)
X_val_comp,   X_val_count,   y_val_log   = downsample_df_and_y(X_val_comp,   X_val_count,   y_val_log,   30, seed=RANDOM_STATE+1)
X_test_comp,  X_test_count,  y_test_log  = downsample_df_and_y(X_test_comp,  X_test_count,  y_test_log,  30, seed=RANDOM_STATE+2)

feature_cols = X_train_comp.columns

X_train = build_X(X_train_comp, X_train_count, feature_cols, eps=EPS)
X_val   = build_X(X_val_comp, X_val_count,   feature_cols, eps=EPS)
X_test  = build_X(X_test_comp, X_test_count,  feature_cols, eps=EPS)

dtrain = xgb.DMatrix(X_train, label=y_train_log)
dval   = xgb.DMatrix(X_val,   label=y_val_log)
dtest  = xgb.DMatrix(X_test,  label=y_test_log)

params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "seed": RANDOM_STATE,
    "verbosity": 1,

    "max_depth": 5,
    "min_child_weight": 10.0,
    "subsample": 0.8,
    "colsample_bytree": 0.5,
    "lambda": 10.0,
    "subsample": 0.7,

    "eta": 0.02,
    "gamma": 0.01,
    "alpha": 0.1,
}

bst = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=30000,
    evals=[(dval, "val")],
    early_stopping_rounds=300,
    verbose_eval=False,
)

best_ntree = bst.best_iteration + 1
bst_best = bst[:best_ntree]

nph_train_r2, nph_train_rmse, nph_train_mae = eval_split_xgboost("Train", dtrain, y_train_log)
nph_val_r2, nph_val_rmse, nph_val_mae = eval_split_xgboost("Val  ", dval,   y_val_log)
nph_test_r2, nph_test_rmse, nph_test_mae = eval_split_xgboost("Test ", dtest,  y_test_log)

print("Model Training on Subsampled NPH Finished")

### VISUALIZING THE RESULTS ###

plot_residuals_3way(
    bst,
    dtrain, y_train_log,
    dval, y_val_log,
    dtest, y_test_log,
    savepath="/home/k5zhao/model_figs/nph_subsampled_residuals.png"
)

scatter_true_vs_pred_3way(
    bst,
    dtrain, y_train_log,
    dval, y_val_log,
    dtest, y_test_log,
    savepath="/home/k5zhao/model_figs/nph_subsampled_true_vs_pred.png"
)

print("Subsampled NPH Model figures saved at: /home/k5zhao/model_figs/nph_subsampled_residuals.png and /home/k5zhao/model_figs/nph_subsampled_true_vs_pred.png")
### SUMMARY STATS ###

summary_df = pd.DataFrame([
    ["NPH",    "Train", nph_train_r2,    nph_train_rmse,    nph_train_mae],
    ["NPH",    "Val",   nph_val_r2,      nph_val_rmse,      nph_val_mae],
    ["NPH",    "Test",  nph_test_r2,     nph_test_rmse,     nph_test_mae],
    ["PacBio", "Train", pacbio_train_r2, pacbio_train_rmse, pacbio_train_mae],
    ["PacBio", "Val",   pacbio_val_r2,   pacbio_val_rmse,   pacbio_val_mae],
    ["PacBio", "Test",  pacbio_test_r2,  pacbio_test_rmse,  pacbio_test_mae],
], columns=["Dataset", "Split", "R2", "RMSE", "MAE"])

summary_df.round(4).to_csv("/home/k5zhao/output/pacbio_metrics_summary.csv", index=False)

print("Summary metrics saved to: /home/k5zhao/output/pacbio_metrics_summary.csv")