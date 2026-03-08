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
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os

RANDOM_STATE = 42

### LOADING THE DATA ###

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

### EVALUATING MODELS ###

def eval_split_rf(name, Xs, ys):
    pred = rf.predict(Xs)
    r2 = r2_score(ys, pred)
    rmse = root_mean_squared_error(ys, pred)
    mae = mean_absolute_error(ys, pred)
    return r2, rmse, mae
    
def eval_split_hgbr(name, Xs, ys):
    pred = hgbr.predict(Xs)
    r2 = r2_score(ys, pred)
    rmse = root_mean_squared_error(ys, pred)
    mae = mean_absolute_error(ys, pred)
    return r2, rmse, mae
    
def eval_split_lgbm(name, Xs, ys):
    pred = lgbm.predict(Xs)
    r2 = r2_score(ys, pred)
    rmse = root_mean_squared_error(ys, pred)
    mae = mean_absolute_error(ys, pred)
    return r2, rmse, mae

def eval_split_xgboost(name, dmat, y_true):
    pred = bst_best.predict(dmat)
    r2 = r2_score(y_true, pred)
    rmse = root_mean_squared_error(y_true, pred)
    mae = mean_absolute_error(y_true, pred)
    return r2, rmse, mae

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

### FEATURE ENGINEERING ###

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

### TRAINING THE MODEL ###

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

evals_result = {}

bst = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=30000,
    evals=[(dtrain, "train"),(dval, "val")],
    early_stopping_rounds=300,
    evals_result=evals_result,
    verbose_eval=False,
)

xgb_curve_tr = evals_result["train"]["rmse"]
xgb_curve_va = evals_result["val"]["rmse"]

best_ntree = bst.best_iteration + 1
bst_best = bst[:best_ntree]

xgb_train_r2, xgb_train_rmse, xgb_train_mae = eval_split_xgboost("Train", dtrain, y_train_log)
xgb_val_r2, xgb_val_rmse, xgb_val_mae = eval_split_xgboost("Val  ", dval,   y_val_log)
xgb_test_r2, xgb_test_rmse, xgb_test_mae = eval_split_xgboost("Test ", dtest,  y_test_log)

print("Finished Model Training")

### VISUALIZING TRAINING ###

plt.figure(figsize=(6, 4))
plt.plot(range(1, len(xgb_curve_tr) + 1), xgb_curve_tr, label="Train RMSE")
plt.plot(range(1, len(xgb_curve_va) + 1), xgb_curve_va, label="Validation RMSE")
plt.xlabel("Boosting round")
plt.ylabel("RMSE")
plt.title("XGBoost Training and Validation Curves")
plt.legend()
plt.tight_layout()
plt.savefig("/home/k5zhao/model_figs/best_model_training.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved model training figure to: /home/k5zhao/model_figs/best_model_training.png")

### SUMMARY METRICS ###

metrics_df = pd.DataFrame([
    ["XGBoost", "Train", xgb_train_r2, xgb_train_rmse, xgb_train_mae],
    ["XGBoost", "Val",   xgb_val_r2,   xgb_val_rmse,   xgb_val_mae],
    ["XGBoost", "Test",  xgb_test_r2,  xgb_test_rmse,  xgb_test_mae],
], columns=["Model", "Split", "R2", "RMSE", "MAE"])

metrics_df.to_csv("/home/k5zhao/output/best_model_metrics.csv", index=False)

print("Saved metrics to: /home/k5zhao/output/best_model_metrics.csv")