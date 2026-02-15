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
    r2 = r2_score(ys, pred):.3f
    rmse = root_mean_squared_error(ys, pred):.3f
    mae = mean_absolute_error(ys, pred):.3f
    return r2, rmse, mae

def eval_split_xgboost(name, dmat, y_true):
    pred = bst_best.predict(dmat)
    r2 = r2_score(y_true, pred):.3f
    rmse = root_mean_squared_error(y_true, pred):.3f
    mae = mean_absolute_error(y_true, pred):.3f
    return r2, rmse, mae

### RANDOM FOREST REGRESSION ###

rf = RandomForestRegressor(
    n_estimators=1200,
    random_state=42,
    n_jobs=-1,
    bootstrap=True,
    max_depth=20,
    min_samples_split=20,
    max_features=0.2,
)

rf.fit(X_train_comp, y_train_log)

train_rf_r2, train_rf_rmse, train_rf_mae = eval_split_rf("Train", X_train_comp, y_train_log)
val_rf_r2, val_rf_rmse, val_rf_mae = eval_split_rf("Val  ", X_val_comp, y_val_log)
test_rf_r2, test_rf_rmse, test_rf_mae = eval_split_rf("Test ", X_test_comp, y_test_log)

### GRADIENT BOOSTED REGRESSION ###

best_base = dict(
    random_state=42,
    max_depth=6,
    min_samples_leaf=20,
    learning_rate=0.02,
    l2_regularization=0.0,
    max_iter=4000,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=50,
)

hgbr = HistGradientBoostingRegressor(
    **best_base,
    max_leaf_nodes=31,
    max_bins=255,
)

hgbr.fit(X_train_comp, y_train_log)

train_hgbr_r2, train_hgbr_rmse, train_hgbr_mae = eval_split_hgbr("Train", X_train_comp, y_train_log)
val_hgbr_r2, val_hgbr_rmse, val_hgbr_mae = eval_split_hgbr("Val  ", X_val_comp, y_val_log)
test_hgbr_r2, test_hgbr_rmse, test_hgbr_mae = eval_split_hgbr("Test ", X_test_comp, y_test_log)

### LIGHT GBM ###

base_params = dict(
    objective='regression',
    n_estimators=20000,
    learning_rate=0.03,
    subsample_freq=1,
    num_leaves=31,
    min_child_samples=5,
    reg_lambda=2.0, 
    max_depth=5, 
    min_child_weight=0.01,
    subsample=1.0, 
    colsample_bytree=1.0, 
    min_split_gain=0.0,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)

lgbm = LGBMRegressor(
    **base_params
)

lgbm.fit(
    X_train_comp, y_train_log,
    eval_set=[(X_val_comp, y_val_log)],
    eval_metric="rmse",
    callbacks=[
        early_stopping(stopping_rounds=200, verbose=True)
    ],
)

train_lgbm_r2, train_lgbm_rmse, train_lgbm_mae = eval_split_lgbm("Train", X_train_comp, y_train_log)
val_lgbm_r2, val_lgbm_rmse, val_lgbm_mae = eval_split_lgbm("Val  ", X_val_comp, y_val_log)
test_lgbm_r2, test_lgbm_rmse, test_lgbm_mae = eval_split_lgbm("Test ", X_test_comp, y_test_log)

### XG BOOST ###

params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "seed": RANDOM_STATE,
    "verbosity": 0,

    "max_depth": 5,
    "min_child_weight": 5.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 5.0,

    "eta": 0.02,
    "gamma": 0.001,
    "alpha": 0.1,
}

num_boost_round = 30000
early_stopping_rounds = 300

bst = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtrain, "train"), (dval, "val")],
    early_stopping_rounds=early_stopping_rounds,
)

best_ntree = bst.best_iteration + 1
bst_best = bst[:best_ntree]

train_xgboost_r2, train_xgboost_rmse, train_xgboost_mae = eval_split_xgboost("Train", dtrain, y_train)
val_xgboost_r2, val_xgboost_rmse, val_xgboost_mae = eval_split_xgboost("Val  ", dval,   y_val)
test_xgboost_r2, test_xgboost_rmse, test_xgboost_mae = eval_split_xgboost("Test ", dtest,  y_test)

### NEURAL NETWORK ###

LABEL_COL = "total"   
EPS = 1e-6                 
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(4)   
DEVICE = "cpu"

def clr_transform(rel, eps=1e-6):
    rel = np.asarray(rel, dtype=np.float64)
    rel = np.clip(rel, eps, None)
    logx = np.log(rel)
    return logx - logx.mean(axis=1, keepdims=True)

class Standardizer:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0, keepdims=True)
        self.std_ = X.std(axis=0, keepdims=True)
        self.std_[self.std_ < 1e-12] = 1.0
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    
class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, d_in, hidden=(256, 128, 64), dropout=0.3):
        super().__init__()
        layers = []
        prev = d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp(
    X_train, y_train,
    X_val, y_val,
    lr=1e-3, weight_decay=1e-3,
    batch_size=128, max_epochs=20000, patience=100,
    device="cpu"
):
    model = MLP(d_in=X_train.shape[1]).to(device)
    train_loader = DataLoader(RegressionDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(RegressionDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        tr_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        va_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                va_losses.append(loss_fn(pred, yb).item())

        tr = float(np.mean(tr_losses))
        va = float(np.mean(va_losses))

        if va < best_val - 1e-5:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)

def spearman_corr(y_true, y_pred):
    a = pd.Series(np.asarray(y_true).ravel()).rank().to_numpy()
    b = pd.Series(np.asarray(y_pred).ravel()).rank().to_numpy()
    return float(np.corrcoef(a, b)[0, 1])

X_train_clr = clr_transform(X_train_comp, eps=EPS)
X_val_clr   = clr_transform(X_val_comp,   eps=EPS)
X_test_clr  = clr_transform(X_test_comp,  eps=EPS)

x_scaler = Standardizer()
X_train_s = x_scaler.fit_transform(X_train_clr)
X_val_s   = x_scaler.transform(X_val_clr)
X_test_s  = x_scaler.transform(X_test_clr)


y_mean, y_std = y_train_log.mean(), y_train_log.std()
if y_std < 1e-12:
    y_std = 1.0

y_train_s = (y_train_log - y_mean) / y_std
y_val_s   = (y_val_log   - y_mean) / y_std

model = train_mlp(X_train_s, y_train_s, X_val_s, y_val_s, device=DEVICE)

model.eval()
with torch.no_grad():
    pred_test_s = model(torch.tensor(X_test_s, dtype=torch.float32).to(DEVICE)).cpu().numpy().reshape(-1)

pred_test_log  = pred_test_s * y_std + y_mean

def compute_metrics(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }

def predict_nn(model, X, kind, device=None):
    if device is None:
        raise ValueError("device must be provided for kind='torch'")
    model.eval()
    with torch.no_grad():
        pred_test_s = model(torch.tensor(X, dtype=torch.float32).to(DEVICE)).cpu().numpy().reshape(-1)
    pred_test_log  = pred_test_s * y_std + y_mean
    return pred_test_log


def eval_model_splits(model_name, model, kind,
                      X_train, y_train, X_val, y_val, X_test, y_test):
    pred_tr = predict_any(model, X_train, kind, device=DEVICE)
    pred_va = predict_any(model, X_val,   kind, device=DEVICE)
    pred_te = predict_any(model, X_test,  kind, device=DEVICE)

    tr = compute_metrics(y_train, pred_tr)
    va = compute_metrics(y_val,   pred_va)
    te = compute_metrics(y_test,  pred_te)

    return {
        "model": model_name,
        "train_r2": tr["r2"], "train_rmse": tr["rmse"], "train_mae": tr["mae"],
        "val_r2":   va["r2"], "val_rmse":   va["rmse"], "val_mae":   va["mae"],
        "test_r2":  te["r2"], "test_rmse":  te["rmse"], "test_mae":  te["mae"],
    }

rows = []

rows.append({"model": "rf",
            "train_r2": train_rf_r2, "train_rmse": train_rf_rmse, "train_mae": train_rf_mae,
            "val_r2": val_rf_r2, "val_rmse": val_rf_rmse, "val_mae": val_rf_mae,
            "test_r2": test_rf_r2, "test_rmse": test_rf_rmse, "test_mae": test_rf_mae})

rows.append({"model": "hgbr",
            "train_r2": train_hgbr_r2, "train_rmse": train_hgbr_rmse, "train_mae": train_hgbr_mae,
            "val_r2": val_hgbr_r2, "val_rmse": val_hgbr_rmse, "val_mae": val_hgbr_mae,
            "test_r2": test_hgbr_r2, "test_rmse": test_hgbr_rmse, "test_mae": test_hgbr_mae})

rows.append({"model": "lgbm",
            "train_r2": train_lgbm_r2, "train_rmse": train_lgbm_rmse, "train_mae": train_lgbm_mae,
            "val_r2": val_lgbm_r2, "val_rmse": val_lgbm_rmse, "val_mae": val_lgbm_mae,
            "test_r2": test_lgbm_r2, "test_rmse": test_lgbm_rmse, "test_mae": test_lgbm_mae})

rows.append({"model": "bst",
            "train_r2": train_xgboost_r2, "train_rmse": train_xgboost_rmse, "train_mae": train_xgboost_mae,
            "val_r2": val_xgboost_r2, "val_rmse": val_xgboost_rmse, "val_mae": val_xgboost_mae,
            "test_r2": test_xgboost_r2, "test_rmse": test_xgboost_rmse, "test_mae": test_xgboost_mae})

rows.append(eval_model_splits(
    "nn", model, "torch",
    X_train_s, y_train_log, X_val_s, y_val_log, X_test_s, y_test_log
))

summary_df = pd.DataFrame(rows)

cols = ["model",
        "train_r2","train_rmse","train_mae",
        "val_r2","val_rmse","val_mae",
        "test_r2","test_rmse","test_mae"]
summary_df = summary_df[cols].sort_values("val_rmse").reset_index(drop=True)
summary_df.round(4)
summary_df.to_excel("/output/model_summary.xlsx", index=False)