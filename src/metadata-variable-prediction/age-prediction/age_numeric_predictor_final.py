#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-generated from: age_numeric_predictor_final.ipynb
Generated on: 2026-03-01T23:09:36

This script was created by extracting code cells from the notebook.
Notebook magics/shell commands (%, %%, !) were commented out.
"""

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

# On log data
# Preprocessing train/test/val datasets
abs_train = pd.read_csv('/ddn_scratch/k5zhao/data/classifier_training/abs_train.csv', low_memory=False)
abs_test = pd.read_csv('/ddn_scratch/k5zhao/data/classifier_training/abs_test.csv', low_memory=False)
abs_val = pd.read_csv('/ddn_scratch/k5zhao/data/classifier_training/abs_val.csv', low_memory=False)

rel_train = pd.read_csv('/ddn_scratch/k5zhao/data/classifier_training/rel_train.csv', low_memory=False)
rel_test = pd.read_csv('/ddn_scratch/k5zhao/data/classifier_training/rel_test.csv', low_memory=False)
rel_val = pd.read_csv('/ddn_scratch/k5zhao/data/classifier_training/rel_val.csv', low_memory=False)

# Stratifying by gender
abs_train = abs_train[abs_train["sex"].isin(['male', 'female'])].copy()
abs_val = abs_val[abs_val["sex"].isin(['male', 'female'])].copy()
abs_test = abs_test[abs_test["sex"].isin(['male', 'female'])].copy()

rel_train = rel_train[rel_train["sex"].isin(['male', 'female'])].copy()
rel_val = rel_val[rel_val["sex"].isin(['male', 'female'])].copy()
rel_test = rel_test[rel_test["sex"].isin(['male', 'female'])].copy()

# Creating the age bin
bins = [18,30,40,50,60,70,100]
abs_train['age_bin'] = pd.cut(
    abs_train['age'],
    bins=bins,
    right=False,
    include_lowest=True
).astype(str)

abs_test['age_bin']=pd.cut(
    abs_test['age'],
    bins=bins,  
    right=False,
    include_lowest=True
).astype(str)

abs_val['age_bin']=pd.cut(
    abs_val['age'],
    bins=bins,  
    right=False,
    include_lowest=True
).astype(str)

rel_train['age_bin'] = pd.cut(
    rel_train['age'],
    bins=bins,
    right=False,
    include_lowest=True
).astype(str)

rel_test['age_bin']=pd.cut(
    rel_test['age'],
    bins=bins,  
    right=False,
    include_lowest=True
).astype(str)

rel_val['age_bin']=pd.cut(
    rel_val['age'],
    bins=bins,  
    right=False,
    include_lowest=True
).astype(str)

# Balancing the training set
# Absolute quant
target_n = 329

abs_balanced_train = (
    abs_train
    .groupby("age_bin", group_keys=False)
    .apply(lambda x: x.sample(
        n=min(len(x), target_n),
        random_state=42
    ))
)

# Relative quant
rel_balanced_train = (
    rel_train
    .groupby("age_bin", group_keys=False)
    .apply(lambda x: x.sample(
        n=min(len(x), target_n),
        random_state=42
    ))
)

# Splitting into features and targets
# Absolute quant data
abs_X_train = abs_balanced_train[abs_train.columns[:1148]].drop(columns=['original_SampleID'])
abs_Y_train = abs_balanced_train['age']

abs_X_test = abs_test[abs_test.columns[:1148]].drop(columns=['original_SampleID'])
abs_Y_test  = abs_test['age']

abs_X_val = abs_val[abs_val.columns[:1148]].drop(columns=['original_SampleID'])
abs_Y_val  = abs_val['age']

# Relative abundance data
rel_X_train = rel_balanced_train[rel_balanced_train.columns[:1148]].drop(columns=['original_SampleID'])
rel_Y_train = rel_balanced_train['age']

rel_X_test = rel_test[rel_test.columns[:1148]].drop(columns=['original_SampleID'])
rel_Y_test  = rel_test['age']

rel_X_val = rel_val[rel_val.columns[:1148]].drop(columns=['original_SampleID'])
rel_Y_val  = rel_val['age']


# Preprocessing
# Using log transform to standardize the data
# Absolute quant data
abs_X_train_log = abs_X_train.copy()
abs_X_train_log = np.log1p(abs_X_train_log)

abs_X_test_log = abs_X_test.copy()
abs_X_test_log = np.log1p(abs_X_test_log)

abs_X_val_log = abs_X_val.copy()
abs_X_val_log = np.log1p(abs_X_val_log)

# Relative abundance data
rel_X_train_log = rel_X_train.copy()
rel_X_train_log = np.log1p(rel_X_train_log)

rel_X_test_log = rel_X_test.copy()
rel_X_test_log = np.log1p(rel_X_test_log)

rel_X_val_log = rel_X_val.copy()
rel_X_val_log = np.log1p(rel_X_val_log)

# Encoding sex
sex_encoder = OneHotEncoder(
    drop="first",           
    handle_unknown="ignore",
    sparse_output=False
)

sex_encoder.fit(abs_train[["sex"]])

# Absolute
abs_train_sex = sex_encoder.transform(abs_balanced_train[["sex"]])
abs_test_sex  = sex_encoder.transform(abs_test[["sex"]])
abs_val_sex   = sex_encoder.transform(abs_val[["sex"]])

# Relative
rel_train_sex = sex_encoder.transform(rel_balanced_train[["sex"]])
rel_test_sex  = sex_encoder.transform(rel_test[["sex"]])
rel_val_sex   = sex_encoder.transform(rel_val[["sex"]])

# Appending to training data
abs_X_train_aug = np.hstack([abs_X_train_log.values, abs_train_sex])
abs_X_test_aug  = np.hstack([abs_X_test_log.values,  abs_test_sex])
abs_X_val_aug   = np.hstack([abs_X_val_log.values,   abs_val_sex])

rel_X_train_aug = np.hstack([rel_X_train_log.values, rel_train_sex])
rel_X_test_aug  = np.hstack([rel_X_test_log.values,  rel_test_sex])
rel_X_val_aug   = np.hstack([rel_X_val_log.values,   rel_val_sex])

abs_rf_reg = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
abs_rf_reg.fit(abs_X_train_aug, abs_Y_train)

# Validation Set
abs_rf_val_preds = abs_rf_reg.predict(abs_X_val_aug)
abs_rf_reg_val_r2 = r2_score(abs_Y_val, abs_rf_val_preds)
abs_rf_reg_val_rmse = np.sqrt(mean_squared_error(abs_Y_val, abs_rf_val_preds))
abs_rf_reg_val_mae = mean_absolute_error(abs_Y_val, abs_rf_val_preds)
print("Validation R²:", abs_rf_reg_val_r2)
print("Validation RMSE:", abs_rf_reg_val_rmse)
print("Validation MAE:", abs_rf_reg_val_mae)

# Test Set
abs_rf_test_preds = abs_rf_reg.predict(abs_X_test_aug)
abs_rf_reg_test_r2 = r2_score(abs_Y_test, abs_rf_test_preds)
abs_rf_reg_test_rmse = np.sqrt(mean_squared_error(abs_Y_test, abs_rf_test_preds))
abs_rf_reg_test_mae = mean_absolute_error(abs_Y_test, abs_rf_test_preds)
print("Test R²:", abs_rf_reg_test_r2)
print("Test RMSE:", abs_rf_reg_test_rmse)
print("Test MAE:", abs_rf_reg_val_mae)

# Relative
rel_rf_reg = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
rel_rf_reg.fit(rel_X_train_aug, rel_Y_train)

# Validation Set
rel_rf_val_preds = rel_rf_reg.predict(rel_X_val_aug)
rel_rf_reg_val_r2 = r2_score(rel_Y_val, rel_rf_val_preds)
rel_rf_reg_val_rmse = np.sqrt(mean_squared_error(rel_Y_val, rel_rf_val_preds))
rel_rf_reg_val_mae = mean_absolute_error(rel_Y_val, rel_rf_val_preds)
print("Validation R²:", rel_rf_reg_val_r2)
print("Validation RMSE:", rel_rf_reg_val_rmse)
print("Validation MAE:", rel_rf_reg_val_mae)

# Test Set
rel_rf_test_preds = rel_rf_reg.predict(rel_X_test_aug)
rel_rf_reg_test_r2 = r2_score(rel_Y_test, rel_rf_test_preds)
rel_rf_reg_test_rmse = np.sqrt(mean_squared_error(rel_Y_test, rel_rf_test_preds))
rel_rf_reg_test_mae = mean_absolute_error(rel_Y_test, rel_rf_test_preds)
print("Test R²:", rel_rf_reg_test_r2)
print("Test RMSE:", rel_rf_reg_test_rmse)
print("Test MAE:", rel_rf_reg_val_mae)

# Plotting the results
r2_abs = r2_score(abs_Y_test, abs_rf_test_preds)
r2_rel = r2_score(rel_Y_test, rel_rf_test_preds)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

min_val = min(abs_Y_test.min(), rel_Y_test.min())
max_val = max(abs_Y_test.max(), rel_Y_test.max())
x_line = np.linspace(min_val, max_val, 100)

# Absolute
axes[0].scatter(abs_Y_test, abs_rf_test_preds, alpha=0.5)
axes[0].plot(x_line, x_line)  # 1:1 perfect prediction line
axes[0].set_title(f"Absolute Abundance RF\nR² = {r2_abs:.3f}")
axes[0].set_xlabel("True Age")
axes[0].set_ylabel("Predicted Age")

# Relative
axes[1].scatter(rel_Y_test, rel_rf_test_preds, alpha=0.5)
axes[1].plot(x_line, x_line)  # 1:1 perfect prediction line
axes[1].set_title(f"Relative Abundance RF\nR² = {r2_rel:.3f}")
axes[1].set_xlabel("True Age")
axes[1].set_ylabel("Predicted Age")

plt.tight_layout()
fig.suptitle("Absolute vs. Relative Abundance Random Forest Regressor Results", y=1.03)
plt.savefig("rf_reg_comparison.png", format="png")
plt.show()

# Bootstrapping to see if differences are due to chance
n_boot = 5000
rng = np.random.default_rng(42)

diffs = []

n = len(abs_Y_test)

for _ in range(n_boot):
    idx = rng.choice(n, n, replace=True)
    
    y_sample = abs_Y_test.iloc[idx]
    abs_sample = abs_rf_test_preds[idx]
    rel_sample = rel_rf_test_preds[idx]
    
    r2_abs = r2_score(y_sample, abs_sample)
    r2_rel = r2_score(y_sample, rel_sample)
    
    diffs.append(r2_rel - r2_abs)

diffs = np.array(diffs)

ci_lower = np.percentile(diffs, 2.5)
ci_upper = np.percentile(diffs, 97.5)
mean_diff = diffs.mean()

print("Mean ΔR²:", mean_diff)
print("95% CI:", ci_lower, "to", ci_upper)

p_value = np.mean(diffs <= 0)
print("P-value:", p_value)

# Initialize model
gbr = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

# Fit
abs_gbr = gbr.fit(abs_X_train_aug, abs_Y_train)

# Validation Set
y_pred = abs_gbr.predict(abs_X_val_aug)
abs_gbr_val_r2 = r2_score(abs_Y_val, y_pred)
abs_gbr_val_rmse = np.sqrt(mean_squared_error(abs_Y_val, y_pred))
abs_gbr_val_mae = mean_absolute_error(abs_Y_val, y_pred)
print("Validation R²:", abs_gbr_val_r2)
print("Validation RMSE:", abs_gbr_val_rmse)
print("Validation MAE:", abs_gbr_val_mae)

# Test Set
y_pred = abs_gbr.predict(abs_X_test_aug)
abs_gbr_test_r2 = r2_score(abs_Y_test, y_pred)
abs_gbr_test_rmse = np.sqrt(mean_squared_error(abs_Y_test, y_pred))
abs_gbr_test_mae = mean_absolute_error(abs_Y_test, y_pred)
print("Test R²:", abs_gbr_test_r2)
print("Test RMSE:", abs_gbr_test_rmse)
print("Test MAE:", abs_gbr_val_mae)

# Fit
# Relative
rel_gbr = gbr.fit(rel_X_train_aug, rel_Y_train)

# Validation Set
y_pred = rel_gbr.predict(rel_X_val_aug)
rel_gbr_val_r2 = r2_score(rel_Y_val, y_pred)
rel_gbr_val_rmse = np.sqrt(mean_squared_error(rel_Y_val, y_pred))
rel_gbr_val_mae = mean_absolute_error(rel_Y_val, y_pred)
print("Validation R²:", rel_gbr_val_r2)
print("Validation RMSE:", rel_gbr_val_rmse)
print("Validation MAE:", rel_gbr_val_mae)

# Test Set
y_pred = rel_gbr.predict(rel_X_test_aug)
rel_gbr_test_r2 = r2_score(rel_Y_test, y_pred)
rel_gbr_test_mae = mean_absolute_error(rel_Y_test, y_pred)
rel_gbr_test_rmse = np.sqrt(mean_squared_error(rel_Y_test, y_pred))
print("Test R²:", rel_gbr_test_r2)
print("Test RMSE:", rel_gbr_test_rmse)
print("Test MAE:", rel_gbr_test_mae)

# Plotting the results
abs_pred = abs_gbr.predict(abs_X_test_aug)
rel_pred = rel_gbr.predict(rel_X_test_aug)

r2_abs = r2_score(abs_Y_test, abs_pred)
r2_rel = r2_score(rel_Y_test, rel_pred)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

min_val = min(abs_Y_test.min(), rel_Y_test.min())
max_val = max(abs_Y_test.max(), rel_Y_test.max())
x_line = np.linspace(min_val, max_val, 100)

fig.suptitle("Absolute vs. Relative Abundance Gradient Boosted Regressor Results")

# Absolute
axes[0].scatter(abs_Y_test, abs_pred, alpha=0.5)
axes[0].plot(x_line, x_line)  # 1:1 perfect prediction line
axes[0].set_title(f"Absolute Abundance\nR² = {r2_abs:.3f}")
axes[0].set_xlabel("True Age")
axes[0].set_ylabel("Predicted Age")

# Relative
axes[1].scatter(rel_Y_test, rel_pred, alpha=0.5)
axes[1].plot(x_line, x_line)  # 1:1 perfect prediction line
axes[1].set_title(f"Relative Abundance\nR² = {r2_rel:.3f}")
axes[1].set_xlabel("True Age")
axes[1].set_ylabel("Predicted Age")

plt.tight_layout()
plt.savefig("gbr_comparison.png",format="png")
plt.show()

# Bootstrapping to see if differences are due to chance
n_boot = 5000
rng = np.random.default_rng(42)

diffs = []

n = len(abs_Y_test)

for _ in range(n_boot):
    idx = rng.choice(n, n, replace=True)
    
    y_sample = abs_Y_test.iloc[idx]
    abs_sample = abs_pred[idx]
    rel_sample = rel_pred[idx]
    
    r2_abs = r2_score(y_sample, abs_sample)
    r2_rel = r2_score(y_sample, rel_sample)
    
    diffs.append(r2_rel - r2_abs)

diffs = np.array(diffs)

ci_lower = np.percentile(diffs, 2.5)
ci_upper = np.percentile(diffs, 97.5)
mean_diff = diffs.mean()

print("Mean ΔR²:", mean_diff)
print("95% CI:", ci_lower, "to", ci_upper)
p_value = np.mean(diffs <= 0)
print("P-value:", p_value)

# Absolute
# Pipeline
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

# Fit
grid_search.fit(abs_X_train_aug, abs_Y_train)
abs_best_svr = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)
print("Best CV R²:", grid_search.best_score_)

# Validation Set
y_pred = abs_best_svr.predict(abs_X_val_aug)
abs_svm_val_r2 = r2_score(abs_Y_val, y_pred)
abs_svm_val_rmse = np.sqrt(mean_squared_error(abs_Y_val, y_pred))
abs_svm_val_mae = mean_absolute_error(abs_Y_val, y_pred)
print("Validation MAE:", abs_svm_val_mae)
print("Validation R²:", abs_svm_val_r2)
print("Validation RMSE:", abs_svm_val_rmse)

# Test Set
abs_svr_pred = abs_best_svr.predict(abs_X_test_aug)
abs_svm_test_r2 = r2_score(abs_Y_test, abs_svr_pred)
abs_svm_test_rmse = np.sqrt(mean_squared_error(abs_Y_test, abs_svr_pred))
abs_svm_test_mae = mean_absolute_error(abs_Y_test, abs_svr_pred)
print("Test MAE:", abs_svm_val_mae)
print("Test R²:", abs_svm_test_r2)
print("Test RMSE:", abs_svm_test_rmse)

# Relative
# Fit
grid_search.fit(rel_X_train_aug, rel_Y_train)
rel_best_svr = grid_search.best_estimator_

print("Best parameters:", grid_search.best_params_)
print("Best CV R²:", grid_search.best_score_)

# Validation Set
y_pred = rel_best_svr.predict(rel_X_val_aug)
rel_svm_val_r2 = r2_score(rel_Y_val, y_pred)
rel_svm_val_rmse = np.sqrt(mean_squared_error(rel_Y_val, y_pred))
rel_svm_val_mae = mean_absolute_error(abs_Y_val, y_pred)
print("Validation R²:", rel_svm_val_r2)
print("Validation RMSE:", rel_svm_val_rmse)
print("Validation MAE:", rel_svm_val_mae)

# Test Set
rel_svr_pred = abs_gbr.predict(rel_X_test_aug)
rel_svm_test_r2 = r2_score(rel_Y_test, rel_svr_pred)
rel_svm_test_rmse = np.sqrt(mean_squared_error(rel_Y_test, rel_svr_pred))
rel_svm_test_mae = mean_absolute_error(abs_Y_test, rel_svr_pred)
print("Test R²:", rel_svm_test_r2)
print("Test RMSE:", rel_svm_test_rmse)
print("Validation MAE:", rel_svm_test_mae)

# Plotting the results
r2_abs = r2_score(abs_Y_test, abs_svr_pred)
r2_rel = r2_score(rel_Y_test, rel_svr_pred)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

min_val = min(abs_Y_test.min(), rel_Y_test.min())
max_val = max(abs_Y_test.max(), rel_Y_test.max())
x_line = np.linspace(min_val, max_val, 100)

# Absolute
axes[0].scatter(abs_Y_test, abs_svr_pred, alpha=0.5)
axes[0].plot(x_line, x_line)  # 1:1 perfect prediction line
axes[0].set_title(f"Absolute Abundance RF\nR² = {r2_abs:.3f}")
axes[0].set_xlabel("True Age")
axes[0].set_ylabel("Predicted Age")

# Relative
axes[1].scatter(rel_Y_test, rel_svr_pred, alpha=0.5)
axes[1].plot(x_line, x_line)  # 1:1 perfect prediction line
axes[1].set_title(f"Relative Abundance RF\nR² = {r2_rel:.3f}")
axes[1].set_xlabel("True Age")
axes[1].set_ylabel("Predicted Age")

fig.suptitle("Absolute vs. Relative Abundance RBF SVM Results")
plt.savefig('rbf_comparison.png', format="png")

plt.tight_layout()
plt.show()

# Bootstrapping to see if differences are due to chance
n_boot = 5000
rng = np.random.default_rng(42)

diffs = []

n = len(abs_Y_test)

for _ in range(n_boot):
    idx = rng.choice(n, n, replace=True)
    
    y_sample = abs_Y_test.iloc[idx]
    abs_sample = abs_svr_pred[idx]
    rel_sample = rel_svr_pred[idx]
    
    r2_abs = r2_score(y_sample, abs_sample)
    r2_rel = r2_score(y_sample, rel_sample)
    
    diffs.append(r2_rel - r2_abs)

diffs = np.array(diffs)

ci_lower = np.percentile(diffs, 2.5)
ci_upper = np.percentile(diffs, 97.5)
mean_diff = diffs.mean()

print("Mean ΔR²:", mean_diff)
print("95% CI:", ci_lower, "to", ci_upper)
p_value = np.mean(diffs <= 0)
print("P-value:", p_value)

tax = pd.read_csv('/ddn_scratch/miter/nph-tables/wolr2-taxonomy.tsv', sep='\t')
tax['genus_raw'] = tax['d__Bacteria; p__Firmicutes; c__Bacilli; o__Bacillales_H; f__Bacillaceae_D; g__Bacillus_S; s__Bacillus_S pseudofirmus'].str.extract(r'g__([^;]+)')
age_bac = ['Haemophilus_D', 'Sutterella', 'Akkermansia', 'Phascolarctobacterium', 
           'Ruminiclostridium_E', 'Cloacibacillus', 'Pseudomonas', 'UBA1685', 
           'UBA10677', 'CAG-314', 'CAG-313', 'QAKW01']

# Retrieving column names from taxonomy mapping with values in age_bac
tax_rel_bac = tax[tax["genus_raw"].isin(age_bac)].copy()
col_names = tax_rel_bac['G000005825'].values

# Filtering train/test/val datasets to only use these columns
# Absolute quant
abs_X_train = abs_balanced_train.loc[:, abs_balanced_train.columns.isin(col_names)].copy()
abs_Y_train = abs_balanced_train['age']

abs_X_test = abs_test.loc[:, abs_test.columns.isin(col_names)].copy()
abs_Y_test  = abs_test['age']

abs_X_val = abs_val.loc[:, abs_val.columns.isin(col_names)].copy()
abs_Y_val  = abs_val['age']

# Relative abundance data
rel_X_train = rel_balanced_train.loc[:, rel_balanced_train.columns.isin(col_names)].copy()
rel_Y_train = rel_balanced_train['age']

rel_X_test = rel_test.loc[:, rel_test.columns.isin(col_names)].copy()
rel_Y_test  = rel_test['age']

rel_X_val = rel_val.loc[:, rel_val.columns.isin(col_names)].copy()
rel_Y_val  = rel_val['age']

# Preprocessing
# Using log transform to standardize the data
# Absolute quant data
abs_X_train_log = abs_X_train.copy()
abs_X_train_log = np.log1p(abs_X_train_log)

abs_X_test_log = abs_X_test.copy()
abs_X_test_log = np.log1p(abs_X_test_log)

abs_X_val_log = abs_X_val.copy()
abs_X_val_log = np.log1p(abs_X_val_log)

# Relative abundance data
rel_X_train_log = rel_X_train.copy()
rel_X_train_log = np.log1p(rel_X_train_log)

rel_X_test_log = rel_X_test.copy()
rel_X_test_log = np.log1p(rel_X_test_log)

rel_X_val_log = rel_X_val.copy()
rel_X_val_log = np.log1p(rel_X_val_log)

# Encoding sex
sex_encoder = OneHotEncoder(
    drop="first",           
    handle_unknown="ignore",
    sparse_output=False
)

sex_encoder.fit(abs_train[["sex"]])

# Absolute
abs_train_sex = sex_encoder.transform(abs_balanced_train[["sex"]])
abs_test_sex  = sex_encoder.transform(abs_test[["sex"]])
abs_val_sex   = sex_encoder.transform(abs_val[["sex"]])

# Relative
rel_train_sex = sex_encoder.transform(rel_balanced_train[["sex"]])
rel_test_sex  = sex_encoder.transform(rel_test[["sex"]])
rel_val_sex   = sex_encoder.transform(rel_val[["sex"]])

# Appending to training data
abs_X_train_aug = np.hstack([abs_X_train_log.values, abs_train_sex])
abs_X_test_aug  = np.hstack([abs_X_test_log.values,  abs_test_sex])
abs_X_val_aug   = np.hstack([abs_X_val_log.values,   abs_val_sex])

rel_X_train_aug = np.hstack([rel_X_train_log.values, rel_train_sex])
rel_X_test_aug  = np.hstack([rel_X_test_log.values,  rel_test_sex])
rel_X_val_aug   = np.hstack([rel_X_val_log.values,   rel_val_sex])

# Absolute
abs_rf_taxa = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
abs_rf_taxa.fit(abs_X_train_aug, abs_Y_train)

# Validation Set
abs_rf_taxa_val_pred = abs_rf_taxa.predict(abs_X_val_aug)
abs_rf_taxa_val_r2 = r2_score(abs_Y_val, y_pred)
abs_rf_taxa_val_rmse = np.sqrt(mean_squared_error(abs_Y_val, abs_rf_taxa_val_pred))
abs_rf_taxa_val_mae = mean_absolute_error(abs_Y_val, abs_rf_taxa_val_pred)
print("Validation MAE:", abs_rf_taxa_val_mae)
print("Validation R²:", abs_rf_taxa_val_r2)
print("Validation RMSE:", abs_rf_taxa_val_rmse)

# Test Set
abs_rf_taxa_test_pred = abs_rf_taxa.predict(abs_X_test_aug)
abs_rf_taxa_test_r2 = r2_score(abs_Y_test, abs_rf_taxa_test_pred)
abs_rf_taxa_test_rmse = np.sqrt(mean_squared_error(abs_Y_test, abs_rf_taxa_test_pred))
abs_rf_taxa_test_mae = mean_absolute_error(abs_Y_test, abs_rf_taxa_test_pred)
print("Test R²:", abs_rf_taxa_test_r2)
print("Test RMSE:", abs_rf_taxa_test_rmse)
print("Test MAE:", abs_rf_taxa_val_mae)

# Relative
rel_rf_taxa = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
rel_rf_taxa.fit(rel_X_train_aug, rel_Y_train)

# Validation Set
rel_rf_taxa_val_pred = rel_rf_taxa.predict(rel_X_val_aug)
rel_rf_taxa_val_r2 = r2_score(rel_Y_val, rel_rf_taxa_val_pred)
rel_rf_taxa_val_rmse = np.sqrt(mean_squared_error(rel_Y_val, rel_rf_taxa_val_pred))
rel_rf_taxa_val_mae = mean_absolute_error(rel_Y_val, rel_rf_taxa_val_pred)
print("Validation MAE:", rel_rf_taxa_val_mae)
print("Validation R²:", rel_rf_taxa_val_r2)
print("Validation RMSE:", rel_rf_taxa_val_rmse)

# Test Set
rel_rf_taxa_test_pred = rel_rf_taxa.predict(abs_X_test_aug)
rel_rf_taxa_test_r2 = r2_score(rel_Y_test, rel_rf_taxa_test_pred)
rel_rf_taxa_test_rmse = np.sqrt(mean_squared_error(rel_Y_test, rel_rf_taxa_test_pred))
rel_rf_taxa_test_mae = mean_absolute_error(rel_Y_test, rel_rf_taxa_test_pred)
print("Test R²:", rel_rf_taxa_test_r2)
print("Test RMSE:", rel_rf_taxa_test_rmse)
print("Test MAE:", rel_rf_taxa_val_mae)

# Plotting the results
abs_pred = abs_rf_taxa.predict(abs_X_test_aug)
rel_pred = rel_rf_taxa.predict(rel_X_test_aug)

r2_abs = r2_score(abs_Y_test, abs_rf_taxa_test_pred)
r2_rel = r2_score(rel_Y_test, rel_rf_taxa_test_pred)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

min_val = min(abs_Y_test.min(), rel_Y_test.min())
max_val = max(abs_Y_test.max(), rel_Y_test.max())
x_line = np.linspace(min_val, max_val, 100)

# Absolute
axes[0].scatter(abs_Y_test, abs_rf_taxa_test_pred, alpha=0.5)
axes[0].plot(x_line, x_line)  # 1:1 perfect prediction line
axes[0].set_title(f"Absolute Abundance\nR² = {r2_abs:.3f}")
axes[0].set_xlabel("True Age")
axes[0].set_ylabel("Predicted Age")

# Relative
axes[1].scatter(rel_Y_test, rel_rf_taxa_test_pred, alpha=0.5)
axes[1].plot(x_line, x_line)  # 1:1 perfect prediction line
axes[1].set_title(f"Relative Abundance\nR² = {r2_rel:.3f}")
axes[1].set_xlabel("True Age")
axes[1].set_ylabel("Predicted Age")

fig.suptitle('Absolute vs. Relative Abundance with Taxa as Features')
plt.tight_layout()
plt.savefig("rf_taxa_comparison.png",format="png")
plt.show()

# Bootstrapping to see if differences are due to chance
n_boot = 5000
rng = np.random.default_rng(42)

diffs = []

n = len(abs_Y_test)

for _ in range(n_boot):
    idx = rng.choice(n, n, replace=True)
    
    y_sample = abs_Y_test.iloc[idx]
    abs_sample = abs_rf_taxa_test_pred[idx]
    rel_sample = rel_rf_taxa_test_pred[idx]
    
    r2_abs = r2_score(y_sample, abs_sample)
    r2_rel = r2_score(y_sample, rel_sample)
    
    diffs.append(r2_rel - r2_abs)

diffs = np.array(diffs)

ci_lower = np.percentile(diffs, 2.5)
ci_upper = np.percentile(diffs, 97.5)
mean_diff = diffs.mean()

print("Mean ΔR²:", mean_diff)
print("95% CI:", ci_lower, "to", ci_upper)
p_value = np.mean(diffs <= 0)
print("P-value:", p_value)

# Creating the table
task = ['Regression', 'Regression', 'Regression', 'Regression', 'Regression', 'Regression', 'Regression', 'Regression']
target = ['age', 'age', 'age', 'age', 'age', 'age', 'age', 'age',]
representation = ['Absolute', 'Absolute', 'Absolute', 'Absolute', 'Relative', 'Relative', 'Relative', 'Relative']
model = ['RandomForest', 'GBR', 'SVM_RBF', 'RandomForest', 'RandomForest', 'GBR', 'SVM_RBF', 'RandomForest']
valMAE = [abs_rf_reg_val_mae, abs_gbr_val_mae, abs_svm_val_mae, abs_rf_taxa_val_mae, rel_rf_reg_val_mae, rel_gbr_val_mae, rel_svm_val_mae, rel_rf_taxa_val_mae]
testMAE = valMAE = [abs_rf_reg_test_mae, abs_gbr_test_mae, abs_svm_test_mae, abs_rf_taxa_test_mae, rel_rf_reg_test_mae, rel_gbr_test_mae, rel_svm_test_mae, rel_rf_taxa_test_mae]
valR2 = [abs_rf_reg_val_r2, abs_gbr_val_r2, abs_svm_val_r2, abs_rf_taxa_val_r2, rel_rf_reg_val_r2, rel_gbr_val_r2, rel_svm_val_r2, rel_rf_taxa_val_r2]
testR2 = [abs_rf_reg_test_r2, abs_gbr_test_r2, abs_svm_test_r2, abs_rf_taxa_test_r2, rel_rf_reg_test_r2, rel_gbr_test_r2, rel_svm_test_r2, rel_rf_taxa_test_r2]
valRMSE = [abs_rf_reg_val_rmse, abs_gbr_val_rmse, abs_svm_val_rmse, abs_rf_taxa_val_rmse, rel_rf_reg_val_rmse, rel_gbr_val_rmse, rel_svm_val_rmse, rel_rf_taxa_val_rmse]
testRMSE = [abs_rf_reg_test_rmse, abs_gbr_test_rmse, abs_svm_test_rmse, abs_rf_taxa_test_rmse, rel_rf_reg_test_rmse, rel_gbr_test_rmse, rel_svm_test_rmse, rel_rf_taxa_test_rmse]

results_df = pd.DataFrame({
    'Task': task,
    'Target': target,
    'Representation': representation,
    'Model': model,
    'Val_MAE': valMAE,
    'Test_MAE': testMAE,
    'Val_R2': valR2,
    'Test_R2': testR2,
    'Val_RMSE': valRMSE,
    'Test_RMSE': testRMSE
})

results_df = results_df.round(6)
results_df

results_df.to_csv("age_reg_model_comparisons.csv", index=False)
