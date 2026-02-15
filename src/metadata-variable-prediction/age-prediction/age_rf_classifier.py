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
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error, mean_squared_error, r2_score, roc_curve, auc, roc_auc_score, f1_score

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

# Basic RandomForestClassifier
# Loading in data
abs_train = pd.read_csv('/ddn_scratch/k5zhao/data/classifier_training/abs_train.csv')
abs_test = pd.read_csv('/ddn_scratch/k5zhao/data/classifier_training/abs_test.csv')
abs_val = pd.read_csv('/ddn_scratch/k5zhao/data/classifier_training/abs_val.csv')

rel_train = pd.read_csv('/ddn_scratch/k5zhao/data/classifier_training/rel_train.csv')
rel_test = pd.read_csv('/ddn_scratch/k5zhao/data/classifier_training/rel_test.csv')
rel_val = pd.read_csv('/ddn_scratch/k5zhao/data/classifier_training/rel_val.csv')

# Creating X and Y features
# Using American Gut Microbiome Project age cutoffs, remove outliers
c_abs_train = abs_train.loc[(abs_train["age"] >= 20) & (abs_train["age"] <= 69)]
c_abs_test = abs_test.loc[(abs_test["age"] >= 20) & (abs_test["age"] <= 69)]
c_abs_val = abs_val.loc[(abs_val["age"] >= 20) & (abs_val["age"] <= 69)]

c_rel_train = rel_train.loc[(rel_train["age"] >= 20) & (rel_train["age"] <= 69)]
c_rel_test = rel_test.loc[(rel_test["age"] >= 20) & (rel_test["age"] <= 69)]
c_rel_val = rel_val.loc[(rel_val["age"] >= 20) & (rel_val["age"] <= 69)]

# Absolute quant data
c_abs_X_train = c_abs_train[c_abs_train.columns[:1148]].drop(columns=['original_SampleID'])
c_abs_Y_train = c_abs_train['age']

c_abs_X_test = c_abs_test[c_abs_test.columns[:1148]].drop(columns=['original_SampleID'])
c_abs_Y_test  = c_abs_test['age']

c_abs_X_val = c_abs_val[c_abs_val.columns[:1148]].drop(columns=['original_SampleID'])
c_abs_Y_val  = c_abs_val['age']

# Relative abundance data
c_rel_X_train = c_rel_train[c_rel_train.columns[:1148]].drop(columns=['original_SampleID'])
c_rel_Y_train = c_rel_train['age']

c_rel_X_test = c_rel_test[c_rel_test.columns[:1148]].drop(columns=['original_SampleID'])
c_rel_Y_test  = c_rel_test['age']

c_rel_X_val = c_rel_val[c_rel_val.columns[:1148]].drop(columns=['original_SampleID'])
c_rel_Y_val  = c_rel_val['age']

# Preprocessing
# Using log transform to standardize the data
# Absolute quant data
c_abs_X_train_log = c_abs_X_train.copy()
c_abs_X_train_log = np.log1p(c_abs_X_train_log)

c_abs_X_test_log = c_abs_X_test.copy()
c_abs_X_test_log = np.log1p(c_abs_X_test_log)

c_abs_X_val_log = c_abs_X_val.copy()
c_abs_X_val_log = np.log1p(c_abs_X_val_log)

# Relative abundance data
c_rel_X_train_log = c_rel_X_train.copy()
c_rel_X_train_log = np.log1p(c_rel_X_train_log)

c_rel_X_test_log = c_rel_X_test.copy()
c_rel_X_test_log = np.log1p(c_rel_X_test_log)

c_rel_X_val_log = c_rel_X_val.copy()
c_rel_X_val_log = np.log1p(c_rel_X_val_log)

# Testing classification on bins of 10 years first
bins = [20, 30, 40, 50, 60, 70]

c_abs_Y_class_train = pd.cut(
    c_abs_Y_train,
    bins=bins,
    right=False,
    include_lowest=True
).astype(str)

c_abs_Y_class_test=pd.cut(
    c_abs_Y_test,
    bins=bins,  
    right=False,
    include_lowest=True
).astype(str)

c_abs_Y_class_val=pd.cut(
    c_abs_Y_val,
    bins=bins,  
    right=False,
    include_lowest=True
).astype(str)

c_rel_Y_class_train = pd.cut(
    c_rel_Y_train,
    bins=bins,
    right=False,
    include_lowest=True
).astype(str)

c_rel_Y_class_test=pd.cut(
    c_rel_Y_test,
    bins=bins,  
    right=False,
    include_lowest=True
).astype(str)

c_rel_Y_class_val=pd.cut(
    c_rel_Y_val,
    bins=bins,  
    right=False,
    include_lowest=True
).astype(str)

# Absolute Quant Random Forest Classifier
# Training the model on absolute quant data
abs_rf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

abs_rf.fit(c_abs_X_train_log, c_abs_Y_class_train)

# Evaluating the model
val_preds = abs_rf.predict(c_abs_X_val_log)
print("Validation accuracy:", accuracy_score(c_abs_Y_class_val, val_preds))

# confusion matrix
cm = confusion_matrix(c_abs_Y_class_val, val_preds)
labels = abs_rf.classes_
cm_norm = cm / cm.sum(axis=1, keepdims=True)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)
plt.xlabel("Predicted age bin")
plt.ylabel("True age bin")
plt.title("Normalized Confusion Matrix (Row-wise)")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Plotting the ROC Curves
c_abs_y_pred_prob = abs_rf.predict_proba(c_abs_X_test_log)
c_abs_Y_test_cat = c_abs_Y_class_test.astype("category")
c_abs_Y_test_codes = c_abs_Y_test_cat.cat.codes

classes = list(range(len(c_abs_Y_test_cat.cat.categories)))

c_abs_y_test_bin = label_binarize(c_abs_Y_test_codes, classes=classes)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(classes)):
    fpr[i], tpr[i], _ = roc_curve(
        c_abs_y_test_bin[:, i],
        c_abs_y_pred_prob[:, i]
    )
    roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8,6))

bin_labels = c_abs_Y_test_cat.cat.categories

for i in range(len(classes)):
    plt.plot(
        fpr[i],
        tpr[i],
        lw=2,
        label=f"{bin_labels[i]} (AUC = {roc_auc[i]:.2f})"
    )

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Absolute Abundance Model")
plt.legend(loc="lower right")
plt.show()

# Macro AUC
macro_auc = roc_auc_score(
    c_abs_y_test_bin,
    c_abs_y_pred_prob,
    average="macro"
)
print("Macro-average AUC:", macro_auc)

# AUC for each class
for i in range(len(roc_auc)):
    print(bin_labels[i], roc_auc[i])

# Relative Abundance Random Forest Classifier
# Training the model on absolute quant data
# Treating the classes as more balanced to see if it improves accuracy
rel_rf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rel_rf.fit(c_rel_X_train_log, c_rel_Y_class_train)

# Evaluating the model
val_preds = rel_rf.predict(c_rel_X_val_log)
print("Validation accuracy:", accuracy_score(c_rel_Y_class_val, val_preds))

# Confusion matrix
cm = confusion_matrix(c_rel_Y_class_val, val_preds)
labels = rel_rf.classes_
cm_norm = cm / cm.sum(axis=1, keepdims=True)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)
plt.xlabel("Predicted age bin")
plt.ylabel("True age bin")
plt.title("Normalized Confusion Matrix (Row-wise)")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Plotting the ROC Curves
c_rel_y_pred_prob = rel_rf.predict_proba(c_rel_X_test_log)
c_rel_Y_test_cat = c_rel_Y_class_test.astype("category")
c_rel_Y_test_codes = c_rel_Y_test_cat.cat.codes

classes = list(range(len(c_rel_Y_test_cat.cat.categories)))
c_rel_y_test_bin = label_binarize(c_rel_Y_test_codes, classes=classes)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(classes)):
    fpr[i], tpr[i], _ = roc_curve(
        c_abs_y_test_bin[:, i],
        c_abs_y_pred_prob[:, i]
    )
    roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8,6))

bin_labels = c_rel_Y_test_cat.cat.categories

for i in range(len(classes)):
    plt.plot(
        fpr[i],
        tpr[i],
        lw=2,
        label=f"{bin_labels[i]} (AUC = {roc_auc[i]:.2f})"
    )

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Absolute Abundance Model")
plt.legend(loc="lower right")
plt.show()

# Macro AUC
macro_auc = roc_auc_score(
    c_rel_y_test_bin,
    c_rel_y_pred_prob,
    average="macro"
)
print("Macro-average AUC:", macro_auc)

# AUC for each class
for i in range(len(roc_auc)):
    print(bin_labels[i], roc_auc[i])

# Tuning: Changing bin sizes and adding classification categories for outliers
# Absolute quant data
abs_X_train = abs_train[abs_train.columns[:1148]].drop(columns=['original_SampleID'])
abs_Y_train = abs_train['age']

abs_X_test = abs_test[abs_test.columns[:1148]].drop(columns=['original_SampleID'])
abs_Y_test  = abs_test['age']

abs_X_val = abs_val[abs_val.columns[:1148]].drop(columns=['original_SampleID'])
abs_Y_val  = abs_val['age']

# Relative abundance data
rel_X_train = rel_train[rel_train.columns[:1148]].drop(columns=['original_SampleID'])
rel_Y_train = rel_train['age']

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

# Rebinning
bins = [0, 20, 35, 50, 65, 80, np.inf]

abs_Y_class_train = pd.cut(
    abs_Y_train,
    bins=bins,
    right=False,
    include_lowest=True
).astype(str)

abs_Y_class_test=pd.cut(
    abs_Y_test,
    bins=bins,  
    right=False,
    include_lowest=True
).astype(str)

abs_Y_class_val=pd.cut(
    abs_Y_val,
    bins=bins,  
    right=False,
    include_lowest=True
).astype(str)

rel_Y_class_train = pd.cut(
    rel_Y_train,
    bins=bins,
    right=False,
    include_lowest=True
).astype(str)

rel_Y_class_test=pd.cut(
    rel_Y_test,
    bins=bins,  
    right=False,
    include_lowest=True
).astype(str)

rel_Y_class_val=pd.cut(
    rel_Y_val,
    bins=bins,  
    right=False,
    include_lowest=True
).astype(str)

# Re-Tuned Absolute Quant Random Forest Classifier
# Training the model on absolute quant data
# Treating the classes as more balanced to see if it improves accuracy
abs_rf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
abs_rf.fit(abs_X_train_log, abs_Y_class_train)

# Evaluating the model
val_preds = abs_rf.predict(abs_X_val_log)
print("Validation accuracy:", accuracy_score(abs_Y_class_val, val_preds))

# confusion matrix
cm = confusion_matrix(abs_Y_class_val, val_preds)
labels = abs_rf.classes_
cm_norm = cm / cm.sum(axis=1, keepdims=True)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)
plt.xlabel("Predicted age bin")
plt.ylabel("True age bin")
plt.title("Normalized Confusion Matrix (Row-wise)")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Plotting the ROC Curves
abs_y_pred_prob = abs_rf.predict_proba(abs_X_test_log)
abs_Y_test_cat = abs_Y_class_test.astype("category")
abs_Y_test_codes = abs_Y_test_cat.cat.codes

classes = list(range(len(abs_Y_test_cat.cat.categories)))

abs_y_test_bin = label_binarize(abs_Y_test_codes, classes=classes)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(classes)):
    fpr[i], tpr[i], _ = roc_curve(
        abs_y_test_bin[:, i],
        abs_y_pred_prob[:, i]
    )
    roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8,6))

bin_labels = abs_Y_test_cat.cat.categories

for i in range(len(classes)):
    plt.plot(
        fpr[i],
        tpr[i],
        lw=2,
        label=f"{bin_labels[i]} (AUC = {roc_auc[i]:.2f})"
    )

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Absolute Abundance Model")
plt.legend(loc="lower right")
plt.show()

# Macro AUC
macro_auc = roc_auc_score(
    abs_y_test_bin,
    abs_y_pred_prob,
    average="macro"
)
print("Macro-average AUC:", macro_auc)

# AUC for each class
for i in range(len(roc_auc)):
    print(bin_labels[i], roc_auc[i])

# Re-Tuned Relative Abundance Random Forest Classifier
# Training the model on absolute quant data
# Treating the classes as more balanced to see if it improves accuracy
rel_rf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rel_rf.fit(rel_X_train_log, rel_Y_class_train)

# Evaluating the model
val_preds = rel_rf.predict(rel_X_val_log)
print("Validation accuracy:", accuracy_score(rel_Y_class_val, val_preds))

# confusion matrix
cm = confusion_matrix(rel_Y_class_val, val_preds)
labels = rel_rf.classes_
cm_norm = cm / cm.sum(axis=1, keepdims=True)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)
plt.xlabel("Predicted age bin")
plt.ylabel("True age bin")
plt.title("Normalized Confusion Matrix (Row-wise)")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Plotting the ROC Curves
rel_y_pred_prob = rel_rf.predict_proba(rel_X_test_log)
rel_Y_test_cat = rel_Y_class_test.astype("category")
rel_Y_test_codes = rel_Y_test_cat.cat.codes

classes = list(range(len(rel_Y_test_cat.cat.categories)))

rel_y_test_bin = label_binarize(rel_Y_test_codes, classes=classes)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(classes)):
    fpr[i], tpr[i], _ = roc_curve(
        abs_y_test_bin[:, i],
        abs_y_pred_prob[:, i]
    )
    roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8,6))

bin_labels = rel_Y_test_cat.cat.categories

for i in range(len(classes)):
    plt.plot(
        fpr[i],
        tpr[i],
        lw=2,
        label=f"{bin_labels[i]} (AUC = {roc_auc[i]:.2f})"
    )

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Absolute Abundance Model")
plt.legend(loc="lower right")
plt.show()

# Macro AUC
macro_auc = roc_auc_score(
    rel_y_test_bin,
    rel_y_pred_prob,
    average="macro"
)
print("Macro-average AUC:", macro_auc)

# AUC for each class
for i in range(len(roc_auc)):
    print(bin_labels[i], roc_auc[i])

# Final performance evaluation
# Combining training and validation sets for training
abs_train = pd.concat([abs_train, abs_val])
rel_train = pd.concat([rel_train, rel_val])

# Absolute quant data
abs_X_train = abs_train[abs_train.columns[:1148]].drop(columns=['original_SampleID'])
abs_Y_train = abs_train['age']

abs_X_test = abs_test[abs_test.columns[:1148]].drop(columns=['original_SampleID'])
abs_Y_test  = abs_test['age']

# Relative abundance data
rel_X_train = rel_train[rel_train.columns[:1148]].drop(columns=['original_SampleID'])
rel_Y_train = rel_train['age']

rel_X_test = rel_test[rel_test.columns[:1148]].drop(columns=['original_SampleID'])
rel_Y_test  = rel_test['age']

# Preprocessing
# Using log transform to standardize the data
# Absolute quant data
abs_X_train_log = abs_X_train.copy()
abs_X_train_log = np.log1p(abs_X_train_log)

abs_X_test_log = abs_X_test.copy()
abs_X_test_log = np.log1p(abs_X_test_log)

# Relative abundance data
rel_X_train_log = rel_X_train.copy()
rel_X_train_log = np.log1p(rel_X_train_log)

rel_X_test_log = rel_X_test.copy()
rel_X_test_log = np.log1p(rel_X_test_log)

# Rebinning
bins = [0, 20, 35, 50, 65, 80, np.inf]

abs_Y_class_train = pd.cut(
    abs_Y_train,
    bins=bins,
    right=False,
    include_lowest=True
).astype(str)

abs_Y_class_test=pd.cut(
    abs_Y_test,
    bins=bins,  
    right=False,
    include_lowest=True
).astype(str)

rel_Y_class_train = pd.cut(
    rel_Y_train,
    bins=bins,
    right=False,
    include_lowest=True
).astype(str)

rel_Y_class_test=pd.cut(
    rel_Y_test,
    bins=bins,  
    right=False,
    include_lowest=True
).astype(str)

# Fitting the models
# Absolute quant Random Forest Classifier
abs_rf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
abs_rf.fit(abs_X_train_log, abs_Y_class_train)

# Relative abundance Random Forest Classifier
rel_rf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rel_rf.fit(rel_X_train_log, rel_Y_class_train)

# Evaluating the model
# Absolute
y_abs_pred = abs_rf.predict(abs_X_test_log)
y_abs_prob = abs_rf.predict_proba(abs_X_test_log)

# Relative
y_rel_pred = rel_rf.predict(rel_X_test_log)
y_rel_prob = rel_rf.predict_proba(rel_X_test_log)

# Accuracy
acc_abs = accuracy_score(abs_Y_class_test, y_abs_pred)
acc_rel = accuracy_score(rel_Y_class_test, y_rel_pred)
print("Absolute Accuracy:", acc_abs)
print("Relative Accuracy:", acc_rel)

# Macro F-1 Score
f1_abs = f1_score(abs_Y_class_test, y_abs_pred, average="macro")
f1_rel = f1_score(rel_Y_class_test, y_rel_pred, average="macro")
print("Absolute Macro-F1:", f1_abs)
print("Relative Macro-F1:", f1_rel)

# Convert string labels to categorical codes for ROC/AUC
abs_test_codes = abs_Y_class_test.astype("category").cat.codes
rel_test_codes = rel_Y_class_test.astype("category").cat.codes

# Get the number of classes
classes = range(len(abs_Y_class_test.astype("category").cat.categories))

# Binarize
abs_y_test_bin = label_binarize(abs_test_codes, classes=classes)
rel_y_test_bin = label_binarize(rel_test_codes, classes=classes)

# Predicted probabilities
abs_y_prob = abs_rf.predict_proba(abs_X_test_log)
rel_y_prob = rel_rf.predict_proba(rel_X_test_log)

# Compute macro-AUC
auc_abs = roc_auc_score(abs_y_test_bin, abs_y_prob, average="macro")
auc_rel = roc_auc_score(rel_y_test_bin, rel_y_prob, average="macro")
print("Absolute Macro-AUC:", auc_abs)
print("Relative Macro-AUC:", auc_rel)

# Plot ROC curves
plt.figure(figsize=(10,6))

bin_labels = abs_Y_class_test.astype("category").cat.categories

for i in classes:
    fpr_abs, tpr_abs, _ = roc_curve(abs_y_test_bin[:, i], abs_y_prob[:, i])
    fpr_rel, tpr_rel, _ = roc_curve(rel_y_test_bin[:, i], rel_y_prob[:, i])
    plt.plot(fpr_abs, tpr_abs, lw=2, label=f"Abs {bin_labels[i]}")
    plt.plot(fpr_rel, tpr_rel, lw=2, linestyle='--', label=f"Rel {bin_labels[i]}")

plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — Absolute vs Relative Abundance")
plt.legend(loc="lower right")
plt.show()

# confusion matrices side by side
# Absolute abundance
disp_abs = ConfusionMatrixDisplay.from_predictions(
    abs_Y_class_test, 
    y_abs_pred, 
    normalize='true',
    cmap='Blues'
)
plt.xticks(rotation=45)  # rotate x-axis labels
plt.show()

# Relative abundance
disp_rel = ConfusionMatrixDisplay.from_predictions(
    rel_Y_class_test, 
    y_rel_pred, 
    normalize='true',
    cmap='Greens'
)
plt.xticks(rotation=45)  # rotate x-axis labels
plt.show()

# Checking if results are due to chance via bootstrapping
n_boot = 1000

acc_diffs = []
auc_diffs = []

abs_test_codes = abs_Y_class_test.astype("category").cat.codes
rel_test_codes = rel_Y_class_test.astype("category").cat.codes
classes = range(len(abs_Y_class_test.astype("category").cat.categories))

abs_y_test_bin = label_binarize(abs_test_codes, classes=classes)
rel_y_test_bin = label_binarize(rel_test_codes, classes=classes)

abs_y_prob = abs_rf.predict_proba(abs_X_test_log)
rel_y_prob = rel_rf.predict_proba(rel_X_test_log)

y_abs_pred = abs_rf.predict(abs_X_test_log)
y_rel_pred = rel_rf.predict(rel_X_test_log)

# Stratified bootstrap helper function
# Makes sure each sample has amounts from each class
def stratified_bootstrap_indices(y):
    indices = []
    for c in np.unique(y):
        class_indices = np.where(y == c)[0]
        resampled = resample(class_indices, replace=True, n_samples=len(class_indices))
        indices.extend(resampled)
    return np.array(indices)

# Bootstrap loop
for i in range(n_boot):
    indices = stratified_bootstrap_indices(abs_test_codes)

    # Calculate accuracy
    acc_abs_i = accuracy_score(abs_Y_class_test.iloc[indices], y_abs_pred[indices])
    acc_rel_i = accuracy_score(rel_Y_class_test.iloc[indices], y_rel_pred[indices])
    acc_diffs.append(acc_abs_i - acc_rel_i)

    # Calculate macro-AUC
    auc_abs_i = roc_auc_score(abs_y_test_bin[indices], abs_y_prob[indices], average='macro')
    auc_rel_i = roc_auc_score(rel_y_test_bin[indices], rel_y_prob[indices], average='macro')
    auc_diffs.append(auc_abs_i - auc_rel_i)

# Compute 95% confidence intervals
acc_ci = np.percentile(acc_diffs, [2.5, 97.5])
auc_ci = np.percentile(auc_diffs, [2.5, 97.5])

print("Accuracy difference 95% CI (Abs - Rel):", acc_ci)
print("Macro-AUC difference 95% CI (Abs - Rel):", auc_ci)

# Interpretation
if acc_ci[0] > 0 or acc_ci[1] < 0:
    print("Accuracy difference is statistically significant.")
else:
    print("Accuracy difference is NOT statistically significant.")

if auc_ci[0] > 0 or auc_ci[1] < 0:
    print("AUC difference is statistically significant.")
else:
    print("AUC difference is NOT statistically significant.")

# Feature importance
abs_feat_importance = pd.DataFrame({
    "feature": abs_X_train.columns,
    "importance": abs_rf.feature_importances_
}).sort_values(by="importance", ascending=False)

rel_feat_importance = pd.DataFrame({
    "feature": rel_X_train.columns,
    "importance": rel_rf.feature_importances_
}).sort_values(by="importance", ascending=False)

# Absolute
top_n = 20

plt.figure(figsize=(8,6))
plt.barh(
    abs_feat_importance["feature"][:top_n][::-1],
    abs_feat_importance["importance"][:top_n][::-1]
)
plt.title("Top 20 Absolute Abundance Features")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# relative
top_n = 20
plt.figure(figsize=(8,6))
plt.barh(
    rel_feat_importance["feature"][:top_n][::-1],
    rel_feat_importance["importance"][:top_n][::-1]
)
plt.title("Top 20 Relative Abundance Features")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# Top 20 feature names
abs_top20 = set(abs_feat_importance["feature"].head(20))
rel_top20 = set(rel_feat_importance["feature"].head(20))

# Overlapping features
overlap = abs_top20.intersection(rel_top20)
print("Overlap:", overlap)
print("Number overlapping:", len(overlap))

# Unique features
print("Unique to Absolute:", abs_top20 - rel_top20)
print("Unique to Relative:", rel_top20 - abs_top20)