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
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error, mean_squared_error, r2_score, roc_curve, auc, roc_auc_score, f1_score

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

# Loading in features from absolute quantitative feature table
# This is just to see what columns are present vs. the joined metadata
artifact = qiime2.Artifact.load("/ddn_scratch/k5zhao/data/metaG-absquant-clean.qza")
df = artifact.view(pd.DataFrame)
df.head()

df.columns[:1148]

df.index

# Loading in absolute quant metadata
# This is just so I can better see the columns
metadata = pd.read_csv('/ddn_scratch/k5zhao/data/metadata_absquant_clean.tsv', sep='\t')
metadata.head()

# Loading in absolute quant data for train/test/validation split
# Should run log transformation
abs_train = pd.read_csv('/ddn_scratch/k5zhao/data/classifier_training/abs_train.csv')
abs_test = pd.read_csv('/ddn_scratch/k5zhao/data/classifier_training/abs_test.csv')
abs_val = pd.read_csv('/ddn_scratch/k5zhao/data/classifier_training/abs_val.csv')

rel_train = pd.read_csv('/ddn_scratch/k5zhao/data/classifier_training/rel_train.csv')
rel_test = pd.read_csv('/ddn_scratch/k5zhao/data/classifier_training/rel_test.csv')
rel_val = pd.read_csv('/ddn_scratch/k5zhao/data/classifier_training/rel_val.csv')

abs_train.head()

abs_train.columns[:1148]

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

# Training the model on absolute quant data
# Treating the classes as more balanced to see if it improves accuracy
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

cm = confusion_matrix(c_abs_Y_class_val, val_preds)
cm

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
plt.savefig("first_pass_abs_rf_cm.png", format="png")
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
plt.savefig("first_pass_abs_rf_roc.png", format="png")
plt.show()

# Macro AUC
macro_auc = roc_auc_score(
    c_abs_y_test_bin,
    c_abs_y_pred_prob,
    average="macro"
)

print("Macro-average AUC:", macro_auc)

for i in range(len(roc_auc)):
    print(bin_labels[i], roc_auc[i])

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
plt.savefig("first_pass_rel_rf_cm.png", format="png")
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
plt.savefig("first_pass_rel_rf_roc.png", format="png")
plt.show()

# Macro AUC
macro_auc = roc_auc_score(
    c_rel_y_test_bin,
    c_rel_y_pred_prob,
    average="macro"
)

print("Macro-average AUC:", macro_auc)

for i in range(len(roc_auc)):
    print(bin_labels[i], roc_auc[i])


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
plt.savefig("retuned_abs_rf_cm.png", format="png")
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
plt.savefig("retuned_abs_rf_roc.png", format="png")
plt.show()

# Macro AUC
macro_auc = roc_auc_score(
    abs_y_test_bin,
    abs_y_pred_prob,
    average="macro"
)

print("Macro-average AUC:", macro_auc)

for i in range(len(roc_auc)):
    print(bin_labels[i], roc_auc[i])

## Re-tuned Relative Quant Classifier

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
plt.savefig("retuned_rel_rf_cm.png", format="png")
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
plt.savefig("retuned_rel_rf_roc.png", format="png")
plt.show()

# Macro AUC
macro_auc = roc_auc_score(
    rel_y_test_bin,
    rel_y_pred_prob,
    average="macro"
)

print("Macro-average AUC:", macro_auc)

for i in range(len(roc_auc)):
    print(bin_labels[i], roc_auc[i])

# Loading in absolute quant data for train/test/validation split
# Should run log transformation
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

abs_rf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

abs_rf.fit(abs_X_train_log, abs_Y_class_train)

rel_rf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rel_rf.fit(rel_X_train_log, rel_Y_class_train)

# Absolute
y_abs_pred = abs_rf.predict(abs_X_test_log)
y_abs_prob = abs_rf.predict_proba(abs_X_test_log)

# Relative
y_rel_pred = rel_rf.predict(rel_X_test_log)
y_rel_prob = rel_rf.predict_proba(rel_X_test_log)

acc_abs = accuracy_score(abs_Y_class_test, y_abs_pred)
acc_rel = accuracy_score(rel_Y_class_test, y_rel_pred)

print("Absolute Accuracy:", acc_abs)
print("Relative Accuracy:", acc_rel)

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
plt.savefig("final_rf_roc.png", format="png")
plt.show()

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

plt.savefig("final_rf_cm.png", format="png")

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
    print("AUC difference is NOT statistically significant!")

abs_feat_importance = pd.DataFrame({
    "feature": abs_X_train.columns,
    "importance": abs_rf.feature_importances_
}).sort_values(by="importance", ascending=False)

rel_feat_importance = pd.DataFrame({
    "feature": rel_X_train.columns,
    "importance": rel_rf.feature_importances_
}).sort_values(by="importance", ascending=False)

top_n = 20

plt.figure(figsize=(8,6))
plt.barh(
    abs_feat_importance["feature"][:top_n][::-1],
    abs_feat_importance["importance"][:top_n][::-1]
)
plt.title("Top 20 Absolute Abundance Features")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("final_abs_rf_top20_features.png", format="png")
plt.show()

top_n = 20

plt.figure(figsize=(8,6))
plt.barh(
    rel_feat_importance["feature"][:top_n][::-1],
    rel_feat_importance["importance"][:top_n][::-1]
)
plt.title("Top 20 Relative Abundance Features")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("final_rel_rf_top20_features.png", format="png")
plt.show()

# Top 20 feature names
abs_top20 = set(abs_feat_importance["feature"].head(20))
rel_top20 = set(rel_feat_importance["feature"].head(20))

overlap = abs_top20.intersection(rel_top20)
print("Overlap:", overlap)
print("Number overlapping:", len(overlap))

print("Unique to Absolute:", abs_top20 - rel_top20)
print("Unique to Relative:", rel_top20 - abs_top20)

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
target_n = 300

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
abs_Y_train = abs_balanced_train['age_bin']

abs_X_test = abs_test[abs_test.columns[:1148]].drop(columns=['original_SampleID'])
abs_Y_test  = abs_test['age_bin']

abs_X_val = abs_val[abs_val.columns[:1148]].drop(columns=['original_SampleID'])
abs_Y_val  = abs_val['age_bin']

# Relative abundance data
rel_X_train = rel_balanced_train[rel_balanced_train.columns[:1148]].drop(columns=['original_SampleID'])
rel_Y_train = rel_balanced_train['age_bin']

rel_X_test = rel_test[rel_test.columns[:1148]].drop(columns=['original_SampleID'])
rel_Y_test  = rel_test['age_bin']

rel_X_val = rel_val[rel_val.columns[:1148]].drop(columns=['original_SampleID'])
rel_Y_val  = rel_val['age_bin']

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

# Absolute
# Grid Search
param_grid = {
    "n_estimators": [500, 800, 1000],
    "max_depth": [None, 10, 20, 40],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}

rf = RandomForestClassifier(
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc_ovr",
    n_jobs=-1,
    verbose=2
)

grid_search.fit(abs_X_train_log, abs_Y_train)

abs_best_rf = grid_search.best_estimator_

print("Best parameters:", grid_search.best_params_)
print("Best CV score:", grid_search.best_score_)

# Absolute
# Validation
val_preds = abs_best_rf.predict(abs_X_val_log)
print("Validation accuracy:", accuracy_score(abs_Y_val, val_preds))

# Test
y_pred = abs_best_rf.predict(abs_X_test_log)
abs_bal_accuracy = accuracy_score(abs_Y_test, y_pred)
print("Test Accuracy:", abs_bal_accuracy)

# Without a grid search
abs_rf_nogrid = RandomForestClassifier(
    n_estimators=500,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

abs_rf_nogrid.fit(abs_X_train_log, abs_Y_train)

# Validation
val_preds = abs_rf_nogrid.predict(abs_X_test_log)
print("Validation accuracy:", accuracy_score(abs_Y_test, val_preds))

# Test
y_pred = abs_best_rf.predict(abs_X_test_log)
abs_bal_nogrid_accuracy = accuracy_score(abs_Y_test, y_pred)
print("Test Accuracy:", abs_bal_nogrid_accuracy)

# Relative
# Grid Search
param_grid = {
    "n_estimators": [500, 800, 1000],
    "max_depth": [None, 10, 20, 40],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}

rf = RandomForestClassifier(
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc_ovr",
    n_jobs=-1,
    verbose=2
)

grid_search.fit(rel_X_train_log, rel_Y_train)

rel_best_rf = grid_search.best_estimator_

print("Best parameters:", grid_search.best_params_)
print("Best CV score:", grid_search.best_score_)

# Absolute
# Validation
val_preds = rel_best_rf.predict(rel_X_test_log)
print("Validation accuracy:", accuracy_score(rel_Y_test, val_preds))

# Test
y_pred = rel_best_rf.predict(rel_X_test_log)
rel_bal_accuracy = accuracy_score(rel_Y_test, y_pred)
print("Test Accuracy:", rel_bal_accuracy)

# Without a grid search
rel_rf_nogrid = RandomForestClassifier(
    n_estimators=500,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rel_rf_nogrid.fit(rel_X_train_log, rel_Y_train)

# Validation
val_preds = rel_rf_nogrid.predict(rel_X_test_log)
print("Validation accuracy:", accuracy_score(rel_Y_test, val_preds))

# Test
y_pred = rel_best_rf.predict(rel_X_test_log)
rel_bal_nogrid_accuracy = accuracy_score(rel_Y_test, y_pred)
print("Test Accuracy:", rel_bal_nogrid_accuracy)

tax = pd.read_csv('/ddn_scratch/miter/nph-tables/wolr2-taxonomy.tsv', sep='\t')
tax['genus_raw'] = tax['d__Bacteria; p__Firmicutes; c__Bacilli; o__Bacillales_H; f__Bacillaceae_D; g__Bacillus_S; s__Bacillus_S pseudofirmus'].str.extract(r'g__([^;]+)')

# Idea: grab the columns more significantly correlated with age and use those for training instead
# Strong age correlation
age_bac = ['Haemophilus_D', 'Sutterella', 'Akkermansia', 'Phascolarctobacterium',
           'Ruminiclostridium_E', 'Cloacibacillus', 'Pseudomonas', 'UBA1685',
           'UBA10677', 'CAG-314', 'CAG-313', 'QAKW01']

# Retrieving column names from taxonomy mapping with values in age_bac
tax_rel_bac = tax[tax["genus_raw"].isin(age_bac)].copy()
col_names = tax_rel_bac['G000005825'].values
col_names

# Filtering train/test/val datasets to only use these columns
# Absolute quant
abs_X_train = abs_balanced_train.loc[:, abs_balanced_train.columns.isin(col_names)].copy()
abs_Y_train = abs_balanced_train['age_bin']

abs_X_test = abs_test.loc[:, abs_test.columns.isin(col_names)].copy()
abs_Y_test  = abs_test['age_bin']

abs_X_val = abs_val.loc[:, abs_val.columns.isin(col_names)].copy()
abs_Y_val  = abs_val['age_bin']

# Relative abundance data
rel_X_train = rel_balanced_train.loc[:, rel_balanced_train.columns.isin(col_names)].copy()
rel_Y_train = rel_balanced_train['age_bin']

rel_X_test = rel_test.loc[:, rel_test.columns.isin(col_names)].copy()
rel_Y_test  = rel_test['age_bin']

rel_X_val = rel_val.loc[:, rel_val.columns.isin(col_names)].copy()
rel_Y_val  = rel_val['age_bin']

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

# Training the RF
# Absolute
rel_rf_sex_gene = RandomForestClassifier(
    n_estimators=500,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rel_rf_sex_gene.fit(rel_X_train_aug, rel_Y_train)

# Validation set
val_preds = rel_rf_sex_gene.predict(rel_X_val_aug)
print("Validation accuracy:", accuracy_score(rel_Y_val, val_preds))

# Test
y_pred = rel_rf_sex_gene.predict(rel_X_test_aug)
rel_rf_sex_gene_accuracy = accuracy_score(rel_Y_test, y_pred)
print("Test Accuracy:", rel_rf_sex_gene_accuracy)

cm = confusion_matrix(rel_Y_test, y_pred)
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
plt.title("Normalized Confusion Matrix (Row-wise) for Relative Abundance RF Using Sex and Genes as Features")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("age_abs_rf_sex_gene_cm.png", format="png")
plt.show()

# Training the RF
# Relative
abs_rf_sex_gene = RandomForestClassifier(
    n_estimators=500,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

abs_rf_sex_gene.fit(abs_X_train_aug, abs_Y_train)

# Validation set
val_preds = abs_rf_sex_gene.predict(abs_X_val_aug)
print("Validation accuracy:", accuracy_score(abs_Y_val, val_preds))

# Test
y_pred = abs_rf_sex_gene.predict(abs_X_test_aug)
abs_rf_sex_gene_accuracy = accuracy_score(abs_Y_test, y_pred)
print("Test Accuracy:", abs_rf_sex_gene_accuracy)

cm = confusion_matrix(abs_Y_test, y_pred)
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
plt.title("Normalized Confusion Matrix (Row-wise) for Absolute Abundance RF Using Sex and Genes as Features")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("age_rel_rf_sex_gene_cm.png", format="png")
plt.show()

# Stratifying by gender
abs_train = abs_train[abs_train["sex"].isin(['male', 'female'])].copy()
abs_test = abs_test[abs_test["sex"].isin(['male', 'female'])].copy()
abs_val = abs_val[abs_val["sex"].isin(['male', 'female'])].copy()

rel_train = rel_train[rel_train["sex"].isin(['male', 'female'])].copy()
rel_test = rel_test[rel_test["sex"].isin(['male', 'female'])].copy()
rel_val = rel_val[rel_val["sex"].isin(['male', 'female'])].copy()

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
target_n = 300

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

# Filtering train/test/val datasets to only use these columns
# Absolute quant
abs_X_train = abs_balanced_train.loc[:, abs_balanced_train.columns.isin(col_names)].copy()
abs_Y_train = abs_balanced_train['age_bin']

abs_X_test = abs_test.loc[:, abs_test.columns.isin(col_names)].copy()
abs_Y_test  = abs_test['age_bin']

abs_X_val = abs_val.loc[:, abs_val.columns.isin(col_names)].copy()
abs_Y_val  = abs_val['age_bin']

# Relative abundance data
rel_X_train = rel_balanced_train.loc[:, rel_balanced_train.columns.isin(col_names)].copy()
rel_Y_train = rel_balanced_train['age_bin']

rel_X_test = rel_test.loc[:, rel_test.columns.isin(col_names)].copy()
rel_Y_test  = rel_test['age_bin']

rel_X_val = rel_val.loc[:, rel_val.columns.isin(col_names)].copy()
rel_Y_val  = rel_val['age_bin']

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
abs_rf_sex = RandomForestClassifier(
    n_estimators=500,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

abs_rf_sex.fit(abs_X_train_aug, abs_Y_train)

# Validation
val_preds = abs_rf_sex.predict(abs_X_test_aug)
print("Validation accuracy:", accuracy_score(abs_Y_test, val_preds))

# Test
y_pred = abs_rf_sex.predict(abs_X_test_aug)
abs_rf_sex_accuracy = accuracy_score(abs_Y_test, y_pred)
print("Test Accuracy:", abs_rf_sex_accuracy)

cm = confusion_matrix(abs_Y_test, y_pred)
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
plt.title("Normalized Confusion Matrix (Row-wise) for Absolute Abundance RF with Sex as a Feature")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("age_abs_rf_sex_cm.png", format="png")
plt.show()

# Relative
rel_rf_sex = RandomForestClassifier(
    n_estimators=500,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rel_rf_sex.fit(rel_X_train_aug, rel_Y_train)

# Validation
val_preds = rel_rf_sex.predict(rel_X_test_aug)
print("Validation accuracy:", accuracy_score(rel_Y_test, val_preds))

# Test
y_pred = rel_rf_sex.predict(rel_X_test_aug)
rel_rf_sex_accuracy = accuracy_score(rel_Y_test, y_pred)
print("Test Accuracy:", rel_rf_sex_accuracy)

cm = confusion_matrix(rel_Y_test, y_pred)
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
plt.title("Normalized Confusion Matrix (Row-wise) for Relative Abundance RF with Sex as a Feature")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("age_rel_rf_sex_cm.png", format="png")
plt.show()

# Final Comparison
print("TEST ACCURACY COMPARISON")

print(f"{'Model':<35}{'Absolute':>12}{'Relative':>12}")
print("-" * 60)

print(f"{'Balanced RF (grid)':<35}"
      f"{abs_bal_accuracy:>12.4f}"
      f"{rel_bal_accuracy:>12.4f}")

print(f"{'Balanced RF (no grid)':<35}"
      f"{abs_bal_nogrid_accuracy:>12.4f}"
      f"{rel_bal_nogrid_accuracy:>12.4f}")

print(f"{'RF + Sex':<35}"
      f"{abs_rf_sex_accuracy:>12.4f}"
      f"{rel_rf_sex_accuracy:>12.4f}")

print(f"{'RF + Sex + Gene':<35}"
      f"{abs_rf_sex_gene_accuracy:>12.4f}"
      f"{rel_rf_sex_gene_accuracy:>12.4f}")

print("\n=====================================================\n")

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
target_n = 300

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
abs_Y_train = abs_balanced_train['age_bin']

abs_X_test = abs_test[abs_test.columns[:1148]].drop(columns=['original_SampleID'])
abs_Y_test  = abs_test['age_bin']

abs_X_val = abs_val[abs_val.columns[:1148]].drop(columns=['original_SampleID'])
abs_Y_val  = abs_val['age_bin']

# Relative abundance data
rel_X_train = rel_balanced_train[rel_balanced_train.columns[:1148]].drop(columns=['original_SampleID'])
rel_Y_train = rel_balanced_train['age_bin']

rel_X_test = rel_test[rel_test.columns[:1148]].drop(columns=['original_SampleID'])
rel_Y_test  = rel_test['age_bin']

rel_X_val = rel_val[rel_val.columns[:1148]].drop(columns=['original_SampleID'])
rel_Y_val  = rel_val['age_bin']

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

# Without a grid search
abs_rf_nogrid = RandomForestClassifier(
    n_estimators=500,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

abs_rf_nogrid.fit(abs_X_train_log, abs_Y_train)

# Validation
val_preds = abs_rf_nogrid.predict(abs_X_test_log)
print("Validation accuracy:", accuracy_score(abs_Y_test, val_preds))

# Test
abs_pred = abs_best_rf.predict(abs_X_test_log)
abs_bal_nogrid_accuracy = accuracy_score(abs_Y_test, y_pred)
print("Test Accuracy:", abs_bal_nogrid_accuracy)

# Absolute
# Validation
val_preds = rel_best_rf.predict(rel_X_test_log)
print("Validation accuracy:", accuracy_score(rel_Y_test, val_preds))

# Test
rel_pred = rel_best_rf.predict(rel_X_test_log)
rel_bal_accuracy = accuracy_score(rel_Y_test, y_pred)
print("Test Accuracy:", rel_bal_accuracy)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Absolute
ConfusionMatrixDisplay.from_predictions(
    abs_Y_test,
    abs_pred,
    normalize='true',
    cmap='Blues',
    ax=axes[0]
)
axes[0].set_title("Absolute Abundance")

# Relative
ConfusionMatrixDisplay.from_predictions(
    rel_Y_test,
    rel_pred,
    normalize='true',
    cmap='Greens',
    ax=axes[1]
)
axes[1].set_title("Relative Abundance")

plt.suptitle("Confusion Matrices for RF Classifiers")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("balanced_rf_cm_comparison.png", dpi=300)
plt.show()

# Macro AUC
# Helper function
def multiclass_macro_roc(y_true, y_proba, classes):
    y_true_bin = label_binarize(y_true, classes=classes)  # shape (n, K)
    K = len(classes)

    fpr = {}
    tpr = {}
    auc_per_class = []

    for i in range(K):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        auc_per_class.append(
            roc_auc_score(y_true_bin[:, i], y_proba[:, i])
        )

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(K)]))
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(K):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= K
    auc_macro = float(np.mean(auc_per_class))
    return all_fpr, mean_tpr, auc_macro

# Probabilities
abs_probs = abs_rf_nogrid.predict_proba(abs_X_test_log)
rel_probs = rel_rf_nogrid.predict_proba(rel_X_test_log)

classes = abs_rf_nogrid.classes_

abs_macro_auc = roc_auc_score(abs_Y_test, abs_probs, multi_class="ovr", average="macro")
rel_macro_auc = roc_auc_score(rel_Y_test, rel_probs, multi_class="ovr", average="macro")

abs_fpr_macro, abs_tpr_macro, abs_macro_auc_curve = multiclass_macro_roc(abs_Y_test, abs_probs, classes)
rel_fpr_macro, rel_tpr_macro, rel_macro_auc_curve = multiclass_macro_roc(rel_Y_test, rel_probs, classes)

# Plot
plt.figure(figsize=(7, 6))
plt.plot(abs_fpr_macro, abs_tpr_macro, label=f"Absolute (Macro AUC = {abs_macro_auc:.4f})")
plt.plot(rel_fpr_macro, rel_tpr_macro, label=f"Relative (Macro AUC = {rel_macro_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Macro ROC Curve (OvR): Absolute vs Relative (Age Bins)")
plt.legend()
plt.tight_layout()
plt.savefig("rf_macro_auc_comparison.png", dpi=300)
plt.show()

# Checking if results are due to chance via bootstrapping
n_boot = 1000

acc_diffs = []
auc_diffs = []

abs_test_codes = abs_Y_test.astype("category").cat.codes
rel_test_codes = rel_Y_test.astype("category").cat.codes
classes = range(len(abs_Y_test.astype("category").cat.categories))

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
    acc_abs_i = accuracy_score(abs_Y_test.iloc[indices], y_abs_pred[indices])
    acc_rel_i = accuracy_score(rel_Y_test.iloc[indices], y_rel_pred[indices])
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
    print("AUC difference is NOT statistically significant!")

abs_feat_importance = pd.DataFrame({
    "feature": abs_X_train.columns,
    "importance": abs_rf_nogrid.feature_importances_
}).sort_values(by="importance", ascending=False)

rel_feat_importance = pd.DataFrame({
    "feature": rel_X_train.columns,
    "importance": rel_rf_nogrid.feature_importances_
}).sort_values(by="importance", ascending=False)

top_n = 20

plt.figure(figsize=(8,6))
plt.barh(
    abs_feat_importance["feature"][:top_n][::-1],
    abs_feat_importance["importance"][:top_n][::-1]
)
plt.title("Top 20 Absolute Abundance Features")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("abs_rf_nogrid_top20_features.png", format="png")
plt.show()

top_n = 20

plt.figure(figsize=(8,6))
plt.barh(
    rel_feat_importance["feature"][:top_n][::-1],
    rel_feat_importance["importance"][:top_n][::-1]
)
plt.title("Top 20 Relative Abundance Features")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("rel_rf_nogrid_top20_features.png", format="png")
plt.show()

# Top 20 feature names
abs_top20 = set(abs_feat_importance["feature"].head(20))
rel_top20 = set(rel_feat_importance["feature"].head(20))

overlap = abs_top20.intersection(rel_top20)
print("Overlap:", overlap)
print("Number overlapping:", len(overlap))
print("Unique to Absolute:", abs_top20 - rel_top20)
print("Unique to Relative:", rel_top20 - abs_top20)

# SHAP Analysis
# Absolute
X = abs_X_test_log.copy()
X = X[abs_rf_nogrid.feature_names_in_]

abs_explainer = shap.TreeExplainer(abs_rf_nogrid)
abs_shap = abs_explainer(X)

sv = abs_shap.values
if sv.ndim == 3:
    class_idx = 1
    sv = sv[:, :, class_idx]

plt.figure(figsize=(8, 6))
shap.summary_plot(
    sv, X,
    plot_type="bar",
    max_display=20,
    show=False
)

ax = plt.gca()
ax.set_title("SHAP Values - Absolute Abundance", fontsize=14, pad=12)
plt.tight_layout()
plt.savefig("abs_rf_nogrid_SHAP.png", format="png")
plt.show()

# SHAP Analysis
# Relative
X = rel_X_test_log.copy()
X = X[rel_rf_nogrid.feature_names_in_]

rel_explainer = shap.TreeExplainer(rel_rf_nogrid)
rel_shap = rel_explainer(X)

sv = rel_shap.values
if sv.ndim == 3:
    class_idx = 1
    sv = sv[:, :, class_idx]

plt.figure(figsize=(8, 6))
shap.summary_plot(
    sv, X,
    plot_type="bar",
    max_display=20,
    show=False
)

ax = plt.gca()
ax.set_title("SHAP Values - Relative Abundance", fontsize=14, pad=12)
plt.tight_layout()
plt.savefig("rel_rf_nogrid_SHAP.png", format="png")
plt.show()