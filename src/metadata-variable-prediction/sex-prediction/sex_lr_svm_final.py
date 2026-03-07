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
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error, mean_squared_error, r2_score, roc_curve, auc, roc_auc_score, f1_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR


RANDOM_STATE = 42
FEATURE_END = 1148
ID_COLUMN = 'original_SampleID'
TARGET_COLUMN = 'sex'
ALLOWED_SEXES = ['male', 'female']
TRAIN_BALANCE_N = 585
BALANCED_TEST_N = 210
BALANCED_VAL_N = 196
POS_CLASS = 'male'

DATA_PATHS = {
    'Absolute': {
        'train': '/ddn_scratch/k5zhao/data/classifier_training/abs_train.csv',
        'test': '/ddn_scratch/k5zhao/data/classifier_training/abs_test.csv',
        'val': '/ddn_scratch/k5zhao/data/classifier_training/abs_val.csv',
    },
    'Relative': {
        'train': '/ddn_scratch/k5zhao/data/classifier_training/rel_train.csv',
        'test': '/ddn_scratch/k5zhao/data/classifier_training/rel_test.csv',
        'val': '/ddn_scratch/k5zhao/data/classifier_training/rel_val.csv',
    }
}

SVM_PARAM_GRID = {
    'svc__C': [0.01, 0.1, 1, 10, 100],
    'svc__gamma': ['scale', 0.01, 0.1, 1],
    'svc__kernel': ['rbf']
}


def load_representation(paths):
    splits = {}
    for split_name, path in paths.items():
        df = pd.read_csv(path, low_memory=False)
        splits[split_name] = df[df[TARGET_COLUMN].isin(ALLOWED_SEXES)].copy()
    return splits


def balance_by_target(df, target_n, target_col=TARGET_COLUMN):
    return (
        df.groupby(target_col, group_keys=False)
          .apply(lambda x: x.sample(n=min(len(x), target_n), random_state=RANDOM_STATE))
          .reset_index(drop=True)
    )


def select_features_and_target(df):
    feature_columns = [col for col in df.columns[:FEATURE_END] if col != ID_COLUMN]
    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def log_transform_split(X):
    return np.log1p(X.copy())


def prepare_dataset(paths, balance_config=None):
    raw_splits = load_representation(paths)
    processed = {}

    for split_name, df in raw_splits.items():
        if balance_config is not None and split_name in balance_config:
            df = balance_by_target(df, balance_config[split_name])

        X, y = select_features_and_target(df)
        processed[split_name] = {
            'df': df,
            'X': X,
            'y': y,
            'X_log': log_transform_split(X)
        }

    return processed


def compute_metrics(y_true, y_pred, y_prob=None, use_balanced_accuracy=False):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro'),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted')
    }

    if use_balanced_accuracy:
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)

    if y_prob is not None:
        metrics['auc'] = roc_auc_score(y_true, y_prob)

    return metrics


def evaluate_classifier(model, X_val, y_val, X_test, y_test, use_balanced_accuracy=False):
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    val_prob = None
    test_prob = None
    if hasattr(model, 'predict_proba'):
        val_prob = model.predict_proba(X_val)[:, 1]
        test_prob = model.predict_proba(X_test)[:, 1]

    return {
        'val': compute_metrics(y_val, val_pred, val_prob, use_balanced_accuracy=use_balanced_accuracy),
        'test': compute_metrics(y_test, test_pred, test_prob, use_balanced_accuracy=use_balanced_accuracy),
        'val_pred': val_pred,
        'test_pred': test_pred,
        'val_prob': val_prob,
        'test_prob': test_prob,
        'classes': model.classes_
    }


def print_metric_block(name, results, use_balanced_accuracy=False):
    print(f'\n{name}')
    print('Validation accuracy:', results['val']['accuracy'])
    if use_balanced_accuracy:
        print('Validation Balanced Accuracy:', results['val']['balanced_accuracy'])
    if 'auc' in results['val']:
        print('Validation AUC:', results['val']['auc'])
    print('Validation Macro F1:', results['val']['macro_f1'])
    print('Validation Weighted F1:', results['val']['weighted_f1'])

    print('Test accuracy:', results['test']['accuracy'])
    if use_balanced_accuracy:
        print('Test Balanced Accuracy:', results['test']['balanced_accuracy'])
    if 'auc' in results['test']:
        print('Test AUC:', results['test']['auc'])
    print('Test Macro F1:', results['test']['macro_f1'])
    print('Test Weighted F1:', results['test']['weighted_f1'])


def plot_confusion_matrix_heatmap(y_true, y_pred, labels, title, filename, ax=None):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    if ax is None:
        plt.figure(figsize=(10, 8))
        ax = plt.gca()

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )
    ax.set_xlabel('Predicted sex')
    ax.set_ylabel('Actual Sex')
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels, rotation=0)
    ax.set_title(title)

    if filename is not None and ax is plt.gca():
        plt.tight_layout()
        plt.savefig(filename, format='png')
        plt.show()


def save_single_confusion_plot(y_true, y_pred, labels, title, filename):
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    plot_confusion_matrix_heatmap(y_true, y_pred, labels, title, None, ax=ax)
    plt.tight_layout()
    plt.savefig(filename, format='png')
    plt.show()


def save_comparison_confusion_plot(left_y_true, left_y_pred, left_labels, left_title,
                                   right_y_true, right_y_pred, right_labels, right_title,
                                   filename):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_confusion_matrix_heatmap(left_y_true, left_y_pred, left_labels, left_title, None, ax=axes[0])
    plot_confusion_matrix_heatmap(right_y_true, right_y_pred, right_labels, right_title, None, ax=axes[1])
    plt.tight_layout()
    plt.savefig(filename, format='png')
    plt.show()


def save_roc_comparison_plot(abs_y_true, abs_prob, abs_classes, abs_label,
                             rel_y_true, rel_prob, rel_classes, rel_label,
                             title, filename, pos_class=POS_CLASS):
    abs_pos_index = list(abs_classes).index(pos_class)
    rel_pos_index = list(rel_classes).index(pos_class)

    abs_scores = abs_prob if abs_prob.ndim == 1 else abs_prob[:, abs_pos_index]
    rel_scores = rel_prob if rel_prob.ndim == 1 else rel_prob[:, rel_pos_index]

    abs_fpr, abs_tpr, _ = roc_curve(abs_y_true, abs_scores, pos_label=pos_class)
    rel_fpr, rel_tpr, _ = roc_curve(rel_y_true, rel_scores, pos_label=pos_class)

    abs_auc = auc(abs_fpr, abs_tpr)
    rel_auc = auc(rel_fpr, rel_tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(abs_fpr, abs_tpr, label=f'{abs_label} (AUC = {abs_auc:.3f})')
    plt.plot(rel_fpr, rel_tpr, label=f'{rel_label} (AUC = {rel_auc:.3f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, format='png')
    plt.show()


def stratified_bootstrap_indices(y, rng):
    y = np.asarray(y)
    indices = []
    for c in np.unique(y):
        class_idx = np.where(y == c)[0]
        boot_idx = rng.choice(class_idx, size=len(class_idx), replace=True)
        indices.extend(boot_idx)
    return np.array(indices)


def bootstrap_metric_differences(abs_y_true, abs_y_pred, abs_y_prob,
                                 rel_y_true, rel_y_pred, rel_y_prob,
                                 n_boot=1000):
    rng = np.random.default_rng(RANDOM_STATE)
    acc_diffs = []
    auc_diffs = []

    abs_y_true = np.asarray(abs_y_true)
    rel_y_true = np.asarray(rel_y_true)
    abs_y_pred = np.asarray(abs_y_pred)
    rel_y_pred = np.asarray(rel_y_pred)
    abs_y_prob = np.asarray(abs_y_prob)
    rel_y_prob = np.asarray(rel_y_prob)

    for _ in range(n_boot):
        indices = stratified_bootstrap_indices(abs_y_true, rng)
        acc_abs = accuracy_score(abs_y_true[indices], abs_y_pred[indices])
        acc_rel = accuracy_score(rel_y_true[indices], rel_y_pred[indices])
        acc_diffs.append(acc_abs - acc_rel)

        auc_abs = roc_auc_score(abs_y_true[indices], abs_y_prob[indices])
        auc_rel = roc_auc_score(rel_y_true[indices], rel_y_prob[indices])
        auc_diffs.append(auc_abs - auc_rel)

    acc_ci = np.percentile(acc_diffs, [2.5, 97.5])
    auc_ci = np.percentile(auc_diffs, [2.5, 97.5])
    return acc_ci, auc_ci


def print_bootstrap_summary(acc_ci, auc_ci):
    print('Accuracy difference 95% CI (Abs - Rel):', acc_ci)
    print('Macro-AUC difference 95% CI (Abs - Rel):', auc_ci)

    if acc_ci[0] > 0 or acc_ci[1] < 0:
        print('Accuracy difference is statistically significant.')
    else:
        print('Accuracy difference is NOT statistically significant.')

    if auc_ci[0] > 0 or auc_ci[1] < 0:
        print('AUC difference is statistically significant.')
    else:
        print('AUC difference is NOT statistically significant.')


def plot_logreg_feature_importance(model, X_train, title, filename, top_n=20):
    feat_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': np.abs(model.coef_[0])
    }).sort_values(by='importance', ascending=False)

    plt.figure(figsize=(8, 6))
    plt.barh(
        feat_importance['feature'].head(top_n).iloc[::-1],
        feat_importance['importance'].head(top_n).iloc[::-1]
    )
    plt.title(title)
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(filename, format='png')
    plt.show()

    return feat_importance


def print_top_feature_overlap(abs_feat_importance, rel_feat_importance, top_n=20):
    abs_top = set(abs_feat_importance['feature'].head(top_n))
    rel_top = set(rel_feat_importance['feature'].head(top_n))
    overlap = abs_top.intersection(rel_top)

    print('Overlap:', overlap)
    print('Number overlapping:', len(overlap))
    print('Unique to Absolute:', abs_top - rel_top)
    print('Unique to Relative:', rel_top - abs_top)


def run_logreg_shap(model, X_test, title, filename):
    X = X_test.copy()
    X = X[model.feature_names_in_]

    explainer = shap.LinearExplainer(model, X, feature_perturbation='independent')
    shap_values = explainer(X)

    sv = shap_values.values
    if sv.ndim == 3:
        class_idx = list(model.classes_).index(POS_CLASS)
        sv = sv[:, :, class_idx]

    plt.figure(figsize=(8, 6))
    shap.summary_plot(sv, X, plot_type='bar', max_display=20, show=False)
    ax = plt.gca()
    ax.set_title(title, fontsize=14, pad=12)
    plt.tight_layout()
    plt.savefig(filename, format='png')
    plt.show()


def fit_logistic_regression(X_train, y_train):
    model = LogisticRegression(solver='saga', class_weight='balanced')
    return model.fit(X_train, y_train)


def fit_svm_grid(X_train, y_train):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(probability=True))
    ])

    grid = GridSearchCV(
        pipeline,
        SVM_PARAM_GRID,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )
    fitted_grid = grid.fit(X_train, y_train)
    print('Best params:', fitted_grid.best_params_)
    print('Best CV AUC:', fitted_grid.best_score_)
    return fitted_grid


def fit_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        class_weight='balanced',
        n_jobs=-1
    )
    return model.fit(X_train, y_train)


def run_representation_model(name, representation_name, dataset, model_type):
    X_train = dataset['train']['X_log']
    y_train = dataset['train']['y']
    X_val = dataset['val']['X_log']
    y_val = dataset['val']['y']
    X_test = dataset['test']['X_log']
    y_test = dataset['test']['y']

    if model_type == 'logreg':
        model = fit_logistic_regression(X_train, y_train)
        results = evaluate_classifier(model, X_val, y_val, X_test, y_test, use_balanced_accuracy=False)
    elif model_type == 'svm':
        grid = fit_svm_grid(X_train, y_train)
        model = grid.best_estimator_
        results = evaluate_classifier(model, X_val, y_val, X_test, y_test, use_balanced_accuracy=True)
        results['grid'] = grid
    elif model_type == 'rf':
        model = fit_random_forest(X_train, y_train)
        results = evaluate_classifier(model, X_val, y_val, X_test, y_test, use_balanced_accuracy=False)
    else:
        raise ValueError(f'Unsupported model_type: {model_type}')

    results['model'] = model
    results['representation'] = representation_name
    results['name'] = name
    return results


def create_results_table(abs_rf_results, abs_lr_results, abs_svm_results,
                         rel_rf_results, rel_lr_results, rel_svm_results):
    results_df = pd.DataFrame({
        'Task': ['Classification'] * 6,
        'Target': ['sex'] * 6,
        'Representation': ['Absolute', 'Absolute', 'Absolute', 'Relative', 'Relative', 'Relative'],
        'Model': ['RandomForest', 'Logistic Regression', 'SVM_RBF', 'RandomForest', 'Logistic Regression', 'SVM_RBF'],
        'Val_Accuracy': [
            abs_rf_results['val']['accuracy'], abs_lr_results['val']['accuracy'], abs_svm_results['val']['accuracy'],
            rel_rf_results['val']['accuracy'], rel_lr_results['val']['accuracy'], rel_svm_results['val']['accuracy']
        ],
        'Test_Accuracy': [
            abs_rf_results['test']['accuracy'], abs_lr_results['test']['accuracy'], abs_svm_results['test']['accuracy'],
            rel_rf_results['test']['accuracy'], rel_lr_results['test']['accuracy'], rel_svm_results['test']['accuracy']
        ],
        'Val_MacroF1': [
            abs_rf_results['val']['macro_f1'], abs_lr_results['val']['macro_f1'], abs_svm_results['val']['macro_f1'],
            rel_rf_results['val']['macro_f1'], rel_lr_results['val']['macro_f1'], rel_svm_results['val']['macro_f1']
        ],
        'Test_MacroF1': [
            abs_rf_results['test']['macro_f1'], abs_lr_results['test']['macro_f1'], abs_svm_results['test']['macro_f1'],
            rel_rf_results['test']['macro_f1'], rel_lr_results['test']['macro_f1'], rel_svm_results['test']['macro_f1']
        ],
        'Val_WeightedF1': [
            abs_rf_results['val']['weighted_f1'], abs_lr_results['val']['weighted_f1'], abs_svm_results['val']['weighted_f1'],
            rel_rf_results['val']['weighted_f1'], rel_lr_results['val']['weighted_f1'], rel_svm_results['val']['weighted_f1']
        ],
        'Test_WeightedF1': [
            abs_rf_results['test']['weighted_f1'], abs_lr_results['test']['weighted_f1'], abs_svm_results['test']['weighted_f1'],
            rel_rf_results['test']['weighted_f1'], rel_lr_results['test']['weighted_f1'], rel_svm_results['test']['weighted_f1']
        ]
    }).round(6)
    return results_df


def main():
    unbalanced_dataset = {
        representation: prepare_dataset(paths, balance_config={'train': TRAIN_BALANCE_N})
        for representation, paths in DATA_PATHS.items()
    }

    balanced_eval_dataset = {
        representation: prepare_dataset(
            paths,
            balance_config={'train': TRAIN_BALANCE_N, 'test': BALANCED_TEST_N, 'val': BALANCED_VAL_N}
        )
        for representation, paths in DATA_PATHS.items()
    }

    # Logistic Regression
    abs_lr_results = run_representation_model('Absolute Logistic Regression', 'Absolute', unbalanced_dataset['Absolute'], 'logreg')
    rel_lr_results = run_representation_model('Relative Logistic Regression', 'Relative', unbalanced_dataset['Relative'], 'logreg')

    print_metric_block('Absolute Logistic Regression', abs_lr_results)
    print_metric_block('Relative Logistic Regression', rel_lr_results)

    save_single_confusion_plot(
        unbalanced_dataset['Absolute']['test']['y'],
        abs_lr_results['test_pred'],
        abs_lr_results['classes'],
        'Normalized Confusion Matrix for Absolute Logistic Regression',
        'sex_abs_lr_cm.png'
    )
    save_single_confusion_plot(
        unbalanced_dataset['Relative']['test']['y'],
        rel_lr_results['test_pred'],
        rel_lr_results['classes'],
        'Normalized Confusion Matrix for Relative Abundance Logistic Regression',
        'sex_rel_lr_cm.png'
    )
    save_comparison_confusion_plot(
        unbalanced_dataset['Absolute']['test']['y'], abs_lr_results['test_pred'], abs_lr_results['classes'],
        'Normalized Confusion Matrix for Absolute LR Classifier',
        unbalanced_dataset['Relative']['test']['y'], rel_lr_results['test_pred'], rel_lr_results['classes'],
        'Normalized Confusion Matrix for Relative LR Classifier',
        'sex_lr_cm_comparison.png'
    )

    print(classification_report(unbalanced_dataset['Absolute']['test']['y'], abs_lr_results['test_pred']))
    print(classification_report(unbalanced_dataset['Relative']['test']['y'], rel_lr_results['test_pred']))

    save_roc_comparison_plot(
        unbalanced_dataset['Absolute']['test']['y'],
        abs_lr_results['test_prob'],
        abs_lr_results['classes'],
        'Absolute Abundance LR',
        unbalanced_dataset['Relative']['test']['y'],
        rel_lr_results['test_prob'],
        rel_lr_results['classes'],
        'Relative Abundance LR',
        f'ROC Curve Comparison (Positive Class = {POS_CLASS})',
        'sex_lr_roc_comparison.png'
    )

    lr_acc_ci, lr_auc_ci = bootstrap_metric_differences(
        unbalanced_dataset['Absolute']['test']['y'], abs_lr_results['test_pred'], abs_lr_results['test_prob'],
        unbalanced_dataset['Relative']['test']['y'], rel_lr_results['test_pred'], rel_lr_results['test_prob']
    )
    print_bootstrap_summary(lr_acc_ci, lr_auc_ci)

    abs_feat_importance = plot_logreg_feature_importance(
        abs_lr_results['model'], unbalanced_dataset['Absolute']['train']['X_log'],
        'Top 20 Absolute Abundance Features for Logistic Regression',
        'sex_abs_lr_top20_features.png'
    )
    rel_feat_importance = plot_logreg_feature_importance(
        rel_lr_results['model'], unbalanced_dataset['Relative']['train']['X_log'],
        'Top 20 Relative Abundance Features for Logistic Regression',
        'sex_rel_lr_top20_features.png'
    )
    print_top_feature_overlap(abs_feat_importance, rel_feat_importance)

    run_logreg_shap(
        abs_lr_results['model'], unbalanced_dataset['Absolute']['test']['X_log'],
        'SHAP Values for Logistic Regression - Absolute Abundance',
        'sex_abs_lr_shap.png'
    )
    run_logreg_shap(
        rel_lr_results['model'], unbalanced_dataset['Relative']['test']['X_log'],
        'SHAP Values for Logistic Regression - Relative Abundance',
        'sex_rel_lr_shap.png'
    )

    # SVM
    abs_svm_results = run_representation_model('Absolute SVM', 'Absolute', unbalanced_dataset['Absolute'], 'svm')
    rel_svm_results = run_representation_model('Relative SVM', 'Relative', unbalanced_dataset['Relative'], 'svm')

    print_metric_block('Absolute SVM', abs_svm_results, use_balanced_accuracy=True)
    print_metric_block('Relative SVM', rel_svm_results, use_balanced_accuracy=True)

    save_single_confusion_plot(
        unbalanced_dataset['Absolute']['test']['y'],
        abs_svm_results['test_pred'],
        abs_svm_results['classes'],
        'Normalized Confusion Matrix for Absolute Abundance SVM',
        'sex_abs_svm_cm.png'
    )
    save_single_confusion_plot(
        unbalanced_dataset['Relative']['test']['y'],
        rel_svm_results['test_pred'],
        rel_svm_results['classes'],
        'Normalized Confusion Matrix for Relative Abundance SVM',
        'sex_rel_svm_cm.png'
    )
    save_comparison_confusion_plot(
        unbalanced_dataset['Absolute']['test']['y'], abs_svm_results['test_pred'], abs_svm_results['classes'],
        'Normalized Confusion Matrix for Absolute Abundance SVM Classifier',
        unbalanced_dataset['Relative']['test']['y'], rel_svm_results['test_pred'], rel_svm_results['classes'],
        'Normalized Confusion Matrix for Relative Abundance SVM Classifier',
        'sex_svm_cm_comparison.png'
    )

    print(classification_report(unbalanced_dataset['Absolute']['test']['y'], abs_svm_results['test_pred']))
    print(classification_report(unbalanced_dataset['Relative']['test']['y'], rel_svm_results['test_pred']))

    save_roc_comparison_plot(
        unbalanced_dataset['Absolute']['test']['y'],
        abs_svm_results['test_prob'],
        abs_svm_results['classes'],
        'Absolute Abundance SVM',
        unbalanced_dataset['Relative']['test']['y'],
        rel_svm_results['test_prob'],
        rel_svm_results['classes'],
        'Relative Abundance SVM',
        f'ROC Curve Comparison (Positive Class = {POS_CLASS})',
        'sex_svm_roc_comparison.png'
    )

    svm_acc_ci, svm_auc_ci = bootstrap_metric_differences(
        unbalanced_dataset['Absolute']['test']['y'], abs_svm_results['test_pred'], abs_svm_results['test_prob'],
        unbalanced_dataset['Relative']['test']['y'], rel_svm_results['test_pred'], rel_svm_results['test_prob']
    )
    print_bootstrap_summary(svm_acc_ci, svm_auc_ci)

    # Balanced validation/test checks for LR and SVM using same fitted models
    abs_bal_lr_val_pred = abs_lr_results['model'].predict(balanced_eval_dataset['Absolute']['val']['X_log'])
    print('val accuracy:', accuracy_score(balanced_eval_dataset['Absolute']['val']['y'], abs_bal_lr_val_pred))
    abs_bal_svm_val_pred = abs_svm_results['model'].predict(balanced_eval_dataset['Absolute']['val']['X_log'])
    print('val accuracy:', accuracy_score(balanced_eval_dataset['Absolute']['val']['y'], abs_bal_svm_val_pred))
    rel_bal_lr_val_pred = rel_lr_results['model'].predict(balanced_eval_dataset['Relative']['val']['X_log'])
    print('val accuracy:', accuracy_score(balanced_eval_dataset['Relative']['val']['y'], rel_bal_lr_val_pred))
    rel_bal_svm_val_pred = rel_svm_results['model'].predict(balanced_eval_dataset['Relative']['val']['X_log'])
    print('val accuracy:', accuracy_score(balanced_eval_dataset['Relative']['val']['y'], rel_bal_svm_val_pred))

    # Random Forest
    abs_rf_results = run_representation_model('Absolute RandomForest', 'Absolute', unbalanced_dataset['Absolute'], 'rf')
    rel_rf_results = run_representation_model('Relative RandomForest', 'Relative', unbalanced_dataset['Relative'], 'rf')

    print_metric_block('Absolute RandomForest', abs_rf_results)
    print_metric_block('Relative RandomForest', rel_rf_results)

    # Summary table
    results_df = create_results_table(
        abs_rf_results, abs_lr_results, abs_svm_results,
        rel_rf_results, rel_lr_results, rel_svm_results
    )
    print(results_df)
    results_df.to_csv('model_comparisons.csv', index=False)


if __name__ == '__main__':
    main()
