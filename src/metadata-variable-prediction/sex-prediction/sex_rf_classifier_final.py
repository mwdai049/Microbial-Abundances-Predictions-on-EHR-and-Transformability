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
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error, mean_squared_error, r2_score, roc_curve, auc, roc_auc_score, f1_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR


DATA_DIR = '/ddn_scratch/k5zhao/data/classifier_training'
TRAIN_TARGET_N = 585
TEST_TARGET_N = 210
VAL_TARGET_N = 196
FEATURE_SLICE = 1148
DROP_COL = 'original_SampleID'
TARGET_COL = 'sex'
AGE_COL = 'age'
POS_CLASS = 'male'
RANDOM_STATE = 42
N_ESTIMATORS = 300
ZERO_THRESHOLD = 0.40


# -----------------------------------------------------------------------------
# Data utilities
# -----------------------------------------------------------------------------
def load_split_frames(prefix):
    return {
        'train': pd.read_csv(f'{DATA_DIR}/{prefix}_train.csv', low_memory=False),
        'test': pd.read_csv(f'{DATA_DIR}/{prefix}_test.csv', low_memory=False),
        'val': pd.read_csv(f'{DATA_DIR}/{prefix}_val.csv', low_memory=False),
    }


def filter_binary_sex(df):
    return df[df[TARGET_COL].isin(['male', 'female'])].copy()


def load_dataset(prefix):
    splits = load_split_frames(prefix)
    for split_name in splits:
        splits[split_name] = filter_binary_sex(splits[split_name])
    return splits


def get_base_feature_columns(df):
    cols = list(df.columns[:FEATURE_SLICE])
    return [col for col in cols if col != DROP_COL]


def get_keep_cols_from_abs_train(abs_train, threshold=ZERO_THRESHOLD):
    feature_cols = abs_train.columns[:FEATURE_SLICE]
    frac_zeros = (abs_train[feature_cols] == 0).mean(axis=0)
    keep_cols = list(frac_zeros[frac_zeros <= threshold].index)
    return [col for col in keep_cols if col != DROP_COL]


def balance_by_class(df, target_n):
    return (
        df.groupby(TARGET_COL, group_keys=False)
          .apply(lambda x: x.sample(n=min(len(x), target_n), random_state=RANDOM_STATE))
          .copy()
    )


def maybe_balance_splits(splits, balance_plan=None):
    prepared = {}
    for split_name, df in splits.items():
        if balance_plan is not None and split_name in balance_plan:
            prepared[split_name] = balance_by_class(df, balance_plan[split_name])
        else:
            prepared[split_name] = df.copy()
    return prepared


def build_xy(df, feature_cols, include_age):
    X = df[feature_cols].copy()
    if include_age:
        X = X.assign(age=df[AGE_COL].values)
    y = df[TARGET_COL].copy()
    return X, y


def build_feature_matrices(splits, feature_cols, include_age):
    matrices = {}
    for split_name, df in splits.items():
        X, y = build_xy(df, feature_cols, include_age)
        matrices[split_name] = {
            'X': np.log1p(X),
            'y': y,
        }
    return matrices


# -----------------------------------------------------------------------------
# Model utilities
# -----------------------------------------------------------------------------
def make_rf_model(use_class_weight):
    kwargs = {
        'n_estimators': N_ESTIMATORS,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
    }
    if use_class_weight:
        kwargs['class_weight'] = 'balanced'
    return RandomForestClassifier(**kwargs)


def train_and_evaluate(train_X, train_y, val_X, val_y, test_X, test_y, use_class_weight):
    model = make_rf_model(use_class_weight=use_class_weight)
    model.fit(train_X, train_y)

    val_preds = model.predict(val_X)
    test_preds = model.predict(test_X)

    val_accuracy = accuracy_score(val_y, val_preds)
    test_accuracy = accuracy_score(test_y, test_preds)

    print('Validation accuracy:', val_accuracy)
    print('Test accuracy:', test_accuracy)

    return {
        'model': model,
        'val_preds': val_preds,
        'test_preds': test_preds,
        'val_accuracy': val_accuracy,
        'test_accuracy': test_accuracy,
    }


def plot_normalized_confusion_matrix(y_true, y_pred, labels, title, filename, ax=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums

    own_figure = ax is None
    if own_figure:
        plt.figure(figsize=(10, 8))
        ax = plt.gca()

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel('Predicted sex')
    ax.set_ylabel('Actual Sex')
    ax.set_title(title)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels, rotation=0)

    if own_figure:
        plt.tight_layout()
        plt.savefig(filename, format='png')
        plt.show()


def run_experiment(dataset_name, matrices, include_age, use_class_weight, title_stub, filename_stub):
    results = train_and_evaluate(
        matrices['train']['X'], matrices['train']['y'],
        matrices['val']['X'], matrices['val']['y'],
        matrices['test']['X'], matrices['test']['y'],
        use_class_weight=use_class_weight,
    )

    plot_normalized_confusion_matrix(
        matrices['test']['y'],
        results['test_preds'],
        results['model'].classes_,
        title_stub,
        filename_stub,
    )

    return results


# -----------------------------------------------------------------------------
# Analysis utilities
# -----------------------------------------------------------------------------
def print_accuracy_table(results_map):
    print('TEST ACCURACY COMPARISON')
    print(f"{'Model':<35}{'Absolute':>12}{'Relative':>12}")
    print('-' * 60)
    for label in [
        'RF (unbalanced)',
        'RF (balanced)',
        'RF (balanced, no age)',
        'RF (balanced, no age/missing taxa)',
    ]:
        print(
            f"{label:<35}"
            f"{results_map['abs'][label]['test_accuracy']:>12.4f}"
            f"{results_map['rel'][label]['test_accuracy']:>12.4f}"
        )
    print('\n=====================================================\n')


def print_balanced_validation_checks(models, balanced_eval_sets):
    print('BALANCED VALIDATION CHECKS')
    for dataset_name in ['abs', 'rel']:
        X_val = balanced_eval_sets[dataset_name]['with_age']['val']['X']
        y_val = balanced_eval_sets[dataset_name]['with_age']['val']['y']
        preds = models[dataset_name]['RF (unbalanced)']['model'].predict(X_val)
        print(f'{dataset_name} RF (unbalanced) val accuracy:', accuracy_score(y_val, preds))

        preds = models[dataset_name]['RF (balanced)']['model'].predict(X_val)
        print(f'{dataset_name} RF (balanced) val accuracy:', accuracy_score(y_val, preds))

        X_val_ageless = balanced_eval_sets[dataset_name]['no_age']['val']['X']
        y_val_ageless = balanced_eval_sets[dataset_name]['no_age']['val']['y']
        preds = models[dataset_name]['RF (balanced, no age)']['model'].predict(X_val_ageless)
        print(f'{dataset_name} RF (balanced, no age) val accuracy:', accuracy_score(y_val_ageless, preds))


def plot_best_model_confusion_comparison(abs_result, rel_result, abs_test, rel_test):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    plot_normalized_confusion_matrix(
        abs_test['y'],
        abs_result['test_preds'],
        abs_result['model'].classes_,
        'Normalized Confusion Matrix for Absolute RF Classifier',
        filename=None,
        ax=axes[0],
    )
    plot_normalized_confusion_matrix(
        rel_test['y'],
        rel_result['test_preds'],
        rel_result['model'].classes_,
        'Normalized Confusion Matrix for Relative RF Classifier',
        filename=None,
        ax=axes[1],
    )

    plt.tight_layout()
    plt.savefig('sex_rf_cm_comparison.png', format='png')
    plt.show()


def plot_roc_comparison(abs_result, rel_result, abs_test, rel_test):
    abs_pos_index = list(abs_result['model'].classes_).index(POS_CLASS)
    rel_pos_index = list(rel_result['model'].classes_).index(POS_CLASS)

    abs_probs = abs_result['model'].predict_proba(abs_test['X'])[:, abs_pos_index]
    rel_probs = rel_result['model'].predict_proba(rel_test['X'])[:, rel_pos_index]

    abs_fpr, abs_tpr, _ = roc_curve(abs_test['y'], abs_probs, pos_label=POS_CLASS)
    rel_fpr, rel_tpr, _ = roc_curve(rel_test['y'], rel_probs, pos_label=POS_CLASS)

    abs_auc = auc(abs_fpr, abs_tpr)
    rel_auc = auc(rel_fpr, rel_tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(abs_fpr, abs_tpr, label=f'Absolute RF (AUC = {abs_auc:.3f})')
    plt.plot(rel_fpr, rel_tpr, label=f'Relative RF (AUC = {rel_auc:.3f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve Comparison (Positive Class = {POS_CLASS})')
    plt.legend()
    plt.tight_layout()
    plt.savefig('sex_rf_roc_comparison.png', format='png')
    plt.show()

    return {
        'abs_probs': abs_probs,
        'rel_probs': rel_probs,
        'abs_auc': abs_auc,
        'rel_auc': rel_auc,
    }


def stratified_bootstrap_indices(y, rng):
    indices = []
    for cls in np.unique(y):
        class_idx = np.where(y == cls)[0]
        boot_idx = rng.choice(class_idx, size=len(class_idx), replace=True)
        indices.extend(boot_idx)
    return np.array(indices)


def bootstrap_model_comparison(abs_result, rel_result, abs_test, rel_test, n_boot=1000):
    rng = np.random.default_rng(RANDOM_STATE)

    y_abs_true = abs_test['y'].to_numpy()
    y_rel_true = rel_test['y'].to_numpy()
    y_abs_pred = abs_result['test_preds']
    y_rel_pred = rel_result['test_preds']

    abs_pos_index = list(abs_result['model'].classes_).index(POS_CLASS)
    rel_pos_index = list(rel_result['model'].classes_).index(POS_CLASS)
    abs_y_prob = abs_result['model'].predict_proba(abs_test['X'])[:, abs_pos_index]
    rel_y_prob = rel_result['model'].predict_proba(rel_test['X'])[:, rel_pos_index]

    acc_diffs = []
    auc_diffs = []

    for _ in range(n_boot):
        indices = stratified_bootstrap_indices(y_abs_true, rng)
        acc_abs = accuracy_score(y_abs_true[indices], y_abs_pred[indices])
        acc_rel = accuracy_score(y_rel_true[indices], y_rel_pred[indices])
        acc_diffs.append(acc_abs - acc_rel)

        auc_abs = roc_auc_score(y_abs_true[indices], abs_y_prob[indices])
        auc_rel = roc_auc_score(y_rel_true[indices], rel_y_prob[indices])
        auc_diffs.append(auc_abs - auc_rel)

    acc_ci = np.percentile(acc_diffs, [2.5, 97.5])
    auc_ci = np.percentile(auc_diffs, [2.5, 97.5])

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


def plot_feature_importance(model, train_X, title, filename, top_n=20):
    feat_importance = pd.DataFrame({
        'feature': train_X.columns,
        'importance': model.feature_importances_,
    }).sort_values(by='importance', ascending=False)

    plt.figure(figsize=(8, 6))
    plt.barh(
        feat_importance['feature'].head(top_n)[::-1],
        feat_importance['importance'].head(top_n)[::-1],
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


def plot_shap_summary(model, X_test, title, filename):
    X = X_test.copy()
    X = X[model.feature_names_in_]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    sv = shap_values.values
    if sv.ndim == 3:
        class_idx = list(model.classes_).index(POS_CLASS)
        sv = sv[:, :, class_idx]

    plt.figure(figsize=(8, 6))
    shap.summary_plot(
        sv,
        X,
        plot_type='bar',
        max_display=20,
        show=False,
    )
    ax = plt.gca()
    ax.set_title(title, fontsize=14, pad=12)
    plt.tight_layout()
    plt.savefig(filename, format='png')
    plt.show()


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def build_all_experiments(abs_splits, rel_splits):
    abs_base_cols = get_base_feature_columns(abs_splits['train'])
    rel_base_cols = get_base_feature_columns(rel_splits['train'])
    keep_cols = get_keep_cols_from_abs_train(abs_splits['train'])

    train_balanced_abs = maybe_balance_splits(abs_splits, {'train': TRAIN_TARGET_N})
    train_balanced_rel = maybe_balance_splits(rel_splits, {'train': TRAIN_TARGET_N})

    experiments = {
        'RF (unbalanced)': {
            'abs': build_feature_matrices(abs_splits, abs_base_cols, include_age=True),
            'rel': build_feature_matrices(rel_splits, rel_base_cols, include_age=True),
            'use_class_weight': False,
            'title_templates': {
                'abs': 'Normalized Confusion Matrix for Absolute RF Classifier with Age as a Feature',
                'rel': 'Normalized Confusion Matrix for Relative RF Classifier with Age as a Feature',
            },
            'file_templates': {
                'abs': 'unbalanced_sex_abs_rf_cm.png',
                'rel': 'unbalanced_sex_rel_rf_cm.png',
            },
        },
        'RF (balanced)': {
            'abs': build_feature_matrices(train_balanced_abs, abs_base_cols, include_age=True),
            'rel': build_feature_matrices(train_balanced_rel, rel_base_cols, include_age=True),
            'use_class_weight': True,
            'title_templates': {
                'abs': 'Normalized Confusion Matrix for Balanced Absolute RF Classifier with Age as a Feature',
                'rel': 'Normalized Confusion Matrix for Balanced Relative RF Classifier with Age as a Feature',
            },
            'file_templates': {
                'abs': 'balanced_sex_abs_rf_cm.png',
                'rel': 'balanced_sex_rel_rf_cm.png',
            },
        },
        'RF (balanced, no age)': {
            'abs': build_feature_matrices(train_balanced_abs, abs_base_cols, include_age=False),
            'rel': build_feature_matrices(train_balanced_rel, rel_base_cols, include_age=False),
            'use_class_weight': True,
            'title_templates': {
                'abs': 'Normalized Confusion Matrix for Balanced Absolute RF Classifier without Age as a Feature',
                'rel': 'Normalized Confusion Matrix for Balanced Relative RF Classifier without Age as a Feature',
            },
            'file_templates': {
                'abs': 'balanced_sex_abs_rf_ageless_cm.png',
                'rel': 'balanced_sex_rel_rf_ageless_cm.png',
            },
        },
        'RF (balanced, no age/missing taxa)': {
            'abs': build_feature_matrices(train_balanced_abs, keep_cols, include_age=False),
            'rel': build_feature_matrices(train_balanced_rel, keep_cols, include_age=False),
            'use_class_weight': True,
            'title_templates': {
                'abs': 'Normalized Confusion Matrix for Balanced Absolute RF Classifier without Age and > 40% Missing Columns',
                'rel': 'Normalized Confusion Matrix for Balanced Relative RF Classifier without Age and > 40% Missing Columns',
            },
            'file_templates': {
                'abs': 'balanced_sex_abs_rf_ageless_miss_cm.png',
                'rel': 'balanced_sex_rel_rf_ageless_miss_cm.png',
            },
        },
    }

    return experiments, keep_cols, abs_base_cols, rel_base_cols, train_balanced_abs, train_balanced_rel


def build_balanced_eval_sets(abs_splits, rel_splits, abs_base_cols, rel_base_cols):
    abs_balanced = maybe_balance_splits(
        abs_splits,
        {'train': TRAIN_TARGET_N, 'test': TEST_TARGET_N, 'val': VAL_TARGET_N},
    )
    rel_balanced = maybe_balance_splits(
        rel_splits,
        {'train': TRAIN_TARGET_N, 'test': TEST_TARGET_N, 'val': VAL_TARGET_N},
    )

    return {
        'abs': {
            'with_age': build_feature_matrices(abs_balanced, abs_base_cols, include_age=True),
            'no_age': build_feature_matrices(abs_balanced, abs_base_cols, include_age=False),
        },
        'rel': {
            'with_age': build_feature_matrices(rel_balanced, rel_base_cols, include_age=True),
            'no_age': build_feature_matrices(rel_balanced, rel_base_cols, include_age=False),
        },
    }


def main():
    abs_splits = load_dataset('abs')
    rel_splits = load_dataset('rel')

    experiments, keep_cols, abs_base_cols, rel_base_cols, train_balanced_abs, train_balanced_rel = build_all_experiments(abs_splits, rel_splits)

    results_map = {'abs': {}, 'rel': {}}
    for label, config in experiments.items():
        print(f'\n--- {label}: Absolute ---')
        results_map['abs'][label] = run_experiment(
            dataset_name='abs',
            matrices=config['abs'],
            include_age=('no age' not in label),
            use_class_weight=config['use_class_weight'],
            title_stub=config['title_templates']['abs'],
            filename_stub=config['file_templates']['abs'],
        )

        print(f'\n--- {label}: Relative ---')
        results_map['rel'][label] = run_experiment(
            dataset_name='rel',
            matrices=config['rel'],
            include_age=('no age' not in label),
            use_class_weight=config['use_class_weight'],
            title_stub=config['title_templates']['rel'],
            filename_stub=config['file_templates']['rel'],
        )

    balanced_eval_sets = build_balanced_eval_sets(abs_splits, rel_splits, abs_base_cols, rel_base_cols)
    print_balanced_validation_checks(results_map, balanced_eval_sets)
    print_accuracy_table(results_map)

    best_abs = results_map['abs']['RF (balanced, no age)']
    best_rel = results_map['rel']['RF (balanced, no age)']

    best_abs_test = experiments['RF (balanced, no age)']['abs']['test']
    best_rel_test = experiments['RF (balanced, no age)']['rel']['test']

    plot_best_model_confusion_comparison(best_abs, best_rel, best_abs_test, best_rel_test)

    print(classification_report(best_abs_test['y'], best_abs['test_preds']))
    print(classification_report(best_rel_test['y'], best_rel['test_preds']))

    plot_roc_comparison(best_abs, best_rel, best_abs_test, best_rel_test)
    bootstrap_model_comparison(best_abs, best_rel, best_abs_test, best_rel_test, n_boot=1000)

    abs_feat_importance = plot_feature_importance(
        best_abs['model'],
        experiments['RF (balanced, no age)']['abs']['train']['X'],
        'Top 20 Absolute Abundance Features',
        'sex_abs_rf_top20_features.png',
        top_n=20,
    )
    rel_feat_importance = plot_feature_importance(
        best_rel['model'],
        experiments['RF (balanced, no age)']['rel']['train']['X'],
        'Top 20 Relative Abundance Features',
        'sex_rel_rf_top20_features.png',
        top_n=20,
    )
    print_top_feature_overlap(abs_feat_importance, rel_feat_importance, top_n=20)

    plot_shap_summary(
        best_abs['model'],
        best_abs_test['X'],
        'SHAP Values - Absolute Abundance',
        'sex_abs_rf_shap.png',
    )
    plot_shap_summary(
        best_rel['model'],
        best_rel_test['X'],
        'SHAP Values - Relative Abundance',
        'sex_rel_rf_shap.png',
    )


if __name__ == '__main__':
    main()
