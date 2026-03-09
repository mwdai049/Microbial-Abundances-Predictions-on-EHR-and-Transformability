import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skbio import DistanceMatrix
import seaborn as sns
from scipy import stats

from scipy.spatial import procrustes
from skbio.stats.distance import mantel
from skbio.stats.ordination import pcoa
from scipy.spatial.distance import pdist, squareform

synth_train_df = pd.read_csv('/ddn_scratch/mwdai/capstone/data/synthetic_train.tsv', sep='\t', index_col='original_SampleID')
synth_val_df = pd.read_csv('/ddn_scratch/mwdai/capstone/data/synthetic_val.tsv', sep='\t', index_col='original_SampleID')
synth_test_df = pd.read_csv('/ddn_scratch/mwdai/capstone/data/synthetic_test.tsv', sep='\t', index_col='original_SampleID')

abs_train_df = pd.read_csv('/ddn_scratch/k5zhao/data/classifier_training/abs_train.csv', index_col='original_SampleID')
abs_val_df = pd.read_csv('/ddn_scratch/k5zhao/data/classifier_training/abs_val.csv', index_col='original_SampleID')
abs_test_df = pd.read_csv('/ddn_scratch/k5zhao/data/classifier_training/abs_test.csv', index_col='original_SampleID')

abs_train_df = abs_train_df.iloc[:, :synth_train_df.shape[1]]
abs_val_df = abs_val_df.iloc[:, :synth_val_df.shape[1]]
abs_test_df = abs_test_df.iloc[:, :synth_test_df.shape[1]]

def bray_curtis_distance(df: pd.DataFrame) -> DistanceMatrix:
    X = df.to_numpy(dtype=float)
    D = squareform(pdist(X, metric="braycurtis"))
    return DistanceMatrix(D, ids=df.index.astype(str).tolist())

def procrustes_plot(dm_t: DistanceMatrix,
                    dm_p: DistanceMatrix,
                    split,
                    ax,
                    n_components = 2):
    '''
    PCoA on both DMs → take first n_components → Procrustes → plot.
    '''
    # PCoA
    ord_t = pcoa(dm_t)
    ord_p = pcoa(dm_p)

    Xt = ord_t.samples.iloc[:, :n_components].to_numpy()
    Xp = ord_p.samples.iloc[:, :n_components].to_numpy()

    # procrustes
    mtx1, mtx2, disparity = procrustes(Xt, Xp)

    ax.scatter(mtx1[:, 0], mtx1[:, 1], label="True", alpha=0.6, color='#b2bf64')
    ax.scatter(mtx2[:, 0], mtx2[:, 1], label="Pred", alpha=0.55, color='#e6725f')

    # connect matched points
    for i in range(dm_t.shape[0]):
        ax.plot([mtx1[i, 0], mtx2[i, 0]], [mtx1[i, 1], mtx2[i, 1]], alpha=0.4, linewidth=0.5, color='gray')

    ax.set_xlabel("Dimension 1", fontsize=18)
    ax.set_ylabel("Dimension 2", fontsize=18)
    ax.set_title(f"{split} Split\nDisparity = {disparity:.4f}", fontsize=18)
    
    return mtx1, mtx2, disparity

# pcoa procrustes plot

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

fig.suptitle('PCoA → Procrustes on Bray-Curtis Dissimilarity Matrices (True vs Pred Absolute Abundance)', fontsize=22, fontweight='bold')

true_df = abs_test_df.copy()
pred_df = synth_test_df.copy()

splits = ['Train', 'Val', 'Test']

true_dfs = [
    abs_train_df,
    abs_val_df,
    abs_test_df
]

pred_dfs = [
    synth_train_df,
    synth_val_df,
    synth_test_df
]

residuals = []
mtx1s = []

for ax, true_df, pred_df, split in zip(axes, true_dfs, pred_dfs, splits):
    common_samples = true_df.index.intersection(pred_df.index)
    common_feats = true_df.columns.intersection(pred_df.columns)

    true_aligned = true_df.loc[common_samples, common_feats]
    pred_aligned = pred_df.loc[common_samples, common_feats]

    dm_true = bray_curtis_distance(true_aligned)
    dm_pred = bray_curtis_distance(pred_aligned)

    mtx1, mtx2, disparity = procrustes_plot(dm_true, dm_pred, split, ax)
    
    residuals.append(np.linalg.norm(mtx1 - mtx2, axis=1))
    mtx1s.append(mtx1)
    
axes[2].legend(
    loc='upper left',
    bbox_to_anchor=(0.98, 0.98),
    frameon=False,
    fontsize=18,
    markerscale=2.0
)
axes[1].set_ylabel('')
axes[2].set_ylabel('')
                    
plt.tight_layout()

plt.savefig('./figs/pcoa_procrustes.png', dpi=500, bbox_inches='tight')


# mantel test
mantel_stat, p, n = mantel(dm_true, dm_pred, method='spearman')

print(f'Mantel r: {mantel_stat}')

# procrustes residuals
vmin = min(r.min() for r in residuals)
vmax = max(r.max() for r in residuals)

fig, axes = plt.subplots(1, 3, figsize=(24, 6))

for i in range(3):
    sc = axes[i].scatter(
        mtx1s[i][:,0],
        mtx1s[i][:,1],
        c=residuals[i],
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
        s=80
    )
    axes[i].set_title(f'{splits[i]} Split')
    if i == 0:
        axes[i].set_ylabel('Dimension 2')
    axes[i].set_xlabel('Dimension 1')

cbar = fig.colorbar(sc, ax=axes, label="Procrustes residual")

plt.suptitle("Ordination with Procrustes Residuals By Split", fontsize=22, fontweight='bold')

plt.savefig('./figs/procrustes_residuals.png', dpi=500, bbox_inches='tight')


# procrustes residuals on train split only
i = 0

sc = plt.scatter(
        mtx1s[i][:,0],
        mtx1s[i][:,1],
        c=residuals[i],
        cmap="viridis",
        s=80
    )
if i == 0:
    plt.ylabel('Dimension 2')
plt.xlabel('Dimension 1')

cbar = fig.colorbar(sc, label="Procrustes residual")

plt.suptitle("Ordination with Procrustes Residuals for Train Only", fontsize=15, fontweight='bold')

plt.savefig('./figs/procrustes_residuals_train.png', dpi=500, bbox_inches='tight')

# examine residuals against load and metadata
true_load = abs_test_df.sum(axis=1)
pred_load = synth_test_df.sum(axis=1)

load_error = np.abs(np.log10(pred_load) - np.log10(true_load))
print('test residuals against load error: ', stats.spearmanr(residuals[-1], load_error))

metadata = pd.read_csv("/ddn_scratch/k5zhao/data/metadata_pergenome_clean.tsv", sep='\t')
metadata = metadata.set_index('original_SampleID')

test_metadata = metadata.loc[abs_test_df.index]
test_metadata['residual'] = residuals[2]

print('test residuals against bmi', stats.spearmanr(test_metadata['residual'], test_metadata['bmi']))

print('test residuals against age', stats.spearmanr(test_metadata['residual'], test_metadata['age']))

