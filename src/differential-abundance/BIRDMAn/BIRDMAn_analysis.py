import pandas as pd
import numpy as np

from qiime2 import Metadata, Artifact

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


lineages = pd.read_csv('/home/mwdai/projects/adrc-analysis/ref/lineages.txt', sep='\t')
lineages = lineages.set_index('Feature ID')
lineages['s'] = lineages['Taxon'].apply(lambda x: x.split('; s__')[-1])
lineages.head()

def read_results(p):
    return Artifact.load(p).view(Metadata).to_dataframe()

def unpack_hdi_and_filter(df, col):
    df[['lower', 'upper']] = df[col].str.split(',', expand=True)
    # remove ( from lower and ) from upper and convert to float 
    df.lower = df.lower.str[1:].astype('float')
    df.upper = df.upper.str[:-1].astype('float')
    
    df['credible'] = np.where((df.lower > 0) | (df.upper < 0), 'yes', 'no')
    df.upper = df.upper - df[col.replace('hdi', 'mean')]
    df.lower = df[col.replace('hdi', 'mean')] - df.lower
    
    return df

data_dict = {
    'nph_age': '/home/mwdai/projects/capstone/out/birdman/metaG-rel.age_bin/output.qza',
    'nph_sex': '/home/mwdai/projects/capstone/out/birdman/metaG-rel.sex/output.qza',
    'nph_bmi': '/home/mwdai/projects/capstone/out/birdman/metaG-rel.bmi_bin/output.qza',
    'nph_bowel-movement-quality': '/home/mwdai/projects/capstone/out/birdman/metaG-rel.bowel_movement_quality/output.qza'
}

vars_to_check = {
    'age' : 'age_bin[T.18-29]_',
    'sex' : 'sex[T.male]_',
    'bmi' : 'bmi_bin[T.over]_',
    'bowel-movement-quality': 'bowel_movement_quality[T.Tends toward diarrhea]_'
}

# read in and filter BIRDMAn results for each variable
for k in data_dict.keys():
    df = read_results(data_dict[k])
    for v in vars_to_check: 
        if v in k: 
            var = vars_to_check[v] 
    print(k)
    print('Unfiltered Shape:  ' + str(df.shape))
    sub_df = unpack_hdi_and_filter(df, var+'hdi')
    print('Filtered Shape: ' + str(sub_df.loc[sub_df['credible'] == 'yes'].shape))
    sub_df['feature_name'] = sub_df.index.to_series().apply(lambda x: lineages.loc[x]['s'])
    
    data_dict[k] = sub_df.sort_values(by=var+'mean')

targets = ['nph_age', 'nph_bmi', 'nph_bowel-movement-quality']
dfs = []

# find top 5 enriched, top 5 depleted features, filter to dereplicate same genomeID -> taxon mapping
for target in targets:
    print(f'Processing {target}...')
    df = data_dict[target]
    coef_df = df[[col for col in df.columns if ']_mean' in col]]
    
    enriched = pd.concat([coef_df.max(axis=1), df[['feature_name']]], axis=1)   # most enrichment across categories
    depleted = pd.concat([coef_df.min(axis=1), df[['feature_name']]], axis=1)   # most depletion across categories
    
    top_depleted = (
        depleted
        .sort_values(0, ascending=True)
        .drop_duplicates(subset='feature_name', keep='first')
        .head(5)
    )
    
    top_enriched = (
        enriched
        .sort_values(0, ascending=False)
        .drop_duplicates(subset='feature_name', keep='first')
    )
    
    top_enriched = top_enriched[
        ~top_enriched['feature_name'].isin(top_depleted['feature_name'])
    ].head(5)

    selected = top_enriched.index.to_list() + top_depleted.index.to_list()
    coef_df = coef_df.loc[selected]
    
    if target == 'nph_bmi':
        coef_df = coef_df.loc[:, ['bmi_bin[T.under]_mean', 'bmi_bin[T.over]_mean', 'bmi_bin[T.obese]_mean', 'bmi_bin[T.severe_obese]_mean']]
        
    coef_df.index = coef_df.index.to_series().apply(lambda x: lineages.loc[x]['s'])
    
    dfs.append(coef_df)
    
    print(coef_df.shape)

# heatmap to visualize the taxa selected from above
# made to be combined with the ANCOM-BC heatmap

fig, axes = plt.subplots(1, 3, figsize=(28, 10))
fig.subplots_adjust(wspace=0.65, right=0.92)

custom_cmap = LinearSegmentedColormap.from_list(
    "custom-blue-gray-red",
    ["#7297ec", "#efefef", "#d84e4c"],
    N=256
)

titles = [
    "Age - relative to 40-49",
    "BMI - relative to normal (18.5-25)",
    "Bowel movement quality - \nrelative to normal stool"
]

labels = [
    ['18-29', '30-39', '50-59', '60-69', '70+'],
    ['under', 'over', 'obese', 'sever obese'],
    ['constipation', 'diarrhea']
]

cbar_ax = fig.add_axes([0.865, 0.15, 0.015, 0.7])
vmin=min(df.min().min() for df in dfs)
vmax=max(df.max().max() for df in dfs)

# make only the 3rd heatmap narrower
pos = axes[2].get_position()
new_width = pos.width * 0.6
axes[2].set_position([pos.x0, pos.y0, new_width, pos.height])

for i in range(len(dfs)):
    df = dfs[i]
    ax = axes[i]
    title = titles[i]
    xlabels = labels[i]
    show_cbar = (i == len(dfs) - 1)
    
    sns.heatmap(
        df,
        ax=ax,
        center=0,
        cmap=custom_cmap,
        vmin=vmin, vmax=vmax,
        cbar=show_cbar,
        cbar_ax=cbar_ax,
        linewidths=1.0,
        linecolor="white"
    )

    ax.set_xlabel("")
    ax.set_ylabel("BIRDMAn", fontsize=22, fontweight="bold")

    ax.set_xticklabels(xlabels)
    
    ax.tick_params(axis="x", rotation=35, labelsize=15)
    ax.tick_params(axis="y", labelsize=12)
    
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
        
axes[1].set_ylabel("")
axes[2].set_ylabel("")

cbar = axes[2].collections[0].colorbar

cbar.ax.tick_params(labelsize=15)
cbar.set_label("Log-fold Change", fontsize=18, labelpad=20)

plt.savefig('./figs/birdman_heatmaps.png', dpi=500, bbox_inches='tight')
