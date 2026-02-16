import pandas as pd
import numpy as np
import qiime2 as q2
import seaborn as sns
import matplotlib.pyplot as plt
import re
import os
from statistics import mean, stdev
from skbio import DistanceMatrix
from qiime2.plugins import diversity, emperor
from qiime2 import Metadata, Artifact
from qiime2.plugins.feature_table.methods import rarefy
from q2_types.tree import NewickFormat
objects.packages import isinstalled

import skbio

import plotly.express as px

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

%matplotlib inline

### Loading the Data ###

relative_ft = Artifact.load('/ddn_scratch/miter/nph-tables/metaG-pergenome-filteredblank30-feature-table.qza')
relative_ft_df = relative_ft.view(pd.DataFrame)

absolute_ft = Artifact.load('/ddn_scratch/miter/nph-tables/metaG-absquant-filteredblank30-feature-table.qza')
absolute_ft_df = absolute_ft.view(pd.DataFrame)

metadata_df = pd.read_csv('/ddn_scratch/miter/nph-tables/nph_metadata.tsv', sep='\t', index_col='#SampleID')

### Metadata Cleaning and Preprocessing ###

metadata_df['bowel_movement'] = metadata_df['bowel_movement'].astype(str).str.replace('"', '').radd('"').add('"')

metadata_df.loc[metadata_df['sample_type'] == 'Stool', 'sample_type'] = 'stool'
metadata_df.loc[metadata_df['sample_type'] == 'control shield', 'sample_type'] = 'control_shield'
metadata_df.loc[metadata_df['sample_type'] == 'control blank', 'sample_type'] = 'control_blank'
metadata_df.loc[metadata_df['sample_type'] == 'feces', 'sample_type'] = 'stool'

s = metadata_df['sex_at_birth'].astype(str).str.strip()
metadata_df['sex_at_birth'] = metadata_df['sex_at_birth'] = s.where(~s.str.contains("_", na=False), s.str.split("_", n=1).str[1])

x = pd.to_numeric(metadata_df['latitude'], errors="coerce")
metadata_df['latitude'] = metadata_df['latitude'] = metadata_df['latitude'].where(x.isna(), x)

metadata_df['experiment_design_description'] = np.where(metadata_df['experiment_design_description'] == 
                                                     'Fecal sequencing', 'fecal sequencing',
                                                     metadata_df['experiment_design_description'] )

metadata_df['experiment_design_description'] = metadata_df['experiment_design_description'].str.lower()

metadata_df['bowel_movement_quality'] = metadata_df['bowel_movement_quality'].astype(str).str.replace('"', '').radd('"').add('"')

metadata_df['dna_extracted'] = metadata_df['dna_extracted'].apply(lambda x: True if x == 'TRUE' else x)

metadata_df['qiita_sample_type'] = metadata_df['qiita_sample_type'].str.replace(' ', '_')

metadata_df['mass_storage_tube_and_storage_liquid_before_sample_mg'] = metadata_df['mass_storage_tube_and_storage_liquid_before_sample_mg'].str.replace('.0', '', regex=False)

metadata_df.loc[metadata_df['replicate'] == 'Rep 2', 'replicate'] = 'Rep2'

metadata_df.loc[metadata_df['longitude'] == '-100.0', 'longitude'] = '-100'

metadata_df.loc[metadata_df['elevation'] == '193.0', 'elevation'] = '193'
metadata_df.loc[metadata_df['elevation'] == '0.0', 'elevation'] = '0'

metadata_df['collection_date'] = pd.to_datetime(metadata_df['collection_date'], errors='coerce').dt.strftime('%Y-%m-%d-%H:%M').where(pd.to_datetime(metadata_df['collection_date'], errors='coerce').notna(), metadata_df['collection_date'])

metadata_df = metadata_df[~metadata_df.index.str.lower().str.contains('blank')]

metadata_df['age'] = metadata_df['age'].replace({
    'unknown': np.nan,
    'NA': np.nan,
    'not provided': np.nan,
    '940': np.nan
})

metadata_df['bmi'] = metadata_df['bmi'].replace({
    'not provided': np.nan,
    'NA': np.nan,
    'unknown': np.nan
})

metadata_df['age'] = metadata_df['age'].apply(lambda x: int(x) if type(x) == str else x)
metadata_df['bmi'] = metadata_df['bmi'].apply(lambda x: float(x) if type(x) == str else x)

metadata_df_clean = metadata_df[(metadata_df['age'] >= 18) & (metadata_df['age'] < 100) & 
                                (metadata_df['bmi'] >= 10) & ((metadata_df['bmi'] < 100))].copy()

### Saving Cleaned Tables ###

relative_table_qza = Artifact.import_data("FeatureTable[Frequency]", relative_ft_df)
relative_table_qza.save("/ddn_scratch/k5zhao/data/metaG-pergenome-clean.qza")

absolute_table_qza = Artifact.import_data("FeatureTable[Frequency]", absolute_ft_df)
absolute_table_qza.save("/ddn_scratch/k5zhao/data/metaG-absquant-clean.qza")

relative_metadata_df.to_csv(
    '/ddn_scratch/k5zhao/data/metadata_pergenome_clean.tsv',
    sep="\t",
    index=True,          
    index_label="#SampleID"
)

absolute_metadata_df.to_csv(
    '/ddn_scratch/k5zhao/data/metadata_absquant_clean.tsv',
    sep="\t",
    index=True,          
    index_label="#SampleID"
)

### Preprocessing Data For Model Training ###

abs_ft_meta = absolute_metadata_df.merge(absolute_ft_df, how='inner', left_index=True, right_index=True)
rel_ft_meta = relative_metadata_df.merge(relative_ft_df, how='inner', left_index=True, right_index=True)

rel_ft_df['original_SampleID'] = rel_metadata['original_SampleID']
abs_ft_df['original_SampleID'] = abs_metadata['original_SampleID']

rel_ft_df.index = rel_ft_df['original_SampleID']
abs_ft_df.index = abs_ft_df['original_SampleID']

rel_ft_df = rel_ft_df.drop(columns=['original_SampleID'])
abs_ft_df = abs_ft_df.drop(columns=['original_SampleID'])

final_rel_df = rel_df.merge(rel_metadata, how='left', left_index=True, right_on='original_SampleID')
final_rel_df.index = final_rel_df['original_SampleID']
final_rel_df = final_rel_df.drop(columns=['original_SampleID'])

final_abs_df = abs_df.merge(abs_metadata, how='left', left_index=True, right_on='original_SampleID')
final_abs_df.index = final_abs_df['original_SampleID']
final_abs_df = final_abs_df.drop(columns=['original_SampleID'])