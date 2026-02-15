import pandas as pd
import numpy as np
import qiime2 as q2
from qiime2 import Metadata, Artifact
from sklearn.model_selection import train_test_split

rel_ft = Artifact.load('/ddn_scratch/k5zhao/data/metaG-pergenome-clean.qza')
rel_ft_df = rel_ft.view(pd.DataFrame)

abs_ft = Artifact.load('/ddn_scratch/k5zhao/data/metaG-absquant-clean.qza')
abs_ft_df = abs_ft.view(pd.DataFrame)

rel_metadata = pd.read_csv('/ddn_scratch/k5zhao/data/metadata_pergenome_clean.tsv', sep='\t', index_col='#SampleID')
abs_metadata = pd.read_csv('/ddn_scratch/k5zhao/data/metadata_absquant_clean.tsv', sep='\t', index_col='#SampleID')

rel_ft_df['original_SampleID'] = rel_metadata['original_SampleID']
abs_ft_df['original_SampleID'] = abs_metadata['original_SampleID']

rel_ft_df.index = rel_ft_df['original_SampleID']
abs_ft_df.index = abs_ft_df['original_SampleID']

rel_ft_df = rel_ft_df.drop(columns=['original_SampleID'])
abs_ft_df = abs_ft_df.drop(columns=['original_SampleID'])

row_sums = rel_ft_df.sum(axis=1)
rel_ft_df_comp = rel_ft_df.div(row_sums, axis=0)

abs_ft_df['total'] = abs_ft_df.sum(axis=1)

final_df = rel_ft_df.merge(abs_ft_df[['total']], left_index=True, right_index=True, how='inner')
final_df_comp = rel_ft_df_comp.merge(abs_ft_df[['total']], left_index=True, right_index=True, how='inner')

RANDOM_STATE = 42
TEST_SIZE = 0.20         
VAL_SIZE = 0.20
val_relative = VAL_SIZE / (1 - TEST_SIZE)

train_val_df, test_df = train_test_split(
    final_df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True
)

train_df, val_df = train_test_split(
    train_val_df,
    test_size=val_relative,
    random_state=RANDOM_STATE,
    shuffle=True
)

train_df.to_csv("/ddn_scratch/k5zhao/data/model_training/train.csv", index=True)
val_df.to_csv("/ddn_scratch/k5zhao/data/model_training/val.csv", index=True)
test_df.to_csv("/ddn_scratch/k5zhao/data/model_training/test.csv", index=True)

train_val_df_comp, test_df_comp = train_test_split(
    final_df_comp,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True
)

train_df_comp, val_df_comp = train_test_split(
    train_val_df_comp,
    test_size=val_relative,
    random_state=RANDOM_STATE,
    shuffle=True
)

train_df.to_csv("/ddn_scratch/k5zhao/data/model_training/compositional/train.csv", index=True)
val_df.to_csv("/ddn_scratch/k5zhao/data/model_training/compositional/val.csv", index=True)
test_df.to_csv("/ddn_scratch/k5zhao/data/model_training/compositional/test.csv", index=True)
