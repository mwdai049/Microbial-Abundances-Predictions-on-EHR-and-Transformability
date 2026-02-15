import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

from qiime2 import Metadata, Artifact
from qiime2.plugins.taxa.visualizers import barplot

from pathlib import Path

ft = Artifact.load("/ddn_scratch/k5zhao/data/metaG-pergenome-clean.qza")
ft_df = ft.view(pd.DataFrame)

metadata = pd.read_csv("/ddn_scratch/k5zhao/data/metadata_pergenome_clean.tsv", sep='\t')
metadata = metadata.set_index('#SampleID')

outdir = Path("./out/birdman")
outdir.mkdir(parents=True, exist_ok=True)

categorical_vars = ['bowel_movement', 'bowel_movement_quality']
continuous_vars = ['age', 'bmi']
binary_vars = ['sex']

all_vars = continuous_vars + categorical_vars + binary_vars

def subset_and_save(ft, metadata_df, variables, output_prefix):
    """
    Subset feature table and metadata to samples with non-null values for specified variables.
    """
    ft_df = ft.copy()
    valid_samples = metadata_df[variables].dropna().index
    common_samples = ft_df.index.intersection(valid_samples)    
    
    print(f"  {output_prefix}: {len(common_samples)} samples with complete data")
    
    if len(common_samples) == 0:
        print(f"  WARNING: No samples with complete data for {variables}")
        return None, None
    
    ft_subset = ft_df.loc[common_samples]
    ft_subset = ft_subset.loc[ft_subset.sum(axis=1) > 0]
    
    metadata_subset = metadata_df.loc[common_samples, variables]
    metadata_subset.index.name = "SampleID"
    
    return ft_subset, metadata_subset

def create_run_script(model_dir, table_name, model_name, formula):
    """Create a bash script to run q2-birdman for this specific model."""
    script_path = model_dir / "run_birdman.sh"
    
    script_content = f"""#!/bin/bash
#SBATCH --job-name=NPH-{table_name}-{model_name}
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/home/mwdai/projects/capstone/logs/NPH-{table_name}-{model_name}_%j.out
#SBATCH --error=/home/mwdai/projects/capstone/logs/NPH-{table_name}-{model_name}_%j.err
#SBATCH --mail-user=mwdai@ucsd.edu
#SBATCH --mail-type=ALL

set -e
source ~/.bashrc
conda activate q2-birdman-barnacle

export TMPDIR="/ddn_scratch/mwdai/tmp"

# Run Birdman
qiime birdman run \\
    --i-table table.qza \\
    --m-metadata-file metadata.tsv \\
    --p-formula "{formula}" \\
    --o-output-dir output.qza \\
    --p-threads 32 \\
    --verbose

# Generate visualization
qiime birdman plot \\
    --i-data output.qza \\
    --i-table table.qza \\
    --m-metadata-file metadata.tsv \\
    --o-visualization output.qzv \\
    --p-palette viridis \\
    --p-chart-style forest

echo "✓ Completed: {table_name}.{model_name}"
"""
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    script_path.chmod(0o755)
    
    return script_path

# format: (model_name, variables, formula, description)
model_configs = []

for col in all_vars:
    if 'age' in col or 'sex' in col:
        description = col.replace('_', ' ')
        model_configs.append((col, [col], col, description))
    else:
        variables = [col, 'age']
        formula = '+'.join(variables)
        description = col.replace('_', ' ') + ' adjusted for age'
        model_configs.append((col, variables, formula, description))
    
for table_name, ft, metadata_df in [("nph-metaG-rel", ft_df, metadata)]:
    print(f"\nProcessing {table_name} table...")
    
    for model_name, variables, formula, description in model_configs:
        print(f"  Subsetting for model: {model_name} ({description})")
        
        model_dir = outdir / f"{table_name}.{model_name}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        ft_output = model_dir / "table.qza"
        metadata_output = model_dir / "metadata.tsv"
        
        output_prefix = f"{table_name}.{model_name}"
        ft_subset, metadata_subset = subset_and_save(
            ft_df, metadata_df, variables, output_prefix
        )
        
        if ft_subset is not None:
            ft_artifact_subset = Artifact.import_data(
                "FeatureTable[Frequency]", 
                ft_subset
            )
            ft_artifact_subset.save(ft_output)
            
            metadata_subset.to_csv(metadata_output, sep='\t')
            
            script_path = create_run_script(
                model_dir, table_name, model_name, formula
            )
            
            print(f"    Saved to: {model_dir.name}/")
            print(f"    Samples: {len(ft_subset)}, Features: {len(ft_subset.columns)}")

print("\n✓ Preprocessing complete!")




