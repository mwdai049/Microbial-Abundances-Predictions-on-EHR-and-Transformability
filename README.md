# Microbial Abundances Predictions on EHR and Transformability

## Project Overview
This project investigates how microbiome abundance data can be used for biological interpretation and machine learning. We compare **relative** and **absolute** abundance representations, study how differential abundance methods behave across metadata groups, and test whether host metadata such as age, BMI, sex, and stool quality can be predicted from microbial profiles. We also explore whether absolute abundance can be reconstructed from relative abundance.

## Table of Contents

- [Setup](#setup)
- [Data](#data)
- [Directory structure](#directory-structure)
- Project Scripts
  - [Differential Abundance](#differential-abundance-analysis)
  - [Metadata Variable Prediction](#metadata-variable-prediction)
    - [Age](#age)
    - [BMI](#bmi)
    - [Sex](#sex)
    - [Stool Quality](#stool-quality)
  - [Absolute Abundance Modeling](#modeling-absolute-abundance)
- [Credit](#credit)
- [Contributions](#contributions)

## Setup
### QIIME 2
To upload and process the data, you must install QIIME 2. We used the moshpit distribution, and the yml files are in the project directory. Make sure you have Conda or Mamba installed first (Miniconda, Anaconda, or Mambaforge).

### Conda
#### Linux/Windows WSL
```
conda env create \
  --name qiime2-moshpit-2025.7 \
  --file qiime2-moshpit-ubuntu-latest-conda.yml
```

#### macOS (Apple Silicon)
```
CONDA_SUBDIR=osx-64 conda env create \
  --name qiime2-moshpit-2025.7 \
  --file qiime2-moshpit-latest-conda.yml
conda activate qiime2-moshpit-2025.7
conda config --env --set subdir osx-64
```

#### macOS (Intel)
```
conda env create \
  --name qiime2-moshpit-2025.7 \
  --file qiime2-moshpit-latest-conda.yml
```

### Testing the installation
To activate, run: 
```
conda activate qiime2-moshpit-2025.7
```

You can test your installation by running:
```
conda deactivate
conda activate qiime2-amplicon-2025.7
qiime info
```

### Other dependencies
Once your QIIME 2 environment is activated, install the remaining dependencies via pip:
```
conda activate qiime2-moshpit-2025.7
pip install -r requirements.txt
```

## Data
Our models were trained on human subjects data with personally identifying information (PII). Thus, while our code contains the filepaths to load the data, the data itself is hosted only on the Barnacle2 cluster. 

## Directory Structure
```text
data_preprocessing/
    preprocessing.py
    traintest_split.py
out/
    figs/
      ...
  tables/
      ...
src/
    absolute-abundance-modeling/
        best_model.py
        linear_models.py
        nonlinear_models.py
        pacbio_modeling.py
    differential-abundance/
        ANCOM-BC/
            ancombc_metaG_pergenome.ipynb
            ancombc_metaG_pergenome.py
        BIRDMAn/
            BIRDMAn_analysis.py
            BIRDMAn_env.yml
            generate_BIRDMAn_scripts.py
    metadata-variable-prediction/
        age-prediction/
            age_numeric_predictor_final.py
            age_rf_classifier.py
        bmi-prediction/
            bmi_classification.py
            bmi_regression.py
        bmi-stool-model-analysis/
            bmi_stool_prediction_complete_analysis.ipynb
            shap_analysis.py
        sex-prediction/
            sex_lr_svm_final.py
            sex_rf_classifier_final.py
        stool_prediction/
            stool_classification.py
        
.gitignore
README.md
qiime2-moshpit-macos-latest-conda.yml
qiime2-moshpit-ubuntu-latest-conda.yml
requirements.txt
```

## Differential Abundance Analysis

### Setup

To run the differential abundance analysis, install the following dependencies into your QIIME 2 conda environment:

```bash
conda install -c conda-forge r-base
conda install -c conda-forge r-vegan
```

The following Python packages are also required (available via pip or conda): `qiime2`, `biom-format`, `pandas`, `numpy`, `matplotlib`, `seaborn`, and `rpy2`.

### Inputs

The analysis expects the following input files:

- **Feature table** (`.qza`): Relative abundance metagenomic feature table in QIIME 2 format
- **Host metadata** (`.tsv`): Sample metadata including `bowel_movement`, `bowel_movement_quality`, `age`, `bmi`, and `sex`
- **Absolute abundance table** (`.csv`): Measured absolute abundance training data
- **Estimated absolute abundance table** (`.tsv`): Synthetic/estimated absolute abundance training data
- **Taxonomy annotations** (`.tsv`): WoLR2 taxonomy file mapping feature IDs to taxonomic ranks

### Running the Analysis

The script and notebook for the differential abundance analysis can be found in:

```
src/differential-abundance/
```

The analysis runs ANCOM-BC across three feature table types — **relative**, **true absolute**, and **estimated absolute abundance** — using the following model formula:

```
bowel_movement + bowel_movement_quality + age_bin + bmi_bin + sex
```

Reference levels are: female sex, Type 3/4 stool, age 40–49, and normal BMI. A prevalence cutoff of 10% (`prv_cut = 0.10`) and Benjamini–Hochberg FDR correction are applied. PERMANOVA (via R's `vegan::adonis2`, marginal by-term) is also run on Bray–Curtis distances to assess overall community-level variance explained by each metadata factor.

### Outputs

Results figures are saved to:

```
out/figs/ANCOM-BC/
```

The following figures are produced:

| File | Description |
|---|---|
| `sig_counts.png` | Bar chart of significant differential taxa per metadata term (q < 0.05) |
| `ancom-bc-genus-5targets_rel_heatmaps.png` | 5-panel genus-level LFC heatmap — relative abundance |
| `ancom-bc-genus-5targets_abs_heatmaps.png` | 5-panel genus-level LFC heatmap — true absolute abundance |
| `ancom-bc-genus-5targets_synth_heatmaps.png` | 5-panel genus-level LFC heatmap — estimated absolute abundance |
| `ancom-bc-final_heatmaps.png` | 3-panel species-level heatmap (age, BMI, bowel movement quality) for poster |
| `ancombc_age_trends.png` | Genus LFC trend lines across age bins |
| `ancombc_bmi_trends.png` | Genus LFC trend lines across BMI bins |
| `comm_var_vs_da_sig.png` | Scatter plot of PERMANOVA R² vs number of significant ANCOM-BC features per metadata factor |

### BIRDMAn

### Setup

To run BIRDMAn, you will need to install a separate environment to avoid dependency issues. You can run this command to do so:

```bash
conda env create --name q2-birdman -f src/differential-abundance/BIRDMAn/BIRDMAn_env.yml
```

### Creating the scripts

BIRDMAn requires heavy compute resources to run, and has a long runtime. Running this script will filter the input feature tables and create SLURM scripts that can be submitted to the cluster to run BIRDMAn:

```bash
python src/differential-abundance/BIRDMAn/generate_BIRDMAn_scripts.py
```

Then to submit the scripts, cd into the created directory and run:
```bash
sbatch run_birdman.sh
```

### Running the analysis

The BIRDMAn results were primarily used to identify taxa related to metadata groups and visualize them in a heatmap. The script to run this is in ```src/differential-abundance/BIRDMAn/BIRDMAn_analysis.py```

The primary output is found here: ```out/figs/BIRDMAn/birdman_heatmaps.png```

## Metadata Variable Prediction
You can find the scripts to predict different metadata variables in ```src/metadata-variable-prediction```. 

### Plate diagnosis
You can find the script for diagnosing bias in different sampling plates in  ```bmi-stool-model-analysis/bmi_stool_prediction_complete_analysis.ipynb```. The SHAP analysis was performed in ```bmi-stool-model-analysis/shap_analysis.py```.

### Age
```age-prediction``` contains two scripts: ```age_numeric_predictor_final.py``` and ```age_rf_classifier.py```. 
* ```age_numeric_predictor_final.py``` runs a Random Forest Regressor, a Gradient Boosted Regressor, and a Radial Basis Function (RBF) Support Vector Machine (SVM) to predict age from microbiome data, and constructs 95% confidence intervals via paired stratified bootstrap testing to verify results. SHAP analysis is conducted to identify influential taxa. The resulting figures can be found in ```out/figs/age-prediction/age-numeric-predictor```. 
* ```age_rf_classifier``` runs a Random Forest Classifier to predict whether a sample falls into one of the following age ranges: ‘under 20’, ‘20 - 35’, ‘35 - 50’, ‘50 - 65’, ‘65 - 80’, and ‘over 80’. The script constructs 95% confidence intervals via paired stratified bootstrap procedure to verify results. SHAP analysis is conducted to identify influential taxa.The resulting figures can be found in ```out/figs/age-prediction/age-prediction-classifier```. 

### BMI
```bmi-prediction``` contains two scripts: ```bmi_classification.py``` and ```bmi_regression.py```.
* ```bmi_classification.py``` runs a XXX to predict BMI from microbiome data. SHAP analysis is conducted to identify influential taxa.The resulting figures can be found in ```out/figs/bmi-prediction/bmi-classification```.
* ```bmi_regression.py``` runs a XXX. SHAP analysis is conducted to identify influential taxa.The resulting figures can be found in ```out/figs/bmi-prediction/bmi-regression```.

### Sex
```sex-prediction``` contains two scripts: ```sex_lr_svm_final.py``` and ```sex_rf_classifier_final.py```. 
* ```sex_lr_svm_final.py``` runs a Logistic Regressor model and RBF SVM to predict sex from microbiome data, and compares the results to the Random Forest Classifier constructed in ```age_rf_classifier```, with results verified by constructing 95% confidence intervals via paired stratified bootstrap testing. SHAP analysis is conducted to identify influential taxa. The resulting figures can be found in ```out/figs/sex-prediction/sex-other-models```.
* ```age_rf_classifier``` runs a Random Forest Classifier on different featuresets to predict sex based on microbiome data. The script constructs 95% confidence intervals via paired stratified bootstrap procedure to verify results. SHAP analysis is conducted to identify influential taxa. The resulting figures can be found in ```out/figs/sex-prediction/sex-random-forest-classifier```.

### Stool Quality
```stool-prediction``` contains one script: ```stool_classification.py```. ```stool_classification.py``` runs a XXX to predict stool quality fro microbiome data. SHAP analysis is conducted to identify influential taxa. The resulting figures can be found in ```out/figs/stool-prediction```.

## Modeling Absolute Abundance
The scripts for creating synthetic absolute abundance data from relative abundance data are located in ```absolute-abundance-modeling```. 

This directory contains four scripts:
* ```best_model.py```
  * This script runs the best performing model for generating synthetic absolute abundance data.
* ```linear_models.py```
  * This script runs a series of linear models for performance comparison
* ```nonlinear_models.py```
  * This scripts runs a series of non linear models for performance comparison.
* ```pacbio_modeling.py```
  * This script runs the best performing model on a new, independent dataset and perform subsampling on the original dataset for comparison.

To run any of these scripts, run:
```
python filename.py
```

The resulting figures can be found in ```out/figs/absolute-abundance-modeling```.

The resulting model metrics can be found in ```out/tables/absolute-abundance-modeling```

## Credit
This project was a collaboration between Monica Dai, Katelyn Zhao, Camille Sicat, Nathan Wang, and Sophie Wang at UC San Diego under the Halıcıoğlu Data Science Institute. Our work would not have been possible without the mentorship of Dr. Rob Knight, Dr. Sam Degregori, and Michael Iter, and other members of the Knight Lab.

### Contributions
**Monica Dai** ran the initial BIRDMAn analysis and also helped develop the model for predicting absolute abundance from relative abundance data. 

**Camille Sicat** developed the models for predicting age and sex from microbiome data, and comparing the performance of absolute vs. relative abundance data. 

**Nathan Wang** ran the ANCOM analysis on the absolute vs. relative abundance data and extensively helped write the report. 

**Sophie Wang** developed the models for predicting BMI and stool quality from microbiome data, and comparing the performance of absolute vs. relative abundance data. 

**Katelyn Zhao** performed PCoA analysis on metadata features and also developed the model for predicting absolute abundance from relative abundance data. 
