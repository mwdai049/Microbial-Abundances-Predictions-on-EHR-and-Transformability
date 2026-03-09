# Microbial Abundances Predictions on EHR and Transformability

## Project Overview
This project investigates how microbiome abundance data can be used for biological interpretation and machine learning. We compare **relative** and **absolute** abundance representations, study how differential abundance methods behave across metadata groups, and test whether host metadata such as age, BMI, sex, and stool quality can be predicted from microbial profiles. We also explore whether absolute abundance can be reconstructed from relative abundance.

## Table of Contents

- [Setup] (https://github.com/mwdai049/Microbial-Abundances-Predictions-on-EHR-and-Transformability?tab=readme-ov-file#setup)
- [Data] (https://github.com/mwdai049/Microbial-Abundances-Predictions-on-EHR-and-Transformability?tab=readme-ov-file#data)
- [Directory structure] (https://github.com/mwdai049/Microbial-Abundances-Predictions-on-EHR-and-Transformability?tab=readme-ov-file#directory-structure)
- Project Scripts
  - [Differential Abundance] (https://github.com/mwdai049/Microbial-Abundances-Predictions-on-EHR-and-Transformability?tab=readme-ov-file#differential-abundance-analysis)
  - [Metadata Variable Prediction] (https://github.com/mwdai049/Microbial-Abundances-Predictions-on-EHR-and-Transformability?tab=readme-ov-file#metadata-variable-prediction)
    - [Age] (https://github.com/mwdai049/Microbial-Abundances-Predictions-on-EHR-and-Transformability?tab=readme-ov-file#age)
    - [BMI] (https://github.com/mwdai049/Microbial-Abundances-Predictions-on-EHR-and-Transformability?tab=readme-ov-file#BMI)
    - [Sex] (https://github.com/mwdai049/Microbial-Abundances-Predictions-on-EHR-and-Transformability?tab=readme-ov-file#sex)
    - [Stool Quality] (https://github.com/mwdai049/Microbial-Abundances-Predictions-on-EHR-and-Transformability?tab=readme-ov-file#stool-quality)
  - [Absolute Abundance Modeling] (https://github.com/mwdai049/Microbial-Abundances-Predictions-on-EHR-and-Transformability?tab=readme-ov-file#modeling-absolute-abundance)

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
            BIRDMAn_metaG_pergenome.py
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
To run the files for the differential abundance analysis, please install the following R packages into your environment. You can install them via conda into your QIIME 2 environment:
```
conda install -c conda-forge r-base
conda install -c conda-forge r-vegan
```

To run the differential abundance analysis, run:
```
python src/differential-abundance/ANCOM-BC/ancombc_metaG_pergenome.py
```
or
```
python src/differential-abundance/BIRDMAn/BIRDMAn_metaG_pergenome.py
```

Each file will run either ANCOM-BC or BIRDMAn differential abundance analysis. The resulting figures will be found in ```out/figs/ANCOM-BC```.

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
