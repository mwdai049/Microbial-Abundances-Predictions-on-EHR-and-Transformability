# Microbial Abundances Predictions on EHR and Transformability
## Credit
This project was a collaboration between Monica Dai, Katelyn Zhao, Camille Sicat, Nathan Wang, and Sophie Wang at UC San Diego under the Halıcıoğlu Data Science Institute. Our work would not have been possible without the mentorship of Dr. Rob Knight, Dr. Sam Degregori, and Michael Iter, and other members of the Knight Lab.

### Contributions
**Monica Dai** ran the initial BIRDMAn analysis and also helped develop the model for predicting absolute abundance from relative abundance data. 

**Camille Sicat** developed the models for predicting age and sex from microbiome data, and comparing the performance of absolute vs. relative abundance data. 

**Nathan Wang** ran the ANCOM analysis on the absolute vs. relative abundance data and extensively helped write the report. 

**Sophie Wang** developed the models for predicting BMI and stool quality from microbiome data, and comparing the performance of absolute vs. relative abundance data. 

**Katelyn Zhao** performed PCoA analysis on metadata features and also developed the model for predicting absolute abundance from relative abundance data. 

## PLEASE READ
Most of the work was performed in the Knight Lab's Barnacle2 cluster. As such, this Github repository only contains clean, finalized scripts. 

## Data
Our models were trained on human subjects data with personally identifying information (PII). Thus, while our code contains the filepaths to load the data, the data itself is hosted only on the Barnacle2 cluster. 

## Setup
### QIIME 2
To upload and process the data, you must install QIIME 2. We used the moshpit distribution, and the yml files are in the project directory. Make sure you have Conda or Mamba installed first (Miniconda, Anaconda, or Mambaforge).

### Conda
#### Linux/Windows WSL
Run:
```
conda env create \
  --name qiime2-moshpit-2025.7 \
  --file https://raw.githubusercontent.com/qiime2/distributions/refs/heads/dev/2025.7/moshpit/released/qiime2-moshpit-ubuntu-latest-conda.yml
```
To activate, run:
```
conda activate qiime2-moshpit-2025.7
```

#### macOS (Apple Silicon)
Run:
```
CONDA_SUBDIR=osx-64 conda env create \
  --name qiime2-moshpit-2025.7 \
  --file qiime2-moshpit-latest-conda.yml
```
Then:
```
conda activate qiime2-moshpit-2025.7
conda config --env --set subdir osx-64
```

#### macOS (Intel)
Run:
```
conda env create \
  --name qiime2-moshpit-2025.7 \
  --file qiime2-moshpit-latest-conda.yml
```
To activate, run: 
```
conda activate qiime2-moshpit-2025.7
```

### Testing the installation
You can test your installation by running:
```
conda deactivate
conda activate qiime2-amplicon-2025.7
qiime info
```

# Other dependencies
Once your QIIME 2 environment is activated, install the remaining dependencies via pip:
```
conda activate qiime2-moshpit-2025.7
pip install -r requirements.txt
```

## Differential Abundance Analysis
### Setup
To run the files for the differential abundance analysis, please install the following R packages into your environment. You can install them via conda into your QIIME 2 environment:
```
conda install -c conda-forge r-base
conda install -c conda-forge r-vegan
```
The script for the differential abundance analysis can be found in ```src/differential-abundance```. 
The resulting figures can be found in ```out/figs/ANCOM-BC```.

## Metadata Variable Prediction
You can find the scripts to predict different metadata variables in ```src/metadata-variable-prediction```. 

### Plate diagnosis
You can find the script for diagnosing bias in different sampling plates in  ```bmi-stool-model-analysis/bmi_stool_prediction_complete_analysis.ipynb```. The SHAP analysis was performed in ````bmi-stool-model-analysis/shap_analysis.py```.

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

## Modelling Absolute Abundance
The scripts for creating synthetic absolute abundance data from relative abundance data can be found in ``absolute-abundance-modeling```. There are four scripts:
* ```best_model.py```
* ```linear_models.py```
* ```nonlinear_models.py```
* ```pacbio_modeling.py```

The resulting figures can be found in ```out/figs/absolute-abundance-modeling```.

## Directory Structure
```text
data_preprocessing/
    preprocessing.py
    traintest_split.py
out/
    model_summary.xlsx
    figs/
      ANCOM-BC/
        ancombc_age_heatmap.png
        ancombc_age_trends.png
        ancombc_bmi_heatmap.png
        ancombc_bmi_trends.png
        ancombc_bowel_movement_quality_barh.png
        ancombc_bowel_movement_quality_heatmap.png
        ancombc_bowel_movement_type_barh.png
        ancombc_bowel_movement_type_heatmap.png
        ancombc_genus_lfc_all_covariates.png
        ancombc_max_genus_lfc_per_covariate.png
        ancombc_sex_barh.png
        ancombc_sex_heatmap.png
        comm_var_vs_da_sig.png
        sig_counts.png
      age-prediction/
        age-numeric-predictor/
          gbr_comparison.png
          rbf_comparison.png
          rf_reg_comparison.png
          rf_taxa_comparison.png
        age-prediction-classifier/
          ROC-AUC/
            final_rf_roc.png
            first_pass_abs_rf_roc.png
            first_pass_rel_rf_roc.png
            retuned_abs_rf_roc.png
            retuned_rel_rf_roc.png
            rf_macro_auc_comparison.png
          confusion_matrix/
            age_abs_rf_sex_cm.png
            age_abs_rf_sex_gene_cm.png
            age_rel_rf_sex_cm.png
            age_rel_rf_sex_gene_cm.png
            balanced_rf_cm_comparison.png
            final_rf_cm.png
            first_pass_abs_rf_cm.png
            first_pass_rel_rf_cm.png
            retuned_abs_rf_cm.png
            retuned_rel_rf_cm.png
          feature_importance/
            abs_rf_nogrid_SHAP.png
            abs_rf_nogrid_top20_features.png
            final_abs_rf_top20_features.png
            final_rel_rf_top20_features.png
            rel_rf_nogrid_SHAP.png
      bmi-prediction/
        bmi-classification/
          confusion_matrix/
            bmi_bin_Absolute_HGB_confusion.png
            bmi_bin_Absolute_RandomForest_confusion.png
            bmi_bin_Absolute_SVM_RBF_confusion.png
            bmi_bin_Relative_HGB_confusion.png
            bmi_bin_Relative_RandomForest_confusion.png
            bmi_bin_Relative_SVM_RBF_confusion.png
          roc_plots/
            bmi_bin_Absolute_HGB_roc.png
            bmi_bin_Absolute_RandomForest_roc.png
            bmi_bin_Absolute_SVM_RBF_roc.png
            bmi_bin_Relative_HGB_roc.png
            bmi_bin_Relative_RandomForest_roc.png
            bmi_bin_Relative_SVM_RBF_roc.png
          shap_outputs/
            classification_bmi_bin_Absolute_HGB_bar.png
            classification_bmi_bin_Absolute_HGB_beeswarm_class_obese.png
            classification_bmi_bin_Relative_HGB_bar.png
            classification_bmi_bin_Relative_HGB_beeswarm_class_obese.png
        bmi-regression/
            regression_plots/
              bmi_Absolute_HGB.png
              bmi_Absolute_RandomForest.png
              bmi_Absolute_SVM_RBF.png
              bmi_Relative_HGB.png
              bmi_Relative_RandomForest.png
              bmi_Relative_SVM_RBF.png
            shap_outputs/
              regression_bmi_Absolute_HGB_bar.png
              regression_bmi_Absolute_HGB_beeswarm.png
              regression_bmi_Relative_HGB_bar.png
              regression_bmi_Relative_HGB_beeswarm.png
      sex-prediction/
        sex-prediction-other-models/
           ROC-AUC/
            sex_lr_roc_comparison.png
            sex_svm_roc_comparison.png
          confusion_matrix/
            sex_abs_lr_cm.png
            sex_abs_svm_cm.png
            sex_lr_cm_comparison.png
            sex_rel_lr_cm.png
            sex_rel_svm_cm.png
            sex_svm_cm_comparison.png
          feature_importance/
            sex_abs_lr_shap.png
            sex_abs_lr_top20_features.png
            sex_rel_lr_shap.png
            sex_rel_lr_top20_features.png
        sex-random-forest-classifier/
          ROC-AUC/
            sex_rf_roc_comparison.png
          confusion_matrix/
            balanced_sex_abs_rf_ageless_cm.png
            balanced_sex_abs_rf_ageless_miss_cm.png
            balanced_sex_abs_rf_cm.png
            balanced_sex_rel_rf_ageless_cm.png
            balanced_sex_rel_rf_ageless_miss_cm.png
            balanced_sex_rel_rf_cm.png
            sex_rf_cm_comparison.png
            unbalanced_sex_abs_rf_cm.png
            unbalanced_sex_rel_rf_cm.png
          feature_importance/
            sex_abs_rf_shap.png
            sex_abs_rf_top20_features.png
            sex_rel_rf_shap.png
            sex_rel_rf_top20_features.png
    stool-prediction/
      confusion_matrix/
        bowel_movement_clean_Absolute_HGB_confusion.png
        bowel_movement_clean_Absolute_RandomForest_confusion.png
        bowel_movement_clean_Absolute_SVM_RBF_confusion.png
        bowel_movement_clean_Relative_HGB_confusion.png
        bowel_movement_clean_Relative_RandomForest_confusion.png
        bowel_movement_clean_Relative_SVM_RBF_confusion.png
      roc_plots/
        bowel_movement_clean_Absolute_HGB_roc.png
        bowel_movement_clean_Absolute_RandomForest_roc.png
        bowel_movement_clean_Absolute_SVM_RBF_roc.png
        bowel_movement_clean_Relative_HGB_roc.png
        bowel_movement_clean_Relative_RandomForest_roc.png
        bowel_movement_clean_Relative_SVM_RBF_roc.png
      shap_outputs/
        classification_bowel_movement_clean_Absolute_HGB_bar.png
        classification_bowel_movement_clean_Absolute_HGB_beeswarm_class_normal.png
        classification_bowel_movement_clean_Relative_HGB_bar.png
        classification_bowel_movement_clean_Relative_HGB_beeswarm_class_normal.png
  tables/
    absolute-abundance-modeling/
      best_model_metrics.csv
      model_summary.csv
      pacbio_metrics_summary.csv
    age-prediction/
      age_reg_model_comparisons.csv
    bmi-predicion/
      baseline_results.csv
      cls_bmi_results_all.csv
      reg_bmi_results_all.csv
    sex-prediction/
        sex_model_comparisons.csv
    stool-prediction/
      baseline_results.csv
      cls_stool_results_all.csv
src/
    absolute-abundance-modeling/
        linear_models.py
        nonlinear_models.py
    differential-abundance/
        BIRDMAn/
            BIRDMAn_metaG_pergenome.py
        ANCOM-BC/
            ancombc_metaG_pergenome.ipynb
            ancombc_metaG_pergenome.py
    metadata-variable-prediction/
        age-prediction/
          age_numeric_predictor_final.py
          age_rf_classifier.py
        bmi-prediction/
          bmi_HGBR_rf_regressor.ipynb
        sex-rediction/
          sex_lr_svm_final.py
          sex_rf_classifier_final.py
.gitignore
README.md
qiime2-moshpit-macos-latest-conda.yml
qiime2-moshpit-ubuntu-latest-conda.yml
```


