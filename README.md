# Microbial Abundances Predictions on EHR and Transformability
## Credit
This project was a collaboration between Monica Dai, Katelyn Zhao, Camille Sicat, Nathan Wang, and Sophie Wang at UC San Diego under the Halıcıoğlu Data Science Institute. Our work would not have been possible without the mentorship of Dr. Rob Knight and the assistance of the Knight Lab. 

### Contributions
**Monica Dai** ran the initial BIRDMAn analysis and also helped develop the model for predicting absolute abundance from relative abundance data. 

**Camille Sicat** worked on developing the model for predicting age from microbiome data, and comparing the performance of absolute vs. relative abundance data. 

**Nathan Wang** ran the ANCOM analysis on the absolute vs. relative abundance data and extensively helped write the report. 

**Sophie Wang** worked on developing the model for predicting sex from microbiome data, and comparing the performance of absolute vs. relative abundance data. 

**Katelyn Zhao** performed PCoA analysis on metadata features and also developed the model for predicting absolute abundance from relative abundance data. 

## PLEASE READ
Most of the work was performed in the Knight Lab's Barnacle2 cluster. As such, this Github repository only contains clean, finalized scripts. 

## Setup
### QIIME 2
To upload and process the data, you must install QIIME 2. We used the amplicon distribution, and the yml files are in the project directory. Make sure you have Conda or Mamba installed first (Miniconda, Anaconda, or Mambaforge).

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
  --name qiime2-amplicon-2025.7 \
  --file qiime2-environment-macos.yml
```
Then:
```
conda activate qiime2-amplicon-2025.7
conda config --env --set subdir osx-64
```

#### macOS (Intel)
Run:
```
conda env create \
  --name qiime2-amplicon-2025.7 \
  --file qiime2-environment-macos.yml
```
To activate, run: 
```
conda activate qiime2-amplicon-2025.7
```

### Testing the installation
You can test your installation by running:
```
conda deactivate
conda activate qiime2-amplicon-2025.7
qiime info
```

## Data
Our models were trained on human subjects data with personally identifying information (PII). Thus, while our code contains the filepaths to load the data, the data itself is hosted only on the Barnacle2 cluster. 

### Directory Structure
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
          age_reg_model_comparisons.csv
          gbr_comparison.png
          rbf_comparison.png
          rf_reg_comparison.png
          rf_taxa_comparison.png
        age-prediction-classifier/
          abs_rf_nogrid_SHAP.png
          abs_rf_nogrid_top20_features.png
          age_abs_rf_sex_cm.png
          age_abs_rf_sex_gene_cm.png
          age_rel_rf_sex_cm.png
          age_rel_rf_sex_gene_cm.png
          balanced_rf_cm_comparison.png
          final_abs_rf_top20_features.png
          final_rel_rf_top20_features.png
          final_rf_cm.png
          final_rf_roc.png
          first_pass_abs_rf_cm.png
          first_pass_abs_rf_roc.png
          first_pass_rel_rf_cm.png
          first_pass_rel_rf_roc.png
          rel_rf_nogrid_SHAP.png
          retuned_abs_rf_cm.png
          retuned_abs_rf_roc.png
          retuned_rel_rf_cm.png
          retuned_rel_rf_roc.png
          rf_macro_auc_comparison.png
        age-prediction-regressor/
          abs_rf_reg.png
          abs_rf_reg_spline.png
          rel_rf_reg.png
          rel_rf_reg_spline.png
      sex-prediction/
        sex-prediction-other-models/
          sex_abs_lr_cm.png
          sex_abs_lr_shap.png
          sex_abs_lr_top20_features.png
          sex_lr_cm_comparison.png
          sex_lr_roc_comparison.png
          sex_rel_lr_cm.png
          sex_rel_lr_shap.png
          sex_rel_lr_top20_features.png
          sex_rel_svm_cm.png
          sex_rf_classifier_final.py
          sex_svm_cm_comparison.png
          sex_svm_roc_comparison.png
        sex-random-forest-classifier/
          balanced_sex_abs_rf_ageless_cm.png
          balanced_sex_abs_rf_ageless_miss_cm.png
          balanced_sex_abs_rf_cm.png
          balanced_sex_rel_rf_ageless_cm.png
          balanced_sex_rel_rf_ageless_miss_cm.png
          balanced_sex_rel_rf_cm.png
          sex_abs_rf_shap.png
          sex_abs_rf_top20_features.png
          sex_rel_rf_shap.png
          sex_rel_rf_top20_features.png
          sex_rf_cm_comparison.png
          sex_rf_roc_comparison.png
          unbalanced_sex_abs_rf_cm.png
          unbalanced_sex_rel_rf_cm.png
        model_comparisons.csv
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
          age_numeric_predictor_final.ipynb
          age_numeric_predictor_final.py
          age_rf_classifier.ipynb
          age_rf_classifier.py
          age_rf_regressor.ipynb
          age_rf_regressor.py
        bmi-prediction/
          bmi_HGBR_rf_regressor.ipynb
        sex-rediction/
          sex_lr_svm_final.ipynb
          sex_lr_svm_final.py
          sex_rf_classifier_final.ipynb
          sex_rf_classifier_final.py
qiime2-moshpit-macos-latest-conda.yml
qiime2-moshpit-ubuntu-latest-conda.yml
README.md
```


