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
conda activate qiime2-amplicon-2025.7
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
src/
    absolute-abundance-modeling/
        linear_models.py
        nonlinear_models.py
    differential-abundance/
        BIRDMAn/
            BIRDMAn_metaG_pergenome.py
    metadata-variable-prediction/
        age-prediction/
          age_rf_classifier.ipynb
          age_rf_classifier.py
          age_rf_regressor.ipynb
          age_rf_regressor.py
        bmi-prediction/
          bmi_HGBR_rf_regressor.ipynb
data_preprocessing/
    traintest_split.py
out/
    model_summary.xlsx
    figs/  
qiime2-moshpit-macos-latest-conda.yml
qiime2-moshpit-ubuntu-latest-conda.yml
README.md
```

