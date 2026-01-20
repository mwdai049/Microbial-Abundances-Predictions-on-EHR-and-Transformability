# Microbial Abundances Predictions on EHR and Transformability
## Credit
This project was a collaboration between Monica Dai, Katelyn Zhao, Camille Sicat, Nathan Wang, and Sophie Wang at UC San Diego under the Halıcıoğlu Data Science Institute. Our work would not have been possible without the mentorship of Dr. Rob Knight and the assistance of the Knight Lab. 

## Setup
### QIIME 2
To upload and process the data, you must install QIIME 2. We used the amplicon distribution, and the yml files are in the project directory.

### Conda
#### Linux/Windows WSL

```
conda env create \
  --name qiime2-amplicon-2025.10 \
  --file qiime2-environment-windows.yml
```

#### macOS (Apple Silicon)

```
CONDA_SUBDIR=osx-64 conda env create \
  --name qiime2-amplicon-2025.10 \
  --file qiime2-environment-macos.yml
conda activate qiime2-amplicon-2025.10
conda config --env --set subdir osx-64
```

#### macOS (Intel)

```
conda env create \
  --name qiime2-amplicon-2025.10 \
  --file qiime2-environment-macos.yml
```

To test the installation:

```
conda deactivate
conda activate qiime2-amplicon-2025.10
qiime info
```
