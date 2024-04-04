# Using machine learning to model the distribution of Vegetation Types in Norway
Project on using supervised classification to predict the distribution of Norwegian
Vegetation Types from environmental background variables. 
The code heavily relies on the Google Earth Engine Python API (https://developers.google.com/earth-engine/api_docs) 
and numerous supporting Python packages (see `environment.yml`).

### Installation suggestion to run the project code locally

Install [Anaconda/Miniconda](https://www.anaconda.com/products/distribution) and Git. In a
terminal where the `conda` and `git` commands are available (e.g., Anaconda Prompt), run:

```
cd [path/to/download/target/directory]
git clone https://github.com/lasseke/vt-land-cover-classification.git
cd ./vt-land-cover-classification
conda env create -f environment.yml
conda activate dmvtnor-env
```

Subsequently, navigate to the "notebooks" directory for the analysis scripts used in Keetz et al. (in prep.).
The notebooks may need to be executed sequentially to reproduce results (ascending file numbering).

### Project structure overview

| Directory | File(s) |  Summary |
|----------|-------------|:------:|
| `data/dict/` | `colors.json` | Defines colors shared across different plots. |
|              | `predictors.json` | Defines metadata (long names, etc.) for the predictor variables. |
|              | `spectral_indices.json` | Defines long names and band calculation formulas for the spectral indices. |
|              | `vt_classes.json` | Defines metadata (long names, ecosystem group, etc.) for the Vegetation Type classes. |
| `data/misc/` | `vtdata_5f_spatial_cv_indices.pkl` | Stores 10-fold leave-location-out cross-validation indices for VT feature matrix entries. |
| `data/interim/`   | * | Stores interim outputs of processed datasets. <span style="color:red;">Not included.</span> |
| `data/processed/` | * | Stores final outputs of processed datasets. <span style="color:red;">Not included.</span> |
| `data/raw/`       | * | Stores original datasets. <span style="color:red;">Not included.</span> |
| `notebooks/` | `00-*.ipynb` | Notebook for minor data-preprocessing. |
|              | `01-*.ipynb` | Notebooks to retrieve and export data in required formats. |
|              | `02-*.ipynb` | Notebooks to preprocess the data for model fitting (clean, generate feature matrix, create shared spatial cross validation indices, etc.). |
|              | `03-*.ipynb` | Notebooks to calculate and visualize data statistics (predictor correlation, etc.). |
|              | `04-*.ipynb` | Notebooks for model experiments. |
|              | `A-*.ipynb` | Notebooks for additional experiments and figures that were not included in the paper. |
| `src/` | `*.py` | Python helper code. |
