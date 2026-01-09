# UAS Feature Extraction and vegetation of different post fire ages Classification

## Overview

This project develops a comprehensive framework for **feature extraction and machine learning classification** using **Unmanned Aerial System (UAS)** multispectral imagery (MSI). The system extracts spectral indices, texture features, and other derived metrics from drone-based hyperspectral/multispectral data to classify and classify vegetation of different post fire ages, particularly for fynbos ecosystems in South Africa. It also allow you to measure alpha diversity using a similiarity metrics.

## Key Features

- **Spectral Indices Extraction**: Calculate vegetation indices (NDVI, SAVI, EVI, etc.) from multispectral imagery
- **Texture Analysis**: Extract GLCM (Gray Level Co-occurrence Matrix) texture features
- **Feature Selection**: Advanced feature selection techniques to identify the most discriminative features
- **Classification Pipeline**: Machine learning models (XGBoost, Random Forest, etc.) for classification
- **Variogram Analysis**: Spatial autocorrelation analysis and kriging
- **Performance Evaluation**: Comprehensive metrics including radar charts and confusion matrices
- **Geospatial Processing**: Integration with shapefile data and raster processing

## Project Structure

### Core Processing Scripts

- **`Indicies_fynbos.ipynb` / `Indicies_fynbos_v2.ipynb`**: Extract spectral vegetation indices from multispectral imagery
- **`Texture_fynbos.ipynb` / `Texture_fynbos_0.py`**: Compute texture features using GLCM and fast texture analysis
- **`texture.py`**: Texture analysis wrapper class for GLCM calculations across multiple channels
- **`alpha_estimator.py` / `alpha_estimator v2.py`**: Alpha parameter estimation measuring biodiversity
- **`asd_corrected_spectra_main.py`**: ASD spectra correction pipeline

### Classification and Feature Selection

- **`Classifier_feature_selection.ipynb` / `Classifier_feature_selection.py`**: Feature selection and preprocessing for classification
- **`Classifier_feature_selection_v2.ipynb`**: Advanced feature selection (v2)
- **`classifier.py`**: Main classification model implementation
- **`for_journal_results.py` / `for_journal_results_pc.py`**: Final results compilation and publication-ready outputs

### Analysis and Visualization

- **`Variogram_VI.ipynb` / `Variogram_bandwise.ipynb`**: Variogram analysis for spatial structure assessment
- **`stats.py`**: Statistical analysis utilities
- **`Texture_visualization.py`**: Texture feature visualization
- **`performance_metrics_radar_chart.html`**: Interactive performance metrics visualization

### Testing and Experimental Scripts

- **`fynbos_testing_manisha.ipynb` / `fynbos_testing_manisha1.ipynb`**: Experimental testing notebooks
- **`fynbos_exp.py`**: Additional fynbos experiments
- **`fynbos_variogram.py`**: Variogram experiments for fynbos data

### Configuration

- **`ag_env.yml`**: Conda environment file with all dependencies

## Dependencies

The project uses a Conda environment defined in `ag_env.yml`. Key packages include:

- **Data Processing**: NumPy, Pandas, SciPy
- **Geospatial**: GeoPandas, Rasterio, Rio-xarray, Xarray, GDAL, Shapely, Fiona
- **Image Processing**: OpenCV, scikit-image, Spectral
- **Machine Learning**: scikit-learn, XGBoost
- **Optimization**: Optuna
- **Visualization**: Matplotlib, Seaborn
- **Jupyter**: For interactive notebooks

### Installation

```bash
conda env create -f ag_env.yml
conda activate ag_env
```

## Workflow

### 1. Feature Extraction
- Extract spectral indices from multispectral imagery
- Calculate texture features using GLCM
- Generate additional derived metrics

### 2. Data Integration
- Combine indices and texture features into comprehensive feature matrices
- Handle geospatial coordinates and shapefile data

### 3. Feature Selection
- Apply feature selection techniques to reduce dimensionality
- Identify most discriminative features for classification

### 4. Model Training and Optimization
- Train machine learning classifiers (XGBoost, Random Forest, etc.)
- Use Optuna for hyperparameter optimization
- Validate model performance with cross-validation

### 5. Evaluation and Visualization
- Generate performance metrics (accuracy, precision, recall, F1-score)
- Create radar charts and confusion matrices
- Perform variogram analysis for spatial patterns

## Data Format

The project primarily works with:
- **CSV files**: Feature matrices (indices, textures)
- **Shapefiles**: Geospatial vector data
- **Raster files**: Drone imagery in GeoTIFF or similar formats
- **Multispectral data**: Band data from UAS sensors

## Output

The project generates:
- Classification maps
- Performance metrics and radar charts
- Feature importance rankings
- Variogram plots
- Statistical summaries

## Research Focus

This project focuses on **fynbos ecosystem classification and analysis** in South Africa, utilizing 6+ years of drone survey data across multiple burn plots:
- Burn plots from 2006, 2016, 2017, 2019, 2020, and 2022
- Multi-temporal analysis of vegetation recovery and classification

## Usage

Most analysis is conducted through Jupyter notebooks. To run a specific analysis:

```bash
jupyter notebook Indicies_fynbos.ipynb
```

Or execute Python scripts directly:

```bash
python classifier.py
```

## Performance Visualization
![performance_metrics_radar_chart](https://github.com/user-attachments/assets/fa29f4eb-138f-4c30-b166-0926da5ced06)

