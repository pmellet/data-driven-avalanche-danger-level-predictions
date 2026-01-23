## Project Overview

This project was carried out as part of the **final capstone project** of the **Applied Data Science: Machine Learning Program** at the **EPFL Extension School**.

The objective of this project is to train and compare several Machine Learning models using meteorological data in order to **predict the avalanche danger level in Switzerland**.

---

## Repository Structure

The repository is organized as follows:

### `environments/`
Contains the two conda environments used throughout the project:
- `adsml.yml`: main environment used for data analysis and model training
- `warnreg.yml`: environment dedicated to the creation of geographical maps using **GeoPandas**

### `notebooks/`
All Jupyter notebooks developed for this project.

### `pictures_maps/`
Images and maps that are used and reused within the notebooks.

### `datasets/`
All datasets used in the project, including:
- raw data
- training datasets
- test datasets

### `codes/`
Contains additional code and supporting notebooks:
- `functions.py`: centralizes all functions imported into the notebooks, improving code readability and clarity
- `warnreg.ipynb`: notebook used to construct the geographical maps

### `scores & pca components/`
Stores NumPy files containing:
- the performance scores (accuracies) of the trained models
- the number of PCA components selected for dimensionality reduction
``

