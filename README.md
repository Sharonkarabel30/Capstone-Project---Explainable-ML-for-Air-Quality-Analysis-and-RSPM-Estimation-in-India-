# Predicting RSPM (PM10) Air Pollution Across India

This repository contains my capstone project for AAI-590, Master of Science in Applied Artificial Intelligence. In this project I predict the daily concentration of respirable suspended particulate matter (RSPM, PM10) at air-quality monitoring stations in India, using the India Air Quality Data from Kaggle (Central Pollution Control Board).

Below I have listed every file in this repository and what each one contains.

# Dashboard Link: https://sharonkarabel30.github.io/Capstone-Project---Explainable-ML-for-Air-Quality-Analysis-and-RSPM-Estimation-in-India-/Air_pollution_Dashboard.html
## Files in this repository

| File | What it contains |
|---|---|
| `README.md` | This file. It explains what is in my repository. |
| `LICENSE` | The license for this project. |
| `SharonKarabel_Capstone final.ipynb` | My complete project notebook. It covers data cleaning, exploratory data analysis, model and pipeline design, model training, model optimization, and model analysis and discussion, followed by the final test evaluation. I ran it on Kaggle. |
| `Air_pollution_Dashboard.html` | An interactive web page where I present my data, my model results and live predictions from my two final models. |
| `preprocessor.joblib` | My fitted preprocessing pipeline (filling missing values, scaling and one-hot encoding). |
| `xgboost_rspm.json` | My tuned XGBoost model. |
| `embedding_nn_rspm.keras` | My tuned entity-embedding neural network, saved in Keras format. |
| `nn_weights.npz` | The weights of the same neural network, saved as NumPy arrays. |
| `nn_metadata.json` | The settings my neural network needs to prepare its inputs: category IDs, scaling values and limits. |
| `test_results.csv` | The final results of all my models on the 2015 test data. |
| `validation_results.csv` | The results of all my models on the 2013–2014 validation data. |
| `tuning_ridge.csv` | My hyperparameter tuning results for Ridge Regression. |
| `tuning_xgboost.csv` | My hyperparameter tuning results for XGBoost. |
| `tuning_neural_network.csv` | My hyperparameter tuning results for the neural network. |
| `shap_importance.csv` | The most important features of my XGBoost model, measured with SHAP. |

## Data source

The original dataset is not stored in this repository. My notebook reads it directly from Kaggle: *India Air Quality Data* (`shrutibhargava94/india-air-quality-data`).
