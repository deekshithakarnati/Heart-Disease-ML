# Heart Disease Risk Prediction using Machine Learning

## Project Overview

This project predicts whether a patient has heart disease using machine learning techniques.

The dataset contains 9,000 patient records with 27 columns.

## Dataset

The dataset contains patient health and lifestyle information such as:

- Age
- Sex
- Blood pressure
- Cholesterol
- HDL and LDL
- Triglycerides
- HbA1c
- BMI
- Heart rate
- Chest pain type
- Exercise-induced angina
- Family history
- Smoking status
- Exercise
- Sleep
- Stress
- Daily steps
- Diet quality

## Machine Learning Workflow

1. Data Loading
2. Data Preprocessing
3. Missing Value Checking
4. Duplicate Checking
5. Feature and Target Separation
6. Train-Test Split
7. Numerical Feature Scaling
8. Categorical Feature Encoding
9. Random Forest Model Building
10. Model Training
11. Model Evaluation
12. Feature Importance Analysis

## Model

Random Forest Classifier

- Number of trees: 100
- Test size: 20%
- Random state: 42

## Results

Accuracy: **88.17%**

### Classification Report

| Class | Precision | Recall | F1-Score |
|------|-----------|--------|----------|
| No Heart Disease | 0.89 | 0.95 | 0.92 |
| Heart Disease | 0.86 | 0.72 | 0.79 |

## Important Features

The most important feature identified by the model was:

**Maximum Heart Rate Achieved**

Other important features include:

- ST Depression
- Exercise-Induced Angina
- LDL
- Age
- HDL
- Exercise Minutes per Week
- BMI

## Files

- `Heart_Disease_ML.ipynb` - Complete Jupyter Notebook
- `heart_disease_risk_2026.csv` - Dataset
- `heart_disease_model.pkl` - Trained Random Forest model

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Disclaimer

This project is for educational and research purposes only and should not be used as a medical diagnosis tool.