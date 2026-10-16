# Heart Disease Risk Prediction using Machine Learning

## Project Overview

This project predicts whether a patient has heart disease using machine learning techniques.

The dataset contains 9,000 patient records with 27 columns.

## 📑 Project Deliverables & Technical Documentation

For complete evaluation and submission details, refer to the dedicated documentation files:

1. **[DATA SET ASSIGNED](DATA_SET_ASSIGNED.md)**: Dataset specifications, complete 25-feature data dictionary, clinical descriptions, distributions, and preprocessing steps.
2. **[Deployment Platform](DEPLOYMENT_PLATFORM.md)**: Cloud platform architecture, Streamlit Community Cloud & Hugging Face hosting, container specifications, and continuous deployment pipeline.
3. **[Model Deployment](MODEL_DEPLOYMENT.md)**: End-to-end pipeline architecture, model caching, real-time inference engine, clinical risk categorization logic, and rule-based factor flags.
4. **[Prediction & Testing](PREDICTION_AND_TESTING.md)**: Comprehensive evaluation metrics (88.17% accuracy), confusion matrix, feature importance rankings, automated unit test suite (`test_app_logic.py`), and test case results.
5. **[Performance Evaluation Metrics](PERFORMANCE_EVALUATION_METRICS.md)**: Mathematical formulations and clinical interpretations of Accuracy, Precision (86.40%), Recall (72.29%), Specificity (95.06%), F1-Score (78.72%), and ROC-AUC (0.9362).
6. **[Model Comparison](MODEL_COMPARISON.md)**: Empirical benchmarking across 7 classifiers (Logistic Regression, Gradient Boosting, Random Forest, AdaBoost, Naive Bayes, KNN, Decision Tree) and selection rationale.
7. **[Hyperparameter Tuning](HYPERPARAMETER_TUNING.md)**: Stratified Grid Search CV optimization for Random Forest, sensitivity analysis of trees, depth, splits, and generalization validation.



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

## Web Application & Deployment

The trained model is deployed as an interactive web application built with **Streamlit**.

### 🌟 Web App Features
- **Interactive Risk Prediction:** Enter 25 patient parameters (vitals, blood chemistry, ECG, and lifestyle) to compute cardiac risk probability.
- **Quick Preset Archetypes:** Pre-loaded samples (Healthy Adult, Borderline Risk, High Risk) for instant demonstration.
- **Personalized Factor Analysis:** Highlights elevated risk markers (Blood Pressure, LDL, ST Depression, low physical activity).
- **Feature Importance & Analytics:** Visual breakdown of the most critical predictors identified by Random Forest.

### 🚀 Running the App Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Launch web app
streamlit run app.py
```
Access the application in your browser at `http://localhost:8501`.

### 🌐 Cloud Deployment (Live Link)
See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for 1-click free deployment to **Streamlit Community Cloud** (live link: `https://<your-username>-heart-disease-ml.streamlit.app`) and **Hugging Face Spaces**.

## Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook

## Disclaimer

This project is for educational and research purposes only and should not be used as a medical diagnosis tool.