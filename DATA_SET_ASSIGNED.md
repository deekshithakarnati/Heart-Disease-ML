# DATA SET ASSIGNED: Heart Disease Risk Dataset (2026)

This document provides comprehensive technical documentation for the dataset assigned and utilized for training the Heart Disease Risk Prediction machine learning model.

---

## 1. Dataset Overview

- **Dataset File:** `heart_disease_risk_2026.csv`
- **Total Records:** 9,000 patients
- **Total Attributes:** 27 columns (1 identifier column, 25 feature columns, 1 target column)
- **Target Variable:** `has_heart_disease` (Binary Classification)
  - `0`: No Heart Disease (Negative class)
  - `1`: Heart Disease Present (Positive class)
- **Class Distribution:**
  - Class `0` (No Heart Disease): **6,273 records (69.7%)**
  - Class `1` (Heart Disease): **2,727 records (30.3%)**
- **Data Integrity:**
  - Missing Values: **0 (Zero null/NaN values across all columns)**
  - Duplicate Records: **0 (Zero duplicate rows)**

---

## 2. Feature Dictionary & Data Specifications

The dataset combines clinical physiological measurements, laboratory blood chemistry, electrocardiographic findings, and comprehensive patient lifestyle metrics.

### A. Dropped Column (Non-Predictive)
| Column Name | Data Type | Description | Action |
|-------------|-----------|-------------|--------|
| `patient_id` | Integer | Unique identifier for each patient record | Dropped prior to modeling |

---

### B. Categorical Features (6 Columns)

| Feature Name | Data Type | Permitted Values | Clinical Description |
|--------------|-----------|------------------|----------------------|
| `sex` | String | `Male`, `Female` | Biological sex of the patient |
| `chest_pain_type` | String | `Asymptomatic`, `Non-Anginal Pain`, `Atypical Angina`, `Typical Angina` | Classification of chest discomfort or pain symptoms |
| `smoker_status` | String | `Never`, `Former`, `Current` | Smoking history and tobacco exposure |
| `exercise_induced_angina` | Boolean | `True`, `False` | Angina symptoms triggered during physical exertion |
| `family_history` | Boolean | `True`, `False` | Documented family history of premature cardiovascular disease |
| `wearable_owner` | Boolean | `True`, `False` | Whether the patient regularly uses a health-monitoring wearable |

---

### C. Numerical Features (19 Columns)

| Feature Name | Data Type | Min | Mean | Max | Clinical Significance |
|--------------|-----------|-----|------|-----|-----------------------|
| `age` | Integer | 18 | 53.98 | 90 | Patient age in years |
| `resting_bp_systolic` | Integer | 85 | 127.85 | 181 | Systolic blood pressure (mmHg) at rest |
| `resting_bp_diastolic` | Integer | 50 | 80.79 | 126 | Diastolic blood pressure (mmHg) at rest |
| `cholesterol_total` | Integer | 90 | 188.75 | 314 | Total serum cholesterol (mg/dL) |
| `hdl` | Integer | 18 | 55.22 | 110 | High-density lipoprotein / "good" cholesterol (mg/dL) |
| `ldl` | Integer | 35 | 103.28 | 207 | Low-density lipoprotein / "bad" cholesterol (mg/dL) |
| `triglycerides` | Integer | 35 | 152.05 | 390 | Blood triglyceride levels (mg/dL) |
| `fasting_blood_sugar` | Integer | 60 | 119.47 | 204 | Fasting blood glucose level (mg/dL) |
| `hba1c` | Float | 4.0 | 5.79 | 8.6 | Glycated hemoglobin percentage (%) |
| `bmi` | Float | 15.0 | 25.26 | 43.3 | Body Mass Index ($kg/m^2$) |
| `resting_heart_rate` | Integer | 48 | 80.91 | 111 | Resting pulse rate (beats per minute) |
| `max_heart_rate_achieved` | Integer | 93 | 164.84 | 210 | Maximum heart rate achieved during peak exercise (bpm) |
| `st_depression` | Float | 0.0 | 1.01 | 6.5 | ST segment depression induced by exercise relative to rest |
| `alcohol_units_per_week` | Float | 0.0 | 5.71 | 45.9 | Self-reported weekly alcohol consumption in units |
| `exercise_minutes_per_week` | Integer | 0 | 139.49 | 366 | Weekly moderate-to-vigorous physical exercise duration |
| `sleep_hours` | Float | 3.1 | 6.99 | 11.0 | Average nightly sleep duration (hours) |
| `stress_score` | Float | 0.0 | 47.97 | 100.0 | Standardized psychological stress assessment index (0–100) |
| `daily_steps` | Integer | 500 | 6,163.66 | 13,950 | Mean daily physical step count |
| `diet_quality_score` | Float | 4.8 | 59.47 | 100.0 | Nutritional balance score based on dietary guidelines (0–100) |

---

## 3. Data Preprocessing & Transformation Pipeline

All transformations are integrated within an end-to-end `scikit-learn` `ColumnTransformer`:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)
```

1. **Feature Separation:** The target column `has_heart_disease` was split from the 25 input predictors.
2. **Train-Test Stratified Split:** 
   - Training Set: **7,200 instances (80%)**
   - Testing Set: **1,800 instances (20%)**
   - Stratification applied to preserve the 69.7% / 30.3% target ratio.
3. **Standardization:** All 19 numerical features are zero-mean, unit-variance standardized via `StandardScaler()`.
4. **Encoding:** All 6 categorical features are converted to binary one-hot dummy variables via `OneHotEncoder(handle_unknown="ignore")`.
