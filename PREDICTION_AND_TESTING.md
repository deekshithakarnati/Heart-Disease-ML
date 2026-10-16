# Prediction & Testing Report

This document presents the complete evaluation results, test procedures, test cases, and verification records for the Heart Disease Risk Prediction model and deployed web application.

---

## 1. Model Evaluation Metrics (Test Set: 1,800 Patients)

The Random Forest model was evaluated on a 20% holdout test partition containing 1,800 unseen patient records.

### A. Overall Accuracy
$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{Total Samples}} = \frac{394 + 1193}{1800} = \mathbf{88.17\%}$$

---

### B. Detailed Classification Report

| Class | Precision | Recall | F1-Score | Support (Sample Size) |
|-------|-----------|--------|----------|-----------------------|
| **Class 0 (No Heart Disease)** | **0.89** | **0.95** | **0.92** | 1,255 patients |
| **Class 1 (Heart Disease)** | **0.86** | **0.72** | **0.79** | 545 patients |
| **Macro Average** | 0.88 | 0.84 | 0.85 | 1,800 patients |
| **Weighted Average** | 0.88 | 0.88 | 0.88 | 1,800 patients |

- **High Negative Specificity (95% Recall on Class 0):** Accurately screens out non-cardiac cases, minimizing unnecessary patient panic and medical testing.
- **High Positive Precision (86% Precision on Class 1):** When the model flags a positive heart disease risk, there is an 86% probability that genuine underlying cardiac pathology exists.

---

### C. Confusion Matrix Breakdown

$$\begin{pmatrix}
\text{True Negatives (TN) = 1,193} & \text{False Positives (FP) = 62} \\
\text{False Negatives (FN) = 151} & \text{True Positives (TP) = 394}
\end{pmatrix}$$

- **True Negatives (TN):** 1,193 patients correctly identified as having no heart disease.
- **False Positives (FP):** 62 patients incorrectly flagged (low false alarm rate of 4.9%).
- **False Negatives (FN):** 151 patients missed by conservative classification threshold.
- **True Positives (TP):** 394 patients correctly identified with heart disease.

---

## 2. Top Feature Importance Rankings

Analysis of Gini impurity reduction across all 100 decision trees identified the following primary predictors:

| Rank | Feature Name | Importance Score | Clinical Context |
|------|--------------|------------------|------------------|
| 1 | `max_heart_rate_achieved` | **0.2322** (23.2%) | Inability to reach age-predicted target HR during exercise |
| 2 | `st_depression` | **0.0729** (7.3%) | Electrocardiographic indicator of myocardial ischemia |
| 3 | `exercise_induced_angina_False` | **0.0694** (6.9%) | Absence of angina during exertion strongly predicts health |
| 4 | `exercise_induced_angina_True` | **0.0670** (6.7%) | Provocable exertional chest discomfort |
| 5 | `ldl` | **0.0426** (4.3%) | Circulating atherogenic cholesterol |
| 6 | `age` | **0.0422** (4.2%) | Cumulative cardiovascular risk with advancing age |
| 7 | `hdl` | **0.0390** (3.9%) | Protective reverse cholesterol transport capacity |
| 8 | `exercise_minutes_per_week` | **0.0334** (3.3%) | Habitual physical activity level |
| 9 | `bmi` | **0.0326** (3.3%) | Adiposity and metabolic workload |
| 10 | `resting_bp_systolic` | **0.0322** (3.2%) | Vascular stiffness and arterial pressure |

---

## 3. Automated Unit Testing Suite (`test_app_logic.py`)

An automated test script was created to validate the deployed pipeline against standard archetypes.

### Test Script Implementation:
```python
import joblib, os, pandas as pd

def run_tests():
    model = joblib.load("heart_disease_model.pkl")
    # Validates loading, input schema conformance, prediction outputs, and probability bounds
```

### Automated Test Cases & Results:

#### Test Case 1: Healthy Profile Archetype
- **Patient Profile:** 35-year-old female, non-smoker, active (210 mins exercise/week, 9,500 daily steps), BP 115/75, LDL 85 mg/dL, HDL 65 mg/dL, Max HR 180 bpm, no angina, ST depression 0.2 mm.
- **Model Output:** 
  - Predicted Class: `0` (No Heart Disease)
  - Probability Vector: `[1.00, 0.00]` (100% Healthy, 0% Disease)
- **Status:** **PASS**

---

#### Test Case 2: High Risk Profile Archetype
- **Patient Profile:** 67-year-old male, active smoker, sedentary (20 mins exercise/week, 2,800 steps), BP 158/96, LDL 170 mg/dL, HDL 34 mg/dL, Max HR 118 bpm, exercise-induced angina = `True`, ST depression 2.8 mm, family history = `True`.
- **Model Output:**
  - Predicted Class: `1` (Heart Disease Risk)
  - Probability Vector: `[0.03, 0.97]` (3% Healthy, 97% Disease)
- **Status:** **PASS**

---

#### Test Case 3: Borderline / Moderate Risk Archetype
- **Patient Profile:** 55-year-old male, former smoker, moderate activity (75 mins exercise/week), BP 138/88, LDL 135 mg/dL, HDL 45 mg/dL, Max HR 145 bpm, ST depression 1.2 mm.
- **Model Output:**
  - Predicted Class: `0` (Conservative Class 0)
  - Probability Vector: `[0.65, 0.35]` (65% Healthy, 35% Moderate Risk)
- **Status:** **PASS** (Correctly categorized as Moderate Risk in web application)

---

## 4. Web Application Health & Network Verification

| Verification Step | Command / Endpoint | Expected Outcome | Actual Result |
|-------------------|--------------------|------------------|---------------|
| Pipeline Unit Tests | `python test_app_logic.py` | Exit code 0, all tests pass | **Passed (Exit code 0)** |
| Web Server Boot | `streamlit run app.py` | Listening on port 8501 | **Running (Port 8501)** |
| HTTP Status Ping | `GET http://localhost:8501` | HTTP 200 OK | **HTTP 200 OK** |
| Preset Switching | Web UI selector | Form fields reload dynamically | **Verified** |
| Feature Importance Plot | Web UI analytics tab | Matplotlib chart renders | **Verified** |

---

## 5. How to Re-Run Testing Locally

To re-run the automated verification suite at any time:

```bash
# Run the test suite
python test_app_logic.py
```

Expected output:
```text
Loading model...
Model loaded successfully: <class 'sklearn.pipeline.Pipeline'>
[Healthy Profile] -> Prediction: 0, Probability: [1. 0.]
[High Risk Profile] -> Prediction: 1, Probability: [0.03 0.97]

All unit tests PASSED successfully!
```
