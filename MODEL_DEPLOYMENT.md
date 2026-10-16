# Model Deployment Architecture & Implementation

This document details the technical implementation, inference pipeline, serialization standards, and decision logic used to deploy the trained Heart Disease Machine Learning model into production.

---

## 1. Machine Learning Pipeline Architecture

The deployed model is an end-to-end composite `Pipeline` object from `scikit-learn`.

```
Raw Patient Data (25 Features)
              │
              ▼
  ┌────────────────────────────────────────────────────────┐
  │         ColumnTransformer ('preprocessor')             │
  │  ┌─────────────────────────┐┌────────────────────────┐ │
  │  │ StandardScaler()        ││ OneHotEncoder()        │ │
  │  │ (19 Numerical Features) ││ (6 Categorical Features)│
  │  └─────────────────────────┘└────────────────────────┘ │
  └───────────────────────────┬────────────────────────────┘
                              │ Scaled & One-Hot Encoded Vector
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │       RandomForestClassifier ('classifier')            │
  │  - n_estimators = 100                                  │
  │  - random_state = 42                                   │
  │  - max_depth = None                                    │
  │  - criterion = 'gini'                                  │
  └───────────────────────────┬────────────────────────────┘
                              │
                              ▼
          Prediction (Class 0 or 1) & Probability (%)
```

### Model Serialization
- **Serialized Artifact:** `heart_disease_model.pkl`
- **File Size:** ~11.5 MB
- **Serializer:** `joblib` (compatible with `scikit-learn >= 1.3.0`)
- **Key Advantage:** By saving the entire `Pipeline` (and not just the raw classifier), all feature scalers, means, variances, and one-hot category dictionaries are encapsulated together. Raw patient inputs can be passed directly to `model.predict()` without manual data preparation.

---

## 2. Serving & Web Application Design (`app.py`)

The application is structured into three specialized modules:

```
app.py
├── Tab 1: 🩺 Risk Predictor (Interactive Clinical Assessment)
│   ├── Preset Selector (Healthy, Borderline, High Risk)
│   ├── 4-Section Clinical Input Form
│   ├── Real-time Inference Engine
│   └── Rule-based Risk Factor Breakdown
│
├── Tab 2: 📊 Model & Feature Insights
│   ├── Performance Metric Cards (88.17% Accuracy)
│   ├── Feature Importance Bar Chart (Random Forest weights)
│   └── Classification Matrix Table
│
└── Tab 3: 📖 Clinical Guidance & Info
    ├── AHA/ACC Cardiovascular Benchmarks
    └── Academic & Medical Disclaimer
```

---

## 3. High-Performance Model Caching

To guarantee low latency and prevent redundant disk I/O on every user interaction, the pipeline is cached in memory:

```python
import os, joblib, streamlit as st

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "heart_disease_model.pkl")
    return joblib.load(model_path)

model = load_model()
```

---

## 4. Inference & Probability Calculation

When the user submits the assessment form:

1. **DataFrame Construction:** The 25 patient parameters are assembled into a single-row `pandas.DataFrame` maintaining exact feature column names and original data types:
   ```python
   input_data = pd.DataFrame([{
       "age": age, "sex": sex, "resting_bp_systolic": resting_bp_systolic,
       "resting_bp_diastolic": resting_bp_diastolic, "cholesterol_total": cholesterol_total,
       "hdl": hdl, "ldl": ldl, "triglycerides": triglycerides,
       "fasting_blood_sugar": fasting_blood_sugar, "hba1c": hba1c, "bmi": bmi,
       "resting_heart_rate": resting_heart_rate, "max_heart_rate_achieved": max_heart_rate_achieved,
       "chest_pain_type": chest_pain_type, "exercise_induced_angina": exercise_induced_angina,
       "st_depression": st_depression, "family_history": family_history,
       "smoker_status": smoker_status, "alcohol_units_per_week": alcohol_units_per_week,
       "exercise_minutes_per_week": exercise_minutes_per_week, "sleep_hours": sleep_hours,
       "stress_score": stress_score, "wearable_owner": wearable_owner,
       "daily_steps": daily_steps, "diet_quality_score": diet_quality_score
   }])
   ```

2. **Inference Execution:**
   ```python
   prediction = model.predict(input_data)[0]              # 0 or 1
   probabilities = model.predict_proba(input_data)[0]     # [prob_no_disease, prob_disease]
   disease_prob = probabilities[1] * 100.0                # Percentage (0.0% to 100.0%)
   ```

---

## 5. Risk Categorization & Clinical Expert Logic

Rather than presenting raw binary classifications alone, the deployment system classifies risk into three actionable tiers and highlights specific abnormal clinical markers:

| Risk Tier | Condition | Clinical Interpretation | Action Recommended |
|-----------|-----------|-------------------------|--------------------|
| **Low Risk** | `disease_prob < 30%` | Negative classification with high confidence | Maintain routine healthy habits |
| **Moderate / Borderline** | `30% <= disease_prob < 50%` | Elevated lifestyle or borderline physiological markers | Proactively modify lifestyle factors (exercise, diet, BP monitoring) |
| **Elevated / High Risk** | `disease_prob >= 50%` | Positive classification (Class 1) | Prompt clinical consultation and diagnostic follow-up |

### Rule-Based Factor Flags Generated:
- **Hypertension Warning:** Triggered if Systolic $\ge 130$ mmHg or Diastolic $\ge 85$ mmHg.
- **Atherogenic LDL Warning:** Triggered if LDL $\ge 130$ mg/dL.
- **Low Protective HDL:** Triggered if HDL $< 40$ mg/dL (Male) or $< 50$ mg/dL (Female).
- **Chronotropic Limitation:** Triggered if Max HR Achieved $< 130$ bpm (the #1 predictive feature).
- **Ischemia Indication:** Triggered if ST Depression $\ge 1.5$ mm or Exercise-Induced Angina is `True`.
- **Obesity Workload:** Triggered if BMI $\ge 30.0$ $kg/m^2$.
- **Sedentary Lifestyle:** Triggered if physical exercise $< 90$ minutes/week.

---

## 6. How to Run Model Serving Locally

```bash
# Activate your Python environment
python -m pip install -r requirements.txt

# Run the deployment server
streamlit run app.py
```
Server opens at `http://localhost:8501`.
