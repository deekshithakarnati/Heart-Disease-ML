import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #E63946;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #457B9D;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8F9FA;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #E63946;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .risk-high {
        background-color: #FDE8E8;
        border: 2px solid #E02424;
        border-radius: 10px;
        padding: 1.5rem;
        color: #9B1C1C;
    }
    .risk-low {
        background-color: #DEF7EC;
        border: 2px solid #0E9F6E;
        border-radius: 10px;
        padding: 1.5rem;
        color: #03543F;
    }
    .risk-mod {
        background-color: #FEF08A;
        border: 2px solid #CA8A04;
        border-radius: 10px;
        padding: 1.5rem;
        color: #854D0E;
    }
</style>
""", unsafe_allow_html=True)

# Load trained model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "heart_disease_model.pkl")
    if not os.path.exists(model_path):
        model_path = "heart_disease_model.pkl"
    return joblib.load(model_path)

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)

# Preset patient profiles for quick testing
PRESETS = {
    "Custom / Manual Input": None,
    "Healthy Adult (Low Risk)": {
        "age": 35, "sex": "Female", "resting_bp_systolic": 115, "resting_bp_diastolic": 75,
        "cholesterol_total": 170, "hdl": 65, "ldl": 85, "triglycerides": 100,
        "fasting_blood_sugar": 90, "hba1c": 5.2, "bmi": 22.4, "resting_heart_rate": 68,
        "max_heart_rate_achieved": 180, "chest_pain_type": "Asymptomatic",
        "exercise_induced_angina": False, "st_depression": 0.2, "family_history": False,
        "smoker_status": "Never", "alcohol_units_per_week": 1.5,
        "exercise_minutes_per_week": 210, "sleep_hours": 7.5, "stress_score": 25.0,
        "wearable_owner": True, "daily_steps": 9500, "diet_quality_score": 85.0
    },
    "Borderline / Moderate Risk": {
        "age": 55, "sex": "Male", "resting_bp_systolic": 138, "resting_bp_diastolic": 88,
        "cholesterol_total": 210, "hdl": 45, "ldl": 135, "triglycerides": 175,
        "fasting_blood_sugar": 115, "hba1c": 5.9, "bmi": 27.8, "resting_heart_rate": 78,
        "max_heart_rate_achieved": 145, "chest_pain_type": "Atypical Angina",
        "exercise_induced_angina": False, "st_depression": 1.2, "family_history": True,
        "smoker_status": "Former", "alcohol_units_per_week": 6.0,
        "exercise_minutes_per_week": 75, "sleep_hours": 6.0, "stress_score": 55.0,
        "wearable_owner": False, "daily_steps": 5200, "diet_quality_score": 52.0
    },
    "High Risk Patient": {
        "age": 67, "sex": "Male", "resting_bp_systolic": 158, "resting_bp_diastolic": 96,
        "cholesterol_total": 260, "hdl": 34, "ldl": 170, "triglycerides": 260,
        "fasting_blood_sugar": 145, "hba1c": 7.2, "bmi": 32.5, "resting_heart_rate": 88,
        "max_heart_rate_achieved": 118, "chest_pain_type": "Typical Angina",
        "exercise_induced_angina": True, "st_depression": 2.8, "family_history": True,
        "smoker_status": "Current", "alcohol_units_per_week": 14.0,
        "exercise_minutes_per_week": 20, "sleep_hours": 5.0, "stress_score": 82.0,
        "wearable_owner": False, "daily_steps": 2800, "diet_quality_score": 35.0
    }
}

# Top Navigation Tabs
tab_predict, tab_analytics, tab_guidance = st.tabs([
    "🩺 Risk Predictor", 
    "📈 Model & Feature Insights", 
    "📖 Clinical Guide & Info"
])

# ----------------- TAB 1: PREDICTION DASHBOARD -----------------
with tab_predict:
    st.markdown('<div class="main-header">Heart Disease Risk Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-powered cardiovascular risk assessment using Random Forest (Trained on 9,000 Patient Records with 88.17% Accuracy)</div>', unsafe_allow_html=True)

    if not model_loaded:
        st.error(f"⚠️ Could not load model: {model_error}")
        st.stop()

    # Preset selection
    st.sidebar.header("🎯 Quick Preset Profiles")
    preset_choice = st.sidebar.selectbox(
        "Load sample patient archetype:",
        list(PRESETS.keys())
    )
    preset_data = PRESETS[preset_choice]

    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Model Information:**
    - **Algorithm:** Random Forest Classifier
    - **Dataset Size:** 9,000 records
    - **Features:** 25 clinical & lifestyle parameters
    - **Test Accuracy:** 88.17%
    - **ROC AUC / Recall:** Robust cardiac risk sensitivity
    """)

    # Helper function for setting default values
    def get_val(key, default):
        if preset_data and key in preset_data:
            return preset_data[key]
        return default

    with st.form("patient_data_form"):
        st.subheader("1. Patient Demographics & Vital Signs")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            age = st.number_input("Age (years)", min_value=18, max_value=100, value=int(get_val("age", 54)))
            sex = st.selectbox("Sex", options=["Male", "Female"], index=0 if get_val("sex", "Male") == "Male" else 1)
        with col2:
            resting_bp_systolic = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=220, value=int(get_val("resting_bp_systolic", 125)))
            resting_bp_diastolic = st.number_input("Diastolic BP (mmHg)", min_value=50, max_value=140, value=int(get_val("resting_bp_diastolic", 80)))
        with col3:
            resting_heart_rate = st.number_input("Resting HR (bpm)", min_value=40, max_value=140, value=int(get_val("resting_heart_rate", 75)))
            max_heart_rate_achieved = st.number_input("Max HR Achieved (bpm)", min_value=60, max_value=230, value=int(get_val("max_heart_rate_achieved", 160)))
        with col4:
            bmi = st.number_input("BMI (kg/m²)", min_value=14.0, max_value=55.0, value=float(get_val("bmi", 25.0)), step=0.1)
            wearable_owner = st.selectbox("Owns Health Wearable?", options=["Yes", "No"], index=0 if get_val("wearable_owner", True) else 1) == "Yes"

        st.markdown("---")
        st.subheader("2. Blood Chemistry & Lipid Profile")
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            cholesterol_total = st.number_input("Total Cholesterol (mg/dL)", min_value=80, max_value=400, value=int(get_val("cholesterol_total", 190)))
            triglycerides = st.number_input("Triglycerides (mg/dL)", min_value=30, max_value=500, value=int(get_val("triglycerides", 150)))
        with col6:
            ldl = st.number_input("LDL 'Bad' Cholesterol (mg/dL)", min_value=30, max_value=300, value=int(get_val("ldl", 100)))
            hdl = st.number_input("HDL 'Good' Cholesterol (mg/dL)", min_value=15, max_value=130, value=int(get_val("hdl", 55)))
        with col7:
            fasting_blood_sugar = st.number_input("Fasting Blood Sugar (mg/dL)", min_value=50, max_value=300, value=int(get_val("fasting_blood_sugar", 105)))
        with col8:
            hba1c = st.number_input("HbA1c (%)", min_value=4.0, max_value=14.0, value=float(get_val("hba1c", 5.6)), step=0.1)

        st.markdown("---")
        st.subheader("3. Cardiac Symptoms & ECG Findings")
        col9, col10, col11 = st.columns(3)
        with col9:
            chest_pain_type = st.selectbox(
                "Chest Pain Type",
                options=["Asymptomatic", "Non-Anginal Pain", "Atypical Angina", "Typical Angina"],
                index=["Asymptomatic", "Non-Anginal Pain", "Atypical Angina", "Typical Angina"].index(get_val("chest_pain_type", "Asymptomatic"))
            )
        with col10:
            exercise_induced_angina = st.selectbox(
                "Exercise-Induced Angina?",
                options=["No", "Yes"],
                index=1 if get_val("exercise_induced_angina", False) else 0
            ) == "Yes"
            family_history = st.selectbox(
                "Family History of Heart Disease?",
                options=["No", "Yes"],
                index=1 if get_val("family_history", False) else 0
            ) == "Yes"
        with col11:
            st_depression = st.number_input(
                "ST Depression (ECG)",
                min_value=0.0,
                max_value=7.0,
                value=float(get_val("st_depression", 0.8)),
                step=0.1,
                help="ST segment depression induced by exercise relative to rest"
            )

        st.markdown("---")
        st.subheader("4. Lifestyle, Physical Activity & Wellness")
        col12, col13, col14, col15 = st.columns(4)
        with col12:
            smoker_status = st.selectbox(
                "Smoking Status",
                options=["Never", "Former", "Current"],
                index=["Never", "Former", "Current"].index(get_val("smoker_status", "Never"))
            )
            alcohol_units_per_week = st.number_input("Alcohol Units / Week", min_value=0.0, max_value=60.0, value=float(get_val("alcohol_units_per_week", 3.0)), step=0.5)
        with col13:
            exercise_minutes_per_week = st.number_input("Exercise (mins / week)", min_value=0, max_value=600, value=int(get_val("exercise_minutes_per_week", 140)))
            daily_steps = st.number_input("Daily Steps", min_value=500, max_value=30000, value=int(get_val("daily_steps", 7000)))
        with col14:
            sleep_hours = st.number_input("Sleep (hours / night)", min_value=3.0, max_value=12.0, value=float(get_val("sleep_hours", 7.0)), step=0.5)
            stress_score = st.slider("Perceived Stress (0-100)", min_value=0.0, max_value=100.0, value=float(get_val("stress_score", 40.0)), step=1.0)
        with col15:
            diet_quality_score = st.slider("Diet Quality Score (0-100)", min_value=0.0, max_value=100.0, value=float(get_val("diet_quality_score", 65.0)), step=1.0)

        submitted = st.form_submit_button("🔍 Calculate Heart Disease Risk", use_container_width=True, type="primary")

    if submitted:
        # Prepare input DataFrame matching exact feature column order expected by pipeline
        input_data = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "resting_bp_systolic": resting_bp_systolic,
            "resting_bp_diastolic": resting_bp_diastolic,
            "cholesterol_total": cholesterol_total,
            "hdl": hdl,
            "ldl": ldl,
            "triglycerides": triglycerides,
            "fasting_blood_sugar": fasting_blood_sugar,
            "hba1c": hba1c,
            "bmi": bmi,
            "resting_heart_rate": resting_heart_rate,
            "max_heart_rate_achieved": max_heart_rate_achieved,
            "chest_pain_type": chest_pain_type,
            "exercise_induced_angina": exercise_induced_angina,
            "st_depression": st_depression,
            "family_history": family_history,
            "smoker_status": smoker_status,
            "alcohol_units_per_week": alcohol_units_per_week,
            "exercise_minutes_per_week": exercise_minutes_per_week,
            "sleep_hours": sleep_hours,
            "stress_score": stress_score,
            "wearable_owner": wearable_owner,
            "daily_steps": daily_steps,
            "diet_quality_score": diet_quality_score
        }])

        # Perform inference
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        disease_prob = probabilities[1] * 100.0
        no_disease_prob = probabilities[0] * 100.0

        st.markdown("---")
        st.subheader("📋 Assessment Results")

        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            if prediction == 1 or disease_prob >= 50.0:
                st.markdown(f"""
                <div class="risk-high">
                    <h3>⚠️ Elevated Heart Disease Risk Detected</h3>
                    <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">
                        The model estimates a <strong>{disease_prob:.1f}% probability</strong> of cardiovascular disease.
                    </p>
                    <p style="font-size: 0.95rem;">
                        <strong>Recommendation:</strong> Consult a qualified cardiologist or healthcare professional for detailed diagnostic evaluation (such as an echocardiogram or stress test).
                    </p>
                </div>
                """, unsafe_allow_html=True)
            elif disease_prob >= 30.0:
                st.markdown(f"""
                <div class="risk-mod">
                    <h3>⚠️ Moderate / Borderline Cardiac Risk</h3>
                    <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">
                        The model estimates a <strong>{disease_prob:.1f}% probability</strong> of cardiovascular disease.
                    </p>
                    <p style="font-size: 0.95rem;">
                        <strong>Recommendation:</strong> Proactively manage risk factors such as blood pressure, LDL cholesterol, stress levels, and exercise frequency.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <h3>✅ Low Heart Disease Risk</h3>
                    <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">
                        The model estimates a <strong>{disease_prob:.1f}% probability</strong> of cardiovascular disease ({no_disease_prob:.1f}% healthy probability).
                    </p>
                    <p style="font-size: 0.95rem;">
                        <strong>Recommendation:</strong> Maintain healthy physical activity, balanced nutrition, and regular routine health check-ups.
                    </p>
                </div>
                """, unsafe_allow_html=True)

        with res_col2:
            st.metric("Predicted Condition", "Heart Disease Likely (Class 1)" if prediction == 1 else "No Heart Disease (Class 0)")
            st.metric("Heart Disease Probability", f"{disease_prob:.1f}%")
            st.progress(disease_prob / 100.0)

        # Contributing Clinical Factors
        st.markdown("### 🔍 Patient Factor Analysis")
        flags = []
        if resting_bp_systolic >= 130 or resting_bp_diastolic >= 85:
            flags.append(f"• **Elevated Blood Pressure**: {resting_bp_systolic}/{resting_bp_diastolic} mmHg (AHA Stage 1/2 Hypertension range).")
        if ldl >= 130:
            flags.append(f"• **Elevated LDL Cholesterol**: {ldl} mg/dL (Optimal is < 100 mg/dL).")
        if hdl < 40 and sex == "Male":
            flags.append(f"• **Low Protective HDL**: {hdl} mg/dL (Desirable > 40 mg/dL for men).")
        elif hdl < 50 and sex == "Female":
            flags.append(f"• **Low Protective HDL**: {hdl} mg/dL (Desirable > 50 mg/dL for women).")
        if max_heart_rate_achieved < 130:
            flags.append(f"• **Sub-optimal Max Heart Rate**: {max_heart_rate_achieved} bpm (Strongest predictive feature identified in model training).")
        if exercise_induced_angina:
            flags.append(f"• **Exercise-Induced Angina**: Positive (Significant symptom indicating coronary artery limitation).")
        if st_depression >= 1.5:
            flags.append(f"• **ST Depression**: {st_depression} mm (ECG marker associated with myocardial ischemia).")
        if smoker_status == "Current":
            flags.append(f"• **Active Smoker**: Known major contributor to arterial plaque formation and vessel constriction.")
        if bmi >= 30.0:
            flags.append(f"• **BMI**: {bmi:.1f} kg/m² (Classified as Obese, increasing cardiac workload).")
        if exercise_minutes_per_week < 90:
            flags.append(f"• **Low Physical Activity**: {exercise_minutes_per_week} mins/week (WHO recommends ≥ 150 mins moderate activity).")

        if flags:
            for f in flags:
                st.markdown(f)
        else:
            st.success("All primary clinical and lifestyle risk markers are within desirable or optimal thresholds!")

# ----------------- TAB 2: MODEL & FEATURE INSIGHTS -----------------
with tab_analytics:
    st.header("📊 Model Architecture & Training Insights")
    st.write("Overview of the machine learning pipeline trained in `Heart_Disease_ML.ipynb`.")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Overall Accuracy", "88.17%")
    col_m2.metric("No Disease Precision", "0.89")
    col_m3.metric("Disease Precision", "0.86")
    col_m4.metric("Training Samples", "9,000")

    st.markdown("---")
    st.subheader("Top Predictive Features Identified by Random Forest")
    st.write("Feature importance analysis from the 100-tree Random Forest ensemble:")

    # Top feature importances as discovered in notebook
    features = [
        "Maximum Heart Rate Achieved",
        "ST Depression (ECG)",
        "Exercise-Induced Angina",
        "LDL Cholesterol",
        "Patient Age",
        "HDL Cholesterol",
        "Exercise Minutes / Week",
        "BMI",
        "Resting BP Systolic",
        "HbA1c Blood Sugar",
        "Fasting Blood Sugar",
        "Diet Quality Score",
        "Total Cholesterol",
        "Stress Score"
    ]
    importances = [0.232, 0.073, 0.069, 0.043, 0.042, 0.039, 0.033, 0.033, 0.032, 0.028, 0.028, 0.028, 0.027, 0.026]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(features[::-1], importances[::-1], color="#E63946")
    ax.set_xlabel("Relative Importance Score", fontsize=10)
    ax.set_title("Top 14 Feature Importances (Random Forest)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Classification Matrix Breakdown")
    col_cm1, col_cm2 = st.columns([1, 1])
    with col_cm1:
        st.table(pd.DataFrame({
            "Class": ["No Heart Disease (0)", "Heart Disease (1)", "Weighted Average"],
            "Precision": ["0.89", "0.86", "0.88"],
            "Recall": ["0.95", "0.72", "0.88"],
            "F1-Score": ["0.92", "0.79", "0.88"],
            "Test Support": ["1,255", "545", "1,800"]
        }))
    with col_cm2:
        st.info("""
        **Key Model Highlights:**
        - **Pipeline Integration:** Raw categorical values and numerical scales are handled within the scikit-learn `Pipeline` automatically using `ColumnTransformer` (`StandardScaler` and `OneHotEncoder`).
        - **High Specificity:** 95% recall on healthy cases minimizes false alarm anxiety.
        - **High Precision:** 86% precision on disease identification ensures that positive flags correlate strongly with genuine cardiac risk.
        """)

# ----------------- TAB 3: GUIDANCE & DISCLAIMER -----------------
with tab_guidance:
    st.header("📖 Clinical Guidance & Heart Health Guidelines")
    
    st.markdown("""
    ### Key Cardiovascular Benchmarks (AHA / ACC Standards)
    - **Blood Pressure:** Normal systolic is < 120 mmHg and diastolic is < 80 mmHg.
    - **Lipid Panel:**
      - Total Cholesterol: Desirable < 200 mg/dL.
      - LDL ("Bad"): Optimal < 100 mg/dL.
      - HDL ("Good"): Protective ≥ 50 mg/dL (females) or ≥ 40 mg/dL (males).
      - Triglycerides: Normal < 150 mg/dL.
    - **Physical Activity:** Aim for at least 150 minutes of moderate aerobic activity or 75 minutes of vigorous activity each week.
    - **Diet & Stress:** Diets rich in whole grains, fiber, and lean proteins combined with stress-reduction techniques substantially lower cardiac morbidity.
    """)

    st.markdown("---")
    st.warning("""
    **Medical Disclaimer:**
    This web application and the underlying machine learning model are designed solely for educational, academic, and research purposes.
    It does not provide medical diagnosis, clinical consultation, or treatment prescriptions.
    Individuals experiencing chest pain, shortness of breath, dizziness, or cardiac symptoms should immediately contact emergency medical services or a licensed healthcare provider.
    """)
