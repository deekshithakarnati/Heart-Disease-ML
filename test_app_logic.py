import os
import joblib
import pandas as pd
import numpy as np

def run_tests():
    model_path = "heart_disease_model.pkl"
    assert os.path.exists(model_path), f"Model file {model_path} does not exist!"
    
    print("Loading model...")
    model = joblib.load(model_path)
    print("Model loaded successfully:", type(model))

    # Test presets
    test_cases = [
        {
            "name": "Healthy Profile",
            "data": {
                "age": 35, "sex": "Female", "resting_bp_systolic": 115, "resting_bp_diastolic": 75,
                "cholesterol_total": 170, "hdl": 65, "ldl": 85, "triglycerides": 100,
                "fasting_blood_sugar": 90, "hba1c": 5.2, "bmi": 22.4, "resting_heart_rate": 68,
                "max_heart_rate_achieved": 180, "chest_pain_type": "Asymptomatic",
                "exercise_induced_angina": False, "st_depression": 0.2, "family_history": False,
                "smoker_status": "Never", "alcohol_units_per_week": 1.5,
                "exercise_minutes_per_week": 210, "sleep_hours": 7.5, "stress_score": 25.0,
                "wearable_owner": True, "daily_steps": 9500, "diet_quality_score": 85.0
            }
        },
        {
            "name": "High Risk Profile",
            "data": {
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
    ]

    for tc in test_cases:
        df = pd.DataFrame([tc["data"]])
        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0]
        print(f"[{tc['name']}] -> Prediction: {pred}, Probability: {proba}")
        assert pred in [0, 1], f"Unexpected prediction value: {pred}"
        assert abs(sum(proba) - 1.0) < 1e-4, f"Probabilities do not sum to 1: {proba}"
        assert 0.0 <= proba[1] <= 1.0, f"Invalid disease probability: {proba[1]}"

    print("\nAll unit tests PASSED successfully!")

if __name__ == "__main__":
    run_tests()
