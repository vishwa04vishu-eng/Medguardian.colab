import streamlit as st
import joblib
import numpy as np

# Load the trained pipeline (scaler + model)
model = joblib.load("vishwas.joblib")

st.title("🩺 MedInsight: AI-Assisted Disease Detection and Prediction System")
st.write("Enter patient details below to get a prediction.")

# Input fields (⚠️ must exactly match training feature order)
age = st.number_input("Age", min_value=0, max_value=120, value=30)
gender = st.selectbox("Gender", [0, 1])  # 0 = Female, 1 = Male
blood_pressure = st.number_input("Blood Pressure", min_value=50, max_value=200, value=120)
cholesterol_level = st.number_input("Cholesterol Level", min_value=100, max_value=400, value=200)
exercise_habits = st.number_input("Exercise Habits (e.g., 0,1,2)", min_value=0, max_value=5, value=1)
smoking = st.selectbox("Smoking", [0, 1])
family_heart_disease = st.selectbox("Family Heart Disease", [0, 1])
diabetes = st.selectbox("Diabetes", [0, 1])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0)
high_blood_pressure = st.selectbox("High Blood Pressure", [0, 1])
low_hdl_cholesterol = st.selectbox("Low HDL Cholesterol", [0, 1])
high_ldl_cholesterol = st.selectbox("High LDL Cholesterol", [0, 1])
alcohol_consumption = st.number_input("Alcohol Consumption (e.g., 0,1,2)", min_value=0, max_value=5, value=1)
stress_level = st.number_input("Stress Level", min_value=0, max_value=10, value=5)
sugar_consumption = st.number_input("Sugar Consumption (grams/day)", min_value=0, max_value=500, value=50)
triglyceride_level = st.number_input("Triglyceride Level", min_value=50, max_value=500, value=150)

# ✅ Changed: now accept numeric mg/dl instead of 0/1
fasting_blood_sugar = st.number_input("Fasting Blood Sugar (mg/dl)", min_value=50, max_value=300, value=100)

crp_level = st.number_input("CRP Level", min_value=0.0, max_value=20.0, value=1.0)
homocysteine_level = st.number_input("Homocysteine Level", min_value=0.0, max_value=100.0, value=10.0)

# Predict button
if st.button("Predict"):
    input_data = np.array([[age, gender, blood_pressure, cholesterol_level, exercise_habits,
                            smoking, family_heart_disease, diabetes, bmi, high_blood_pressure,
                            low_hdl_cholesterol, high_ldl_cholesterol, alcohol_consumption,
                            stress_level, sugar_consumption, triglyceride_level,
                            fasting_blood_sugar, crp_level, homocysteine_level]])

    # --- DEBUG INFO ---
    pred = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]
    classes = model.classes_

    st.write("🔍 **Raw Prediction (class):**", pred)
    st.write("📊 **Prediction Probabilities:**", proba)
    st.write("🏷 **Model Classes:**", classes)

    # --- Decision display ---
    # Decide which class is High Risk
    if 1 in classes:
        idx_high_risk = list(classes).index(1)
        proba_high_risk = proba[idx_high_risk]
    else:
        idx_high_risk = list(classes).index(0)
        proba_high_risk = proba[idx_high_risk]

    if proba_high_risk >= 0.5:
        st.error(f"⚠The Person have a Heart Disease! (Probability: {proba_high_risk:.2f})")
    else:
        st.success(f"✅ The Person does not have heart disease (Probability: {proba_high_risk:.2f})")

st.write("---")
st.caption("MedGuardian: Early detection saves lives.")
