import streamlit as st
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt
import zipfile
import os

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")
st.title("❤️‍🩹 Heart Disease Prediction")

# ---------- Load model directly (cached) ----------
# Path to the zip file
zip_path = 'best_rf_model.zip'
unzip_dir = 'model/'  # Directory where the unzipped model will be stored

# Unzip the model file if it does not exist already
if not os.path.exists(unzip_dir):
    os.makedirs(unzip_dir)

if not os.path.exists(os.path.join(unzip_dir, 'best_rf_model.joblib')):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_dir)
    except Exception as e:
        st.error(f"Error extracting model: {e}")
        st.stop()

# Load the model from the unzipped folder
try:
    model = load(os.path.join(unzip_dir, 'best_rf_model.joblib'))
except FileNotFoundError:
    st.error("Could not find best_rf_model.joblib after extraction.")
    st.stop()
    
# ---------- What the model expects ----------
expected = list(getattr(model, "feature_names_in_", []))
if not expected:
    expected = [
        "Fasting Blood Sugar","BMI","Cholesterol Level","Sleep Hours","Age",
        "Stress Level","Sugar Consumption","Exercise Habits",
        "Gender_Male","Smoking_Yes","High Blood Pressure_Yes"
    ]
st.caption(f"Model expects {len(expected)} features: {expected}")

# ---------- Inputs (in a form so it only submits once) ----------
with st.form("predict_form"):
    st.subheader("Patient Inputs")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=0, max_value=100, value=30)
        gender = st.selectbox("Gender", ["Female", "Male"])
        sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0)
        stress_level = st.selectbox("Stress Level (Low/Medium/High)", ["Low","Medium","High"])
        smoking = st.selectbox("Smoking", ["No", "Yes"])
        alcohol = st.selectbox("Alcohol Consumption", ["No","Yes"])
        family_hd = st.selectbox("Family Heart Disease", ["No","Yes"])

    with col2:
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.0)
        cholesterol_level = st.number_input("Total Cholesterol (mg/dL)", min_value=0.0, max_value=450.0, value=190.0)
        fasting_bs = st.number_input("Fasting Blood Sugar",min_value=0.0, max_value=1000.0, value=140.0)
        high_bp = st.selectbox("High Blood Pressure (diagnosed)", ["No", "Yes"])
        diabetes = st.selectbox("Diabetes", ["No","Yes"])
        triglyceride = st.number_input("Triglyceride (mg/dL)", min_value=0.0, max_value=1000.0, value=150.0)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=250, value=120)

    with col3:
        exercise_habits = st.selectbox("Exercise Habits (Low/Medium/High)", ["Low","Medium","High"])
        sugar_consumption = st.selectbox("Sugar Consumption (Low/Medium/High)", ["Low","Medium","High"])
        low_hdl = st.selectbox("Low HDL Cholesterol (diagnosed)", ["No","Yes"])
        high_ldl = st.selectbox("High LDL Cholesterol (diagnosed)", ["No","Yes"])
        crp = st.number_input("CRP Level (mg/L)", min_value=0.0, max_value=50.0, value=1.0)
        homocysteine = st.number_input("Homocysteine (µmol/L)", min_value=0.0, max_value=60.0, value=10.0)
        # spacer / note to keep columns visually even (you can replace with any extra field later)
        st.caption(" ")  # keeps the column height consistent

    submitted = st.form_submit_button("Predict")

# Only compute + render after submit
if submitted:
    # Map to model features
    label_map = {"Low": 0, "Medium": 1, "High": 2}
    row = {
        "Fasting Blood Sugar": 1 if fasting_bs == "High" else 0,
        "BMI": float(bmi),
        "Cholesterol Level": float(cholesterol_level),
        "Sleep Hours": float(sleep_hours),
        "Age": int(age),
        "Stress Level": label_map[stress_level],
        "Sugar Consumption": label_map[sugar_consumption],
        "Exercise Habits": label_map[exercise_habits],
        "Gender_Male": 1 if gender == "Male" else 0,
        "Smoking_Yes": 1 if smoking == "Yes" else 0,
        "High Blood Pressure_Yes": 1 if high_bp == "Yes" else 0,
    }
    input_df = pd.DataFrame([row], columns=expected)

    with st.expander("See the exact feature vector sent to the model"):
        st.write(input_df)
le = LabelEncoder()
y = le.fit_transform(y_raw)
    # Predict once
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0, 1]

    # One result container (prevents duplicates)
    result_box = st.container()
    with result_box:
        st.success(f"Prediction: {'HAS Heart Disease' if pred == 1 else 'NO Heart Disease'}")
        st.subheader("Probability")
        st.metric("Probability (of Heart Disease)", f"{proba:.3f}")

        # Probability bar
        with st.expander("📊 Probability of Heart Disease", expanded=True):

            fig_prob, ax_prob = plt.subplots(figsize=(6, 0.6))
            ax_prob.barh([0], [proba], height=0.4, color="tomato")
            ax_prob.set_xlim(0, 1)
            ax_prob.set_yticks([])
            ax_prob.set_xlabel("0 = Low   ————————   1 = High")
            ax_prob.set_title("Risk Probability")
            for spine in ["top", "right", "left"]:
                ax_prob.spines[spine].set_visible(False)

            st.pyplot(fig_prob)

        # Feature importances
        with st.expander("📌 Top Feature Importances", expanded=True):

            importances = getattr(model, "feature_importances_", None)
            if importances is not None:
                fi = pd.Series(importances, index=expected).sort_values(ascending=True)
                topk = min(8, len(fi))

                fig_fi, ax_fi = plt.subplots(figsize=(6, 0.4*topk + 0.6))
                fi.tail(topk).plot(kind="barh", ax=ax_fi, color="skyblue")
                ax_fi.set_xlabel("Importance")
                ax_fi.set_ylabel("")
                ax_fi.set_title("Top Feature Importances")
                st.pyplot(fig_fi)
            else:
                st.info("This model does not expose feature importances.")

        # Informational ranges
        with st.expander("📋 Your Values vs Common Reference Ranges", expanded=True):

            def range_bar(name, low, high, val, min_axis=None, max_axis=None):
                min_axis = low if min_axis is None else min_axis
                max_axis = high if max_axis is None else max_axis
                fig, ax = plt.subplots(figsize=(6, 0.5))
                ax.barh([0], [max_axis-min_axis], left=min_axis, height=0.35, alpha=0.15, color="grey")
                ax.barh([0], [high-low], left=low, height=0.35, alpha=0.35, color="green")
                ax.plot([val, val], [-0.25, 0.25], color="red")
                ax.set_xlim(min_axis, max_axis)
                ax.set_yticks([])
                ax.set_title(f"{name}: {val:.2f} (target {low}–{high})")
                for spine in ["top", "right", "left"]:
                    ax.spines[spine].set_visible(False)
                st.pyplot(fig)

            range_bar("BMI", 18.5, 24.9, float(bmi), 10, 40)
            range_bar("Sleep Hours", 7.0, 9.0, float(sleep_hours), 0, 12)
            range_bar("Total Cholesterol (mg/dL)", 0.0, 200.0, float(cholesterol_level), 0, 300)

            st.write(f"Fasting Blood Sugar status: **{fasting_bs}**")
