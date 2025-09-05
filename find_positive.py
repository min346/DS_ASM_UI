from joblib import load
import pandas as pd, itertools

model = load("best_rf_model.joblib")

# Define candidate ranges and brute force
ages = [30, 45, 60]
bmis = [18, 25, 35]
cholesterols = [180, 250, 300]
smoking = [0, 1]  # No=0, Yes=1
bp = [0, 1]       # High BP No=0, Yes=1

for age, bmi, chol, smk, hbp in itertools.product(ages, bmis, cholesterols, smoking, bp):
    row = {
        "Fasting Blood Sugar": 1,
        "BMI": bmi,
        "Cholesterol Level": chol,
        "Sleep Hours": 3,
        "Age": age,
        "Stress Level": 2,
        "Sugar Consumption": 2,
        "Exercise Habits": 0,
        "Gender_Male": 1,
        "Smoking_Yes": smk,
        "High Blood Pressure_Yes": hbp,
    }
    df = pd.DataFrame([row])
    pred = model.predict(df)[0]
    if pred == 1:
        print("Found a positive case:", row)
        break
