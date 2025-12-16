import joblib
import pandas as pd
import numpy as np

MODEL_PATH = "artifacts/model_data.joblib"

model_data = joblib.load(MODEL_PATH)
model = model_data["model"]
scaler = model_data["scaler"]
features = model_data["features"]
cols_to_scale = model_data["cols_to_scale"]


def prepare_df(
    age, income, loan_amount, loan_tenure_months,
    avg_dpd_per_delinquency, credit_utilization_ratio,
    employment_years, residence_type, loan_purpose, loan_type
):
    data = {
        'age': age,
        'income': income,
        'loan_amount': loan_amount,
        'loan_tenure_months': loan_tenure_months,
        'credit_utilization_ratio': credit_utilization_ratio,
        'avg_dpd_per_delinquency': avg_dpd_per_delinquency,
        'loan_to_income': loan_amount / income if income > 0 else 0,
        'employment_years': employment_years,

        'residence_type_Owned': 1 if residence_type == 'Owned' else 0,
        'residence_type_Rented': 1 if residence_type == 'Rented' else 0,

        'loan_purpose_Education': 1 if loan_purpose == 'Education' else 0,
        'loan_purpose_Home': 1 if loan_purpose == 'Home' else 0,
        'loan_purpose_Personal': 1 if loan_purpose == 'Personal' else 0,

        'loan_type_Unsecured': 1 if loan_type == 'Unsecured' else 0,
    }

    df = pd.DataFrame([data])

    # ✅ Ensure ALL model features exist
    for col in features:
        if col not in df.columns:
            df[col] = 0

    # ✅ Ensure ALL scaled columns exist
    for col in cols_to_scale:
        if col not in df.columns:
            df[col] = 0

    # ✅ Correct column order
    df = df[features]

    # ✅ Safe scaling
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    return df




def calculate_credit_score(input_df, base_score=300, scale_length=600):
    x = np.dot(input_df.values, model.coef_.T) + model.intercept_
    default_probability = 1 / (1 + np.exp(-x))
    non_default_probability = 1 - default_probability

    score = base_score + non_default_probability.flatten()[0] * scale_length

    if score < 500:
        rating = "Poor"
    elif score < 650:
        rating = "Average"
    elif score < 750:
        rating = "Good"
    else:
        rating = "Excellent"

    return float(default_probability[0]), int(score), rating


def predict(age, income, loan_amount, loan_tenure_months,
            avg_dpd_per_delinquency, credit_utilization_ratio,
            employment_years, residence_type, loan_purpose, loan_type):

    df = prepare_df(
        age, income, loan_amount, loan_tenure_months,
        avg_dpd_per_delinquency, credit_utilization_ratio,
        employment_years, residence_type, loan_purpose, loan_type
    )

    return calculate_credit_score(df)
