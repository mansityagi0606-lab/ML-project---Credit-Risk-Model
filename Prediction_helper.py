import joblib
import pandas as pd
import numpy as np

MODEL_PATH = "artifacts/model_data.joblib"

model_data = joblib.load(MODEL_PATH)
model = model_data["model"]
scaler = model_data["scaler"]
features = model_data["features"]
cols_to_scale = model_data["cols_to_scale"]


def prepare_df(age, income, loan_amount, loan_tenure_months,
               avg_dpd_per_delinquency, credit_utilization_ratio,
               employment_years, residence_type, loan_purpose, loan_type):

    input_data = {
        "age": age,
        "loan_tenure_months": loan_tenure_months,
        "number_of_open_accounts": 1,
        "credit_utilization_ratio": credit_utilization_ratio,
        "loan_to_income": loan_amount / income if income > 0 else 0,
        "delinquency_ratio": 0.1,
        "avg_dpd_per_delinquency": avg_dpd_per_delinquency,

        "residence_type_Owned": int(residence_type == "Owned"),
        "residence_type_Rented": int(residence_type == "Rented"),

        "loan_purpose_Education": int(loan_purpose == "Education"),
        "loan_purpose_Home": int(loan_purpose == "Home"),
        "loan_purpose_Personal": int(loan_purpose == "Personal"),

        "loan_type_Unsecured": int(loan_type == "Unsecured"),

        "number_of_dependents": 1,
        "years_at_current_address": 1,
        "zipcode": 1,
        "sanction_amount": loan_amount,
        "processing_fee": 1,
        "gst": 1,
        "net_disbursement": loan_amount,
        "principal_outstanding": 1,
        "bank_balance_at_application": 1,
        "number_of_closed_accounts": 1,
        "enquiry_count": 1
    }

    df = pd.DataFrame([input_data])

    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    df = df[features]

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
