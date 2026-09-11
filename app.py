"""
Streamlit app for the Customer Churn Prediction (ANN) project.
https://github.com/adigupta-ds/Customer_churn_prediction_ANN

Run:
    streamlit run app.py

Expects churn_model.keras, scaler.joblib, feature_columns.json in the
same folder (created by train_model.py).
"""

import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")


# ----------------------------------------------------------------------
# Load artifacts (cached so they're only loaded once per session)
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("churn_model.keras")
    scaler = joblib.load("scaler.joblib")
    with open("feature_columns.json") as f:
        feature_columns = json.load(f)
    return model, scaler, feature_columns


model, scaler, FEATURE_COLUMNS = load_artifacts()

YES_NO_FIELDS = [
    "Partner", "Dependents", "PhoneService", "MultipleLines",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "PaperlessBilling",
]


def build_raw_input(form_values: dict) -> pd.DataFrame:
    """Build a single-row DataFrame in the exact raw-column shape the
    original notebook used, before encoding/scaling."""
    row = {
        "gender": form_values["gender"],
        "SeniorCitizen": form_values["SeniorCitizen"],
        "Partner": form_values["Partner"],
        "Dependents": form_values["Dependents"],
        "tenure": form_values["tenure"],
        "PhoneService": form_values["PhoneService"],
        "MultipleLines": form_values["MultipleLines"],
        "InternetService": form_values["InternetService"],
        "OnlineSecurity": form_values["OnlineSecurity"],
        "OnlineBackup": form_values["OnlineBackup"],
        "DeviceProtection": form_values["DeviceProtection"],
        "TechSupport": form_values["TechSupport"],
        "StreamingTV": form_values["StreamingTV"],
        "StreamingMovies": form_values["StreamingMovies"],
        "Contract": form_values["Contract"],
        "PaperlessBilling": form_values["PaperlessBilling"],
        "PaymentMethod": form_values["PaymentMethod"],
        "MonthlyCharges": form_values["MonthlyCharges"],
        "TotalCharges": form_values["TotalCharges"],
    }
    return pd.DataFrame([row])


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror train_model.py's preprocessing exactly, then align columns
    to the order the model was trained on."""
    df = df.copy()

    # MultipleLines can only be Yes/No here since we ask PhoneService
    # separately in the UI ("No phone service" collapses to "No").
    for col in YES_NO_FIELDS:
        df[col] = df[col].replace({"Yes": 1, "No": 0})

    df["gender"] = df["gender"].replace({"Female": 1, "Male": 0})

    df = pd.get_dummies(df, columns=["InternetService", "Contract", "PaymentMethod"])

    # Scale the same 3 numeric columns with the SAME fitted scaler
    cols_to_scale = ["tenure", "MonthlyCharges", "TotalCharges"]
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    # Add any one-hot columns missing for this particular input (e.g. a
    # PaymentMethod value that produces columns not created above), then
    # reindex to the exact training column order.
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    df = df[FEATURE_COLUMNS]

    return df.astype("float32")


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("📉 Customer Churn Predictor")
st.caption(
    "ANN model trained on the Telco Customer Churn dataset — "
    "[GitHub repo](https://github.com/adigupta-ds/Customer_churn_prediction_ANN)"
)

with st.form("churn_form"):
    st.subheader("Customer profile")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
    with col2:
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, step=1.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 840.0, step=10.0)
        paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )

    st.subheader("Services")
    col3, col4 = st.columns(2)
    with col3:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple Lines", ["No", "Yes"] if phone_service == "Yes" else ["No"]
        )
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox(
            "Online Security", ["No", "Yes"] if internet_service != "No" else ["No"]
        )
        online_backup = st.selectbox(
            "Online Backup", ["No", "Yes"] if internet_service != "No" else ["No"]
        )
    with col4:
        device_protection = st.selectbox(
            "Device Protection", ["No", "Yes"] if internet_service != "No" else ["No"]
        )
        tech_support = st.selectbox(
            "Tech Support", ["No", "Yes"] if internet_service != "No" else ["No"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV", ["No", "Yes"] if internet_service != "No" else ["No"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies", ["No", "Yes"] if internet_service != "No" else ["No"]
        )
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

    submitted = st.form_submit_button("Predict churn risk", use_container_width=True)

if submitted:
    form_values = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    raw_df = build_raw_input(form_values)
    X = preprocess(raw_df)

    prob = float(model.predict(X, verbose=0)[0][0])
    will_churn = prob > 0.5

    st.divider()
    st.subheader("Result")

    if will_churn:
        st.error(f"⚠️ High churn risk — predicted probability: **{prob:.1%}**")
    else:
        st.success(f"✅ Low churn risk — predicted probability: **{prob:.1%}**")

    st.progress(min(max(prob, 0.0), 1.0))

    with st.expander("See model input (after preprocessing)"):
        st.dataframe(X.T.rename(columns={0: "value"}))

st.divider()
st.caption(
    "Model: 3-layer ANN (26 → 15 → 1, ReLU/sigmoid) trained with Keras. "
    "Preprocessing mirrors the original notebook: Yes/No → 1/0, one-hot "
    "encoding for InternetService/Contract/PaymentMethod, MinMax scaling "
    "for tenure/MonthlyCharges/TotalCharges."
)
