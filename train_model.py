"""
train_model.py
Recreates the preprocessing + ANN from the original
Customer_churn_prediction_ANN notebook (adigupta-ds), and saves the
artifacts the Streamlit app needs for inference:

    - churn_model.keras   (trained ANN)
    - scaler.joblib        (MinMaxScaler fit on tenure/MonthlyCharges/TotalCharges)
    - feature_columns.json (exact column order the model expects)

Run once:  python train_model.py
"""

import json
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras

DATA_PATH = "customer_churn.csv"
RANDOM_STATE = 5


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.drop("customerID", axis="columns", inplace=True)

    # TotalCharges has blank strings for a handful of brand-new customers.
    df = df[df.TotalCharges != " "].copy()
    df.TotalCharges = pd.to_numeric(df.TotalCharges)

    # Collapse "No internet/phone service" into a plain "No"
    df.replace("No internet service", "No", inplace=True)
    df.replace("No phone service", "No", inplace=True)

    yes_no_columns = [
        "Partner", "Dependents", "PhoneService", "MultipleLines",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "PaperlessBilling", "Churn",
    ]
    for col in yes_no_columns:
        df[col] = df[col].replace({"Yes": 1, "No": 0})

    df["gender"] = df["gender"].replace({"Female": 1, "Male": 0})

    df = pd.get_dummies(
        data=df, columns=["InternetService", "Contract", "PaymentMethod"]
    )
    return df


def main():
    df = load_and_clean(DATA_PATH)

    cols_to_scale = ["tenure", "MonthlyCharges", "TotalCharges"]
    scaler = MinMaxScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    X = df.drop("Churn", axis="columns")
    y = df["Churn"].astype("float32")

    feature_columns = X.columns.tolist()
    X = X.astype("float32")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    n_features = len(feature_columns)
    model = keras.Sequential([
        keras.layers.Dense(26, input_shape=(n_features,), activation="relu"),
        keras.layers.Dense(15, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=100, verbose=2)

    print("\n--- Test set evaluation ---")
    model.evaluate(X_test, y_test)

    y_pred = (model.predict(X_test) > 0.5).astype(int)
    print(classification_report(y_test, y_pred))

    # ---- Save artifacts for the Streamlit app ----
    model.save("churn_model.keras")
    joblib.dump(scaler, "scaler.joblib")
    with open("feature_columns.json", "w") as f:
        json.dump(feature_columns, f, indent=2)

    print("\nSaved churn_model.keras, scaler.joblib, feature_columns.json")


if __name__ == "__main__":
    main()
