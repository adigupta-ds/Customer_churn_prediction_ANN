# 💰 Medical Insurance Charges Prediction

An end-to-end Machine Learning project that predicts medical insurance charges using **Decision Tree Regression**, deploys the trained model through an **Azure Machine Learning Online Endpoint**, and provides a **Streamlit** application for real-time predictions.

## 📸 Streamlit Application

![Medical Insurance Charges Predictor](docs/streamlit-app-preview.png)

## 🚀 Project Overview

```text
Dataset
   ↓
Data Cleaning
   ↓
Preprocessing
   ↓
Decision Tree Regression
   ↓
Hyperparameter Tuning
   ↓
Model Evaluation
   ↓
model.pkl
   ↓
Azure ML Online Endpoint
   ↓
REST API
   ↓
Streamlit Application
   ↓
Insurance Charges Prediction
```

## 🧠 Machine Learning

**Problem:** Supervised regression

**Target:** `charges`

**Algorithm:** `DecisionTreeRegressor`

The preprocessing pipeline uses:

- Numerical imputation using the median
- Categorical imputation using the most frequent value
- `OneHotEncoder(handle_unknown="ignore")`
- `ColumnTransformer`
- `Pipeline`

Hyperparameters tuned with `GridSearchCV`:

- `max_depth`
- `min_samples_split`
- `min_samples_leaf`

Evaluation metrics:

- MAE
- MSE
- RMSE
- R²

## 📊 Dataset

Features:

```text
age
sex
bmi
children
smoker
Claim_Amount
past_consultations
num_of_steps
Hospital_expenditure
NUmber_of_past_hospitalizations
Anual_Salary
region
```

Target:

```text
charges
```

The original dataset column names are preserved so that training, Azure inference, and Streamlit use the same schema.

## ☁️ Azure ML Deployment

The trained `model.pkl` is deployed as an Azure Machine Learning managed online endpoint.

```text
model.pkl
    ↓
Azure ML
    ↓
Managed Online Endpoint
    ↓
Scoring URI + API Key
```

The Azure scoring script:

1. Loads the model from `AZUREML_MODEL_DIR`.
2. Receives JSON input.
3. Converts the records to a Pandas DataFrame.
4. Runs `model.predict()`.
5. Returns predictions as JSON.

## 🎨 Streamlit

Streamlit collects:

- Age
- Sex
- BMI
- Children
- Smoker
- Claim Amount
- Past Consultations
- Number of Steps
- Hospital Expenditure
- Past Hospitalizations
- Annual Salary
- Region

The application sends the input to Azure using an HTTP POST request.

```text
Streamlit
    ↓
JSON Payload
    ↓
HTTP POST
    ↓
Azure ML Endpoint
    ↓
Decision Tree
    ↓
Prediction
    ↓
Streamlit
```

## 🔌 API Request

Example:

```json
{
  "data": [
    {
      "age": 30,
      "sex": "male",
      "bmi": 25.5,
      "children": 1,
      "smoker": "no",
      "Claim_Amount": 5000,
      "past_consultations": 2,
      "num_of_steps": 5000,
      "Hospital_expenditure": 10000,
      "NUmber_of_past_hospitalizations": 1,
      "Anual_Salary": 600000,
      "region": "southwest"
    }
  ]
}
```

Example response:

```json
{
  "predictions": [
    14752.63
  ]
}
```

## 🔐 Security

Never hard-code the Azure API key in `app.py`.

Create:

```text
.streamlit/secrets.toml
```

```toml
AZURE_ENDPOINT = "YOUR_AZURE_ENDPOINT_URL"
AZURE_API_KEY = "YOUR_AZURE_API_KEY"
```

Add this to `.gitignore`:

```text
.streamlit/secrets.toml
```

Never commit the API key to GitHub.

## 📁 Project Structure

```text
medical-insurance-ml/
│
├── data/
│   └── insurance.csv
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── model/
│   └── model.pkl
│
├── azure/
│   ├── score.py
│   ├── endpoint.yml
│   ├── deployment.yml
│   └── requirements.txt
│
├── streamlit/
│   └── app.py
│
├── .streamlit/
│   └── secrets.toml
│
├── docs/
│   └── streamlit-app-preview.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

## 💻 Local Setup

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd medical-insurance-ml
```

Create a virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🏋️ Train the Model

```bash
python src/train.py
```

This creates:

```text
model/model.pkl
```

## 🧪 Test Locally

```bash
python src/predict.py
```

## 🌐 Run Streamlit

After adding your Azure endpoint and key to `.streamlit/secrets.toml`:

```bash
streamlit run streamlit/app.py
```

## ☁️ Azure Deployment Flow

1. Create/select an Azure ML workspace.
2. Create a managed online endpoint.
3. Deploy the model and `score.py`.
4. Verify the deployment is healthy.
5. Obtain the scoring URI.
6. Obtain the endpoint key.
7. Test the endpoint directly.
8. Add the URI and key to Streamlit secrets.
9. Run the Streamlit application.

> Azure ML CLI syntax and supported runtime images can change. Verify the current Microsoft Azure ML documentation before deployment.

## 🛠️ Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Decision Tree Regression
- GridSearchCV
- Joblib
- Azure Machine Learning
- REST API
- Streamlit
- Git/GitHub

## 🎯 Interview Summary

> I built an end-to-end medical insurance charges prediction system using Python and Scikit-learn. I treated it as a supervised regression problem and used a Decision Tree Regressor. I handled numerical and categorical features using a ColumnTransformer and combined preprocessing with the model in a Pipeline. I used GridSearchCV with five-fold cross-validation for hyperparameter tuning and evaluated the model using MAE, RMSE and R². I serialized the complete pipeline using Joblib and deployed it as an Azure Machine Learning managed online endpoint. Finally, I created a Streamlit frontend that collects customer information, sends it as JSON to the Azure endpoint using authenticated HTTP requests, and displays the predicted insurance charges.

## 🚀 Future Improvements

- Compare Decision Tree with Random Forest and Gradient Boosting.
- Add experiment tracking and model versioning.
- Add model/data drift monitoring.
- Add CI/CD.
- Add automated testing.
- Add API input validation.
- Use Azure Key Vault for secrets.
- Add logging and monitoring.
- Implement automated retraining.

## 👨‍💻 Author

**Aditya Sagar Gupta**

B.Tech – Information Technology  
YCCE, Nagpur
