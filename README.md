# 📉 Customer Churn Prediction using ANN

A deep learning project that predicts whether a telecom customer is likely to **churn** using an Artificial Neural Network (ANN).

The project includes a complete machine learning workflow covering **data preprocessing, feature encoding, numerical scaling, ANN model training, model evaluation, artifact saving, and Streamlit deployment**.

## 🚀 Project Overview

Customer churn occurs when a customer stops using a company's services.

Predicting churn can help businesses identify customers who may leave and take proactive retention measures.

In this project, a neural network is trained on the **Telco Customer Churn dataset** to predict the probability that a customer will churn.

The trained model is then integrated into a **Streamlit web application**, where users can enter customer information and receive a churn-risk prediction.

### Workflow

```text
Customer Churn Dataset
        ↓
Data Cleaning
        ↓
Categorical Encoding
        ↓
One-Hot Encoding
        ↓
Min-Max Scaling
        ↓
Train/Test Split
        ↓
Artificial Neural Network
        ↓
Model Evaluation
        ↓
Save Model + Preprocessing Artifacts
        ↓
Streamlit Application
        ↓
Churn Probability
```

## ✨ Features

* Customer churn prediction using Deep Learning
* Artificial Neural Network implemented with TensorFlow/Keras
* Data preprocessing pipeline
* Binary and categorical feature encoding
* Min-Max feature scaling
* Classification evaluation using precision, recall and F1-score
* Saved trained model for inference
* Streamlit-based interactive prediction application
* Consistent preprocessing between training and inference

## 🛠️ Tech Stack

| Technology       | Purpose                                        |
| ---------------- | ---------------------------------------------- |
| Python           | Programming language                           |
| Pandas           | Data manipulation                              |
| NumPy            | Numerical operations                           |
| Scikit-learn     | Preprocessing, train/test split and evaluation |
| TensorFlow       | Deep learning framework                        |
| Keras            | ANN model development                          |
| Joblib           | Saving the scaler                              |
| Streamlit        | Web application                                |
| Jupyter Notebook | Model experimentation                          |

## 📂 Project Structure

```text
Customer_churn_prediction_ANN/
│
├── app.py
├── train_model.py
├── churn.ipynb
├── customer_churn.csv
├── churn_model.keras
├── scaler.joblib
├── feature_columns.json
├── requirements.txt
└── README.md
```

### File Description

#### `churn.ipynb`

The original notebook containing the exploratory analysis and ANN development.

#### `train_model.py`

Recreates the preprocessing and ANN training pipeline and generates the artifacts required by the Streamlit application.

It saves:

```text
churn_model.keras
scaler.joblib
feature_columns.json
```

The repository's training script removes `customerID`, handles blank `TotalCharges`, converts `TotalCharges` to numeric, collapses `"No internet service"` and `"No phone service"` into `"No"`, performs encoding, scales selected numerical features, and trains the ANN.

#### `app.py`

Streamlit application used for real-time churn prediction.

The application loads the trained model, scaler and feature-column configuration and performs the same preprocessing used during training before generating a prediction.

#### `customer_churn.csv`

Dataset used to train and evaluate the model.

#### `churn_model.keras`

Saved TensorFlow/Keras ANN model.

#### `scaler.joblib`

Saved `MinMaxScaler` used during training so that inference uses the same scaling transformation.

#### `feature_columns.json`

Stores the exact feature-column order expected by the trained model.

#### `requirements.txt`

Contains the Python dependencies required to run the project.

## 🧹 Data Preprocessing

The preprocessing pipeline performs several transformations.

### 1. Remove Customer ID

The `customerID` column is removed because it is an identifier and does not provide useful predictive information.

```python
df.drop("customerID", axis="columns", inplace=True)
```

### 2. Handle Total Charges

Some records contain blank values in `TotalCharges`.

These rows are removed and the column is converted to numeric format.

```python
df = df[df.TotalCharges != " "].copy()

df.TotalCharges = pd.to_numeric(df.TotalCharges)
```

### 3. Simplify Service Categories

The values:

```text
No internet service
No phone service
```

are converted to:

```text
No
```

This simplifies the categorical representation.

### 4. Convert Yes/No Features

Binary categorical features are converted into numerical values:

```text
Yes → 1
No  → 0
```

The `gender` column is also encoded:

```text
Female → 1
Male   → 0
```

### 5. One-Hot Encoding

The following categorical features are converted using one-hot encoding:

```text
InternetService
Contract
PaymentMethod
```

This converts categorical values into numerical columns that can be used by the neural network.

### 6. Feature Scaling

The following numerical features are scaled using `MinMaxScaler`:

```text
tenure
MonthlyCharges
TotalCharges
```

Scaling transforms these values into a comparable numerical range.

## 🧠 ANN Architecture

The project uses a feed-forward Artificial Neural Network implemented using Keras.

```text
Input Layer
     ↓
Dense Layer: 26 neurons
Activation: ReLU
     ↓
Dense Layer: 15 neurons
Activation: ReLU
     ↓
Output Layer: 1 neuron
Activation: Sigmoid
```

The architecture is implemented as:

```python
model = keras.Sequential([
    keras.layers.Dense(
        26,
        input_shape=(n_features,),
        activation="relu"
    ),
    keras.layers.Dense(
        15,
        activation="relu"
    ),
    keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])
```

The model is compiled using:

```python
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

The training script uses a test size of **20%**, a fixed random state of **5**, and trains for **100 epochs**.

## 📊 Model Evaluation

Since this is a binary classification problem, the project evaluates the model using:

* Precision
* Recall
* F1-score
* Accuracy

The classification report is generated using:

```python
classification_report(y_test, y_pred)
```

The prediction threshold used by the application is:

```text
Probability > 0.5 → Churn
Probability ≤ 0.5 → No Churn
```

## 🌐 Streamlit Application

The trained model is integrated into an interactive Streamlit application.

The application allows users to enter information such as:

### Customer Profile

* Gender
* Senior citizen status
* Partner
* Dependents
* Tenure
* Monthly charges
* Total charges
* Paperless billing
* Payment method

### Services

* Phone service
* Multiple lines
* Internet service
* Online security
* Online backup
* Device protection
* Technical support
* Streaming TV
* Streaming movies
* Contract type

The application then preprocesses the input using the same transformations used during training and generates a churn probability.

### Prediction Output

The application displays:

```text
High churn risk
```

or

```text
Low churn risk
```

along with the predicted probability.

For example:

```text
⚠️ High churn risk — predicted probability: 78.4%
```

The application also provides a progress bar representing the predicted churn probability.

## 🔄 Training vs Prediction Pipeline

An important part of the project is keeping preprocessing consistent.

### During Training

```text
Raw Dataset
     ↓
Cleaning
     ↓
Encoding
     ↓
MinMaxScaler.fit_transform()
     ↓
ANN Training
```

The fitted scaler and feature order are saved.

### During Prediction

```text
User Input
     ↓
Same Encoding
     ↓
Saved MinMaxScaler.transform()
     ↓
Same Feature Order
     ↓
ANN Prediction
```

The application loads:

```text
churn_model.keras
scaler.joblib
feature_columns.json
```

and uses them during inference.

This prevents a common machine-learning deployment problem where the preprocessing used during inference differs from the preprocessing used during training.

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/adigupta-ds/Customer_churn_prediction_ANN.git

cd Customer_churn_prediction_ANN
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it.

#### Windows

```bash
venv\Scripts\activate
```

#### macOS/Linux

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 🏋️ Train the Model

If you want to recreate the trained model and preprocessing artifacts:

```bash
python train_model.py
```

This generates:

```text
churn_model.keras
scaler.joblib
feature_columns.json
```

The training script explicitly saves these three artifacts for use by the Streamlit application.

## ▶️ Run the Application

After the model artifacts are available, run:

```bash
streamlit run app.py
```

The application will open in your browser.

The repository's application is configured to load the saved model, scaler and feature-column configuration from the same directory.

## 🧪 Example Use Case

Suppose a customer has:

```text
Tenure: 5 months
Monthly Charges: High
Contract: Month-to-month
Internet Service: Fiber optic
Payment Method: Electronic check
```

These customer attributes are passed through the same preprocessing pipeline and then into the trained ANN.

The model outputs a probability:

```text
P(Churn = 1)
```

The application uses a threshold of `0.5` to classify the customer as either high or low churn risk.

## 🧠 Key Machine Learning Concepts Demonstrated

This project demonstrates practical understanding of:

* Binary classification
* Artificial Neural Networks
* Deep learning
* Forward propagation
* ReLU activation
* Sigmoid activation
* Binary cross-entropy
* Adam optimizer
* Train/test split
* One-hot encoding
* Feature scaling
* Min-Max normalization
* Classification metrics
* Model serialization
* ML inference
* Streamlit deployment

## 💡 Why Use an ANN?

An ANN can learn nonlinear relationships between customer characteristics and churn behavior.

Compared with a simple linear model, a neural network can learn more complex interactions between features such as:

```text
Contract
+
Tenure
+
Monthly Charges
+
Internet Service
+
Payment Method
```

The goal is not simply to maximize accuracy, but to identify customers who are likely to leave.

## ⚠️ Limitations

* The model depends on the quality and representativeness of the available dataset.
* A probability threshold of `0.5` may not always be optimal for a business churn-retention problem.
* Accuracy alone may not adequately represent performance when churn classes are imbalanced.
* The current application does not provide explainability for individual predictions.
* The ANN architecture and hyperparameters could be further tuned.

## 🔮 Future Improvements

Possible improvements include:

* Hyperparameter tuning
* Early stopping
* Dropout and regularization
* Class-weight handling for class imbalance
* ROC-AUC and PR-AUC evaluation
* Confusion matrix visualization
* Feature importance / explainability using SHAP
* Experiment comparison with Logistic Regression, Random Forest and XGBoost
* Better probability calibration
* Model monitoring
* Cloud deployment
* Database integration for prediction history
* Customer retention recommendation system

## 📌 Learning Outcome

This project helped me understand how a **deep learning classification model can be taken beyond model training and converted into a usable application**.

The complete workflow is:

```text
Data
 ↓
Cleaning
 ↓
Feature Engineering
 ↓
Encoding
 ↓
Scaling
 ↓
ANN
 ↓
Evaluation
 ↓
Model Serialization
 ↓
Streamlit Deployment
 ↓
Real-Time Prediction
```

## 👨‍💻 Author

**Aditya Sagar Gupta**

GitHub:
https://github.com/adigupta-ds

