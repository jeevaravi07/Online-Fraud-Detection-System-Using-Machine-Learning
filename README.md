# Fraud Detection System Using Machine Learning

## 📌 Project Overview

The **Fraud Detection System** is a machine-learning-based web application developed using **Python, Flask, Scikit-learn, XGBoost, LightGBM, and SQLite**. The system analyzes financial transaction information and predicts whether a transaction is **SAFE, SUSPICIOUS, or FRAUD**.

The project combines multiple machine-learning algorithms and handles class imbalance using **SMOTE (Synthetic Minority Over-sampling Technique)**. Different models are trained and evaluated using metrics such as Accuracy, Precision, Recall, F1-Score, ROC-AUC, Precision@K, computational cost, and prediction latency.

The best-performing model based on ROC-AUC is automatically selected and saved for deployment in the Flask web application.

---

## 🎯 Objectives

* Detect fraudulent financial transactions automatically.
* Compare multiple machine-learning classification algorithms.
* Handle imbalanced fraud datasets using SMOTE.
* Perform feature scaling using StandardScaler.
* Encode categorical transaction types using one-hot encoding.
* Evaluate models using multiple performance metrics.
* Calculate the financial cost associated with false positives and false negatives.
* Deploy the trained model through a Flask web application.
* Provide user registration and login functionality.
* Store prediction history using SQLite.
* Display fraud probability and transaction classification to the user.

---

## 🚀 Main Features

### Machine Learning

The training pipeline includes:

* Logistic Regression
* Random Forest
* XGBoost
* LightGBM
* Multi-Layer Perceptron Neural Network

### Data Processing

* Categorical feature encoding
* Train-test splitting
* Standardization
* SMOTE-based class balancing
* Feature selection and preservation

### Model Evaluation

The following metrics are calculated:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Precision@K
* Confusion Matrix
* False Positive Cost
* False Negative Cost
* Prediction Latency

### Web Application

The Flask application provides:

* Home page
* About page
* User registration
* User login
* Logout
* Transaction prediction
* Fraud probability
* Prediction history storage
* SQLite database integration

---

## 🏗️ System Architecture

```text
                ┌─────────────────────┐
                │   Fraud Dataset     │
                │ fraud_dataset.csv   │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Data Preprocessing  │
                │                     │
                │ • Encoding          │
                │ • Feature Selection │
                │ • Train/Test Split  │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Feature Scaling     │
                │ StandardScaler      │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │       SMOTE         │
                │ Class Balancing     │
                └──────────┬──────────┘
                           │
                           ▼
        ┌─────────────────────────────────────┐
        │       Machine Learning Models       │
        │                                     │
        │ • Logistic Regression               │
        │ • Random Forest                     │
        │ • XGBoost                           │
        │ • LightGBM                          │
        │ • Neural Network                    │
        └──────────────────┬──────────────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Model Evaluation    │
                │                     │
                │ ROC-AUC             │
                │ Precision           │
                │ Recall              │
                │ F1 Score            │
                │ Cost                │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   Best ML Model     │
                │ best_model.pkl      │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │    Flask Web App    │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Transaction Input   │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Prediction Result   │
                │                     │
                │ SAFE                │
                │ SUSPICIOUS          │
                │ FRAUD               │
                └─────────────────────┘
```

---

## 🧠 Machine Learning Workflow

### 1. Dataset Loading

The system loads the transaction dataset from:

```text
fraud_dataset.csv
```

The target column used for classification is:

```text
isFraud
```

---

### 2. Categorical Encoding

Categorical columns are converted into numerical representations using one-hot encoding.

```python
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
```

This allows machine-learning algorithms to process categorical transaction information.

---

### 3. Train-Test Split

The dataset is divided into training and testing subsets using an 80:20 ratio.

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Stratification maintains the class distribution between the training and testing sets.

---

### 4. Feature Scaling

Numerical features are standardized using:

```text
StandardScaler
```

The trained scaler is saved as:

```text
model/scaler.pkl
```

---

### 5. Handling Class Imbalance

Fraud datasets generally contain significantly fewer fraudulent transactions than legitimate transactions.

To address this issue, the project uses:

```text
SMOTE
```

SMOTE generates synthetic samples for the minority class using the training data.

---

## 🤖 Machine Learning Models

### Logistic Regression

Logistic Regression provides a statistical baseline for binary fraud classification and estimates the probability that a transaction belongs to the fraudulent class.

### Random Forest

Random Forest combines multiple decision trees to generate a robust classification model. It can capture nonlinear relationships between transaction attributes.

### XGBoost

XGBoost is a gradient-boosting algorithm that builds an ensemble of decision trees sequentially. It is commonly used for structured/tabular classification problems.

### LightGBM

LightGBM is another gradient-boosting framework designed for efficient training and strong performance on tabular datasets.

### Neural Network

The project uses an MLPClassifier with two hidden layers:

```text
64 neurons
   ↓
32 neurons
   ↓
Output
```

The neural network learns nonlinear relationships between transaction features.

---

## 📊 Evaluation Metrics

### Accuracy

Measures the proportion of correctly classified transactions.

### Precision

Measures the proportion of transactions predicted as fraud that are actually fraudulent.

### Recall

Measures how many actual fraudulent transactions are successfully detected.

### F1-Score

Provides a combined measure of Precision and Recall.

### ROC-AUC

Measures the model's ability to distinguish between fraudulent and legitimate transactions across classification thresholds.

### Precision@K

The project calculates Precision@K for the top 5% highest-risk transactions.

This is useful in fraud investigation scenarios where investigators may only be able to manually review a limited number of transactions.

---

## 💰 Cost-Sensitive Evaluation

The project considers different costs for false negatives and false positives.

```python
FN_COST = 10000
FP_COST = 500
```

The total cost is calculated as:

```text
Total Cost =
(False Negatives × FN Cost)
+
(False Positives × FP Cost)
```

This provides an additional business-oriented measure of model performance.

---

## 🌐 Flask Web Application

The Flask application loads the trained model and scaler:

```python
model = joblib.load("model/best_model.pkl")
scaler = joblib.load("model/scaler.pkl")
```

The application accepts transaction information such as:

* Step
* Transaction Type
* Amount
* Original Account Balance
* New Account Balance
* Destination Account Balance
* New Destination Balance

The transaction is then transformed using the same feature structure used during training.

---

## 🔍 Prediction Classification

The application produces three possible outputs:

### SAFE

The model predicts a legitimate transaction with a relatively low fraud probability.

### SUSPICIOUS

The model does not classify the transaction directly as fraud, but the predicted probability crosses the configured suspicious threshold.

### FRAUD

The trained model predicts the transaction as fraudulent.

The application uses:

```text
Prediction = 1 → FRAUD
Probability > 0.30 → SUSPICIOUS
Otherwise → SAFE
```

> The 0.30 suspicious threshold is a project-specific decision threshold and can be adjusted according to the desired fraud-investigation policy.

---

## 🗄️ Database

SQLite is used to store application data.

The database contains two main tables.

### Users

Stores:

```text
id
username
password
```

### Predictions

Stores:

```text
id
username
step
type
amount
result
probability
```

This allows prediction results to be associated with the logged-in user.

---

## 📁 Project Structure

```text
Fraud-Detection-System/
│
├── app.py
├── train_model.py
├── fraud_dataset.csv
├── database.db
├── requirements.txt
├── README.md
│
├── model/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── features.txt
│
├── templates/
│   ├── home.html
│   ├── about.html
│   ├── login.html
│   ├── register.html
│   └── predict.html
│
└── static/
    ├── css/
    ├── js/
    └── images/
```

---

## ⚙️ Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Fraud-Detection-System.git
```

Move into the project directory:

```bash
cd Fraud-Detection-System
```

---

### Step 2: Create a Virtual Environment

For Python 3.8:

```bash
py -3.8 -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

---

### Step 3: Install Dependencies

Install the required libraries:

```bash
pip install flask pandas numpy joblib scikit-learn xgboost lightgbm imbalanced-learn
```

Or use:

```bash
pip install -r requirements.txt
```

---

## 📦 Requirements

Recommended `requirements.txt`:

```text
Flask
pandas
numpy
joblib
scikit-learn
xgboost
lightgbm
imbalanced-learn
```

---

## 🏋️ Training the Model

Place the dataset in the project root:

```text
fraud_dataset.csv
```

Then execute:

```bash
python train_model.py
```

The training script will:

1. Load the dataset.
2. Encode categorical columns.
3. Split the dataset.
4. Scale the features.
5. Apply SMOTE.
6. Train five classification models.
7. Calculate evaluation metrics.
8. Compare ROC-AUC scores.
9. Select the highest ROC-AUC model.
10. Save the best model.

The following files will be generated:

```text
model/best_model.pkl
model/scaler.pkl
model/features.txt
```

---

## ▶️ Running the Flask Application

After training:

```bash
python app.py
```

The application will normally be available at:

```text
http://127.0.0.1:5000/
```

Open the address in a web browser.

---

## 🔐 User Workflow

```text
Home
  ↓
Register
  ↓
Login
  ↓
Prediction Page
  ↓
Enter Transaction Details
  ↓
Preprocessing
  ↓
Machine Learning Model
  ↓
Fraud Probability
  ↓
SAFE / SUSPICIOUS / FRAUD
  ↓
Prediction Stored in SQLite
```

---

## 🔒 Security Note

This project is intended for educational and research purposes.

For production deployment, additional security mechanisms should be implemented, including:

* Password hashing
* Secure session configuration
* CSRF protection
* Input validation
* Rate limiting
* Environment-based secret keys
* Secure database configuration
* HTTPS
* Production WSGI deployment

The example application currently uses a Flask secret key and stores passwords directly in the SQLite database, so it should **not be deployed publicly without security improvements**.

---

## 📈 Future Enhancements

Potential improvements include:

* Interactive fraud analytics dashboard
* Prediction-history visualization
* ROC and Precision-Recall curves
* Confusion matrix visualization
* SHAP-based model explainability
* Real-time transaction monitoring
* Email/SMS fraud alerts
* API-based transaction prediction
* Role-based authentication
* Password hashing
* Docker deployment
* Cloud deployment
* Model retraining pipeline
* Transaction anomaly detection
* Advanced cost-sensitive threshold optimization

---

## 🛠️ Technologies Used

| Technology       | Purpose                         |
| ---------------- | ------------------------------- |
| Python           | Core programming language       |
| Flask            | Web application framework       |
| Pandas           | Dataset processing              |
| NumPy            | Numerical computation           |
| Scikit-learn     | Machine learning and evaluation |
| XGBoost          | Gradient boosting               |
| LightGBM         | Gradient boosting               |
| imbalanced-learn | SMOTE class balancing           |
| Joblib           | Model serialization             |
| SQLite           | Database                        |
| HTML/CSS         | Web interface                   |

---

## 📜 License

This project is intended for educational, academic, and research purposes. You may modify and extend the implementation according to your project requirements.

---

## 👨‍💻 Author

**Jeeva R**

Machine Learning / Python / Flask Project

---

## ⭐ Project Summary

This project demonstrates an end-to-end machine-learning workflow for financial fraud detection, beginning with data preprocessing and class balancing, followed by multi-model training and evaluation, and concluding with deployment through a Flask-based web application. The system provides probability-based transaction risk assessment and stores prediction results for subsequent analysis.
