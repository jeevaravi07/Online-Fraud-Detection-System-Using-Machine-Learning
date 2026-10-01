# ================================
# IMPORTS
# ================================
import pandas as pd
import numpy as np
import joblib
import time
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from imblearn.over_sampling import SMOTE

# ================================
# CREATE MODEL FOLDER
# ================================
os.makedirs("model", exist_ok=True)

# ================================
# LOAD DATA
# ================================
df = pd.read_csv("fraud_dataset.csv")

print("Dataset Shape:", df.shape)
print("\nColumns:\n", df.columns)

# ================================
# TARGET COLUMN
# ================================
TARGET = "isFraud"

# ================================
# HANDLE CATEGORICAL DATA
# ================================
cat_cols = df.select_dtypes(include=["object"]).columns
print("\nCategorical Columns:", cat_cols)

df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# ================================
# SPLIT FEATURES & TARGET
# ================================
X = df.drop(columns=[TARGET])
y = df[TARGET]

# ================================
# TRAIN TEST SPLIT
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================================
# SCALING
# ================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "model/scaler.pkl")

# ================================
# HANDLE IMBALANCE (SMOTE)
# ================================
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:", np.bincount(y_train))

# ================================
# MODELS
# ================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=500, class_weight="balanced"),

    "Random Forest": RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=120,
        max_depth=6,
        learning_rate=0.1,
        eval_metric='logloss'
    ),

    "LightGBM": LGBMClassifier(
        n_estimators=120,
        learning_rate=0.1
    ),

    "Neural Network": MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=60
    )
}

# ================================
# COST SETTINGS
# ================================
FN_COST = 10000
FP_COST = 500

# ================================
# EVALUATION FUNCTION
# ================================
def evaluate(name, model):
    start = time.time()

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    end = time.time()

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    cost = (fn * FN_COST) + (fp * FP_COST)

    # Precision@K (Top 5%)
    k = int(0.05 * len(y_test))
    top_k = np.argsort(y_prob)[-k:]
    precision_k = y_test.iloc[top_k].mean()

    print("\n==============================")
    print(f"Model: {name}")
    print("==============================")
    print(f"Accuracy   : {acc:.4f}")
    print(f"Precision  : {prec:.4f}")
    print(f"Recall     : {rec:.4f}")
    print(f"F1 Score   : {f1:.4f}")
    print(f"ROC-AUC    : {roc:.4f}")
    print(f"Precision@K: {precision_k:.4f}")
    print(f"Cost       : ₹{cost}")
    print(f"Latency    : {(end-start)*1000:.2f} ms")

    return roc, model

# ================================
# TRAIN & EVALUATE
# ================================
results = []

for name, model in models.items():
    print(f"\n🚀 Training {name}...")
    model.fit(X_train, y_train)

    roc, trained_model = evaluate(name, model)
    results.append((name, roc, trained_model))

# ================================
# BEST MODEL SELECTION
# ================================
best_model = sorted(results, key=lambda x: x[1], reverse=True)[0]

print("\nBEST MODEL:", best_model[0])

# ================================
# SAVE BEST MODEL
# ================================
joblib.dump(best_model[2], "model/best_model.pkl")

# Save feature names
with open("model/features.txt", "w") as f:
    for col in X.columns:
        f.write(col + "\n")

print("\n TRAINING COMPLETED SUCCESSFULLY!")

