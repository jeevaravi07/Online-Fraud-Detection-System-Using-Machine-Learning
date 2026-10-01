# ================================
# IMPORTS
# ================================
import numpy as np
import pandas as pd
import joblib
import os

# ================================
# CHECK MODEL FILES
# ================================
if not os.path.exists("model/best_model.pkl"):
    print(" Model not found. Run train.py first.")
    exit()

# ================================
# LOAD MODEL + SCALER + FEATURES
# ================================
model = joblib.load("model/best_model.pkl")
scaler = joblib.load("model/scaler.pkl")

with open("model/features.txt") as f:
    feature_names = [line.strip() for line in f.readlines()]

# ================================
# USER INPUT
# ================================
print("\n==============================")
print(" FRAUD DETECTION SYSTEM ")
print("==============================\n")

try:
    step = float(input("Step (time): "))
    type_input = input("Type (TRANSFER / CASH_OUT / PAYMENT): ").strip().upper()
    amount = float(input("Amount: "))
    oldbalanceOrg = float(input("Old Balance Orig: "))
    newbalanceOrig = float(input("New Balance Orig: "))
    oldbalanceDest = float(input("Old Balance Dest: "))
    newbalanceDest = float(input("New Balance Dest: "))
except:
    print(" Invalid input. Please enter numeric values correctly.")
    exit()

# ================================
# INPUT VALIDATION
# ================================
valid_types = ["TRANSFER", "CASH_OUT", "PAYMENT"]

if type_input not in valid_types:
    print(" Invalid transaction type.")
    print("Use only:", valid_types)
    exit()

# ================================
# CREATE INPUT DICTIONARY
# ================================
input_dict = {
    "step": step,
    "amount": amount,
    "oldbalanceOrg": oldbalanceOrg,
    "newbalanceOrig": newbalanceOrig,
    "oldbalanceDest": oldbalanceDest,
    "newbalanceDest": newbalanceDest
}

# ================================
# HANDLE ONE-HOT ENCODING
# ================================
for col in feature_names:
    if col.startswith("type_"):
        input_dict[col] = 1 if col == f"type_{type_input}" else 0

# ================================
# CONVERT TO DATAFRAME (NO WARNING)
# ================================
input_df = pd.DataFrame([input_dict])

# Ensure exact column order
input_df = input_df.reindex(columns=feature_names, fill_value=0)

# ================================
# SCALE INPUT
# ================================
input_scaled = scaler.transform(input_df)

# ================================
# PREDICTION
# ================================
prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][1]

# ================================
# OUTPUT RESULT
# ================================
print("\n==============================")
print(" Prediction Result ")
print("==============================")

if prediction == 1:
    print(" FRAUD TRANSACTION DETECTED")
elif probability > 0.3:
    print(" SUSPICIOUS TRANSACTION")
else:
    print(" NON-FRAUD (SAFE TRANSACTION)")

print(f"\nFraud Probability: {probability:.4f}")

# ================================
# EXTRA INTERPRETATION (OPTIONAL)
# ================================
if probability > 0.8:
    print("⚠️ Very High Risk Transaction")
elif probability > 0.5:
    print("⚠️ High Risk Transaction")
elif probability > 0.2:
    print("⚠️ Medium Risk Transaction")
else:
    print("✔ Low Risk Transaction")

print("\n✅ System Completed Successfully")
