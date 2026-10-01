# ================================
# IMPORTS
# ================================
from flask import Flask, render_template, request, redirect, session
import sqlite3
import joblib
import pandas as pd
import numpy as np
import os
app = Flask(__name__)
app.secret_key = "secret123"

# ================================
# LOAD MODEL
# ================================
model = joblib.load("model/best_model.pkl")
scaler = joblib.load("model/scaler.pkl")

with open("model/features.txt") as f:
    feature_names = [line.strip() for line in f.readlines()]

# ================================
# DATABASE INIT
# ================================
def init_db():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    # USERS TABLE
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            password TEXT
        )
    """)

    # PREDICTION TABLE
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            step REAL,
            type TEXT,
            amount REAL,
            result TEXT,
            probability REAL
        )
    """)

    conn.commit()
    conn.close()

init_db()

# ================================
# HOME
# ================================
@app.route("/")
def home():
    return render_template("home.html")

# ================================
# ABOUT
# ================================
@app.route("/about")
def about():
    return render_template("about.html")

# ================================
# REGISTER
# ================================
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()

        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        conn.close()

        return redirect("/login")

    return render_template("register.html")

# ================================
# LOGIN
# ================================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            session["user"] = username
            return redirect("/predict")
        else:
            return "Invalid Login"

    return render_template("login.html")

# ================================
# LOGOUT
# ================================
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/")

# ================================
# PREDICTION FUNCTION
# ================================
def make_prediction(data):

    input_dict = {
        "step": float(data["step"]),
        "amount": float(data["amount"]),
        "oldbalanceOrg": float(data["oldbalanceOrg"]),
        "newbalanceOrig": float(data["newbalanceOrig"]),
        "oldbalanceDest": float(data["oldbalanceDest"]),
        "newbalanceDest": float(data["newbalanceDest"])
    }

    type_input = data["type"].upper()

    # One-hot encoding
    for col in feature_names:
        if col.startswith("type_"):
            input_dict[col] = 1 if col == f"type_{type_input}" else 0

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if pred == 1:
        result = "FRAUD"
    elif prob > 0.3:
        result = "SUSPICIOUS"
    else:
        result = "SAFE"

    return result, prob

# ================================
# PREDICT PAGE
# ================================
@app.route("/predict", methods=["GET", "POST"])
def predict():

    if "user" not in session:
        return redirect("/login")

    result = None
    probability = None

    if request.method == "POST":

        result, probability = make_prediction(request.form)

        # Save to DB
        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO predictions (username, step, type, amount, result, probability)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            session["user"],
            request.form["step"],
            request.form["type"],
            request.form["amount"],
            result,
            float(probability)
        ))

        conn.commit()
        conn.close()

    return render_template("predict.html", result=result, probability=probability)

# ================================
# RUN
# ================================
if __name__ == "__main__":
    app.run(debug=True)