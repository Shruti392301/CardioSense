# 🫀 CardioSense – Explainable Cardiovascular Risk Intelligence

CardioSense is a Streamlit-based machine learning dashboard that predicts cardiovascular disease risk using a trained Logistic Regression model.

Unlike a basic prediction system, CardioSense provides **risk explanation, risk tracking, what-if simulation, and model monitoring**.

> **Note:** This project is for educational purposes and is not a medical diagnosis system.

## 🚀 Features

### 🏠 Overview
Provides an introduction to CardioSense and explains the purpose and capabilities of the dashboard.

### 🫀 Risk Assessment
Users enter patient information such as age, blood pressure, cholesterol, maximum heart rate, chest pain type, ECG results, and other health parameters.

The trained ML model generates:
- Risk prediction
- Risk probability
- Assessment result

### 📈 Risk Trajectory
Tracks risk probabilities from multiple assessments during the session.

It provides:
- Previous assessment history
- Risk probability trend
- Prediction history

### 🔍 Explainable AI
Explains **why the model produced a particular prediction** using Logistic Regression coefficients.

It shows:
- Feature importance
- Positive and negative feature contributions
- Individual prediction explanation

### 🧪 What-If Simulator
Allows users to change selected patient values and compare the original prediction with a simulated prediction.

For example:

`Original BP → Modified BP → Compare Risk Probability`

This demonstrates how the trained model responds to different input combinations.

### 🤖 Model Monitoring
Provides basic monitoring of predictions generated during the current session.

It displays:
- Total predictions
- High/low risk distribution
- Average risk probability
- Input statistics
- Prediction history
- CSV export

## 🛠️ Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Logistic Regression
- StandardScaler
- Streamlit
- Joblib

## Run the Project
pip install -r requirements.txt
streamlit run app.py

## 📁 Main Files

```text
Heart-Stroke-Risk-Monitor/
│
├── app.py
├── heart.csv
├── Heart_disease._Prediction.ipynb
├── Heart_prediction.pkl
├── heart_scaler.pkl
├── heart_columns.pkl
├── requirements.txt
└── README.md