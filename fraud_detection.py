from dotenv import load_dotenv

import os
import json

import streamlit as st
import pandas as pd
import joblib
from openai import OpenAI
load_dotenv()

# Load ML model
model = joblib.load('fraud_detection_model2.pkl')

# Set up OpenAI client (uses OPENAI_API_KEY env var)
client = OpenAI()

st.title("Credit Card Fraud Detection")
st.markdown("Enter the transaction details to predict if it's fraudulent or not.")
st.divider()

# ---- Input fields (your columns) ----
transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH OUT", "DEPOSIT"])
amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0)
oldbalanceOrg = st.number_input("Old Balance of Origin Account", min_value=0.0, value=10000.0)
newbalanceOrig = st.number_input("New Balance of Origin Account", min_value=0.0, value=9000.0)
oldbalanceDest = st.number_input("Old Balance of Destination Account", min_value=0.0, value=0.0)
newbalanceDest = st.number_input("New Balance of Destination Account", min_value=0.0, value=0.0)


def generate_ai_explanation(features: dict, prediction: int, fraud_prob: float) -> str:
    """
    Call OpenAI to generate a user-friendly explanation of why
    the transaction was (or wasn't) flagged as fraud.
    """

    # Map label to human-readable meaning
    label_text = "fraudulent" if prediction == 1 else "legitimate"

    # Build a compact payload that the model can reason over
    payload = {
        "transaction_features": features,
        "model_output": {
            "prediction_label": int(prediction),
            "prediction_meaning": label_text,
            "fraud_probability": float(fraud_prob)
        }
    }

    system_prompt = (
        "You are an assistant for a fraud detection system. "
        "Your job is to explain to a non-technical user why a transaction was "
        "predicted as potentially fraudulent or legitimate.\n\n"
        "Rules:\n"
        "- Always mention that this is a prediction, not a guarantee.\n"
        "- Focus on the main risk factors only (2–4 reasons).\n"
        "- Be concise: 2–5 sentences.\n"
        "- Use simple language.\n"
        "- Do not invent data that is not in the JSON.\n"
    )

    user_message = (
        "Here is the transaction and model output as JSON. "
        "Explain the main reasons for this prediction in plain English.\n\n"
        f"```json\n{json.dumps(payload, indent=2)}\n```"
    )

    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",  # or another chat model you prefer :contentReference[oaicite:0]{index=0}
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )

        return completion.choices[0].message.content.strip()

    except Exception as e:
        # Fallback if API call fails
        return f"(Could not generate AI explanation: {e})"


if st.button("Predict"):
    # Create input row for the ML model
    input_data = pd.DataFrame([{
        'type': transaction_type,
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest
    }])

    # Model prediction
    prediction = int(model.predict(input_data)[0])

    # Fraud probability (assumes sklearn-style model with predict_proba)
    try:
        proba = float(model.predict_proba(input_data)[0][1])  # prob of class "1" = fraud
    except AttributeError:
        # If your model doesn't support predict_proba, fall back to 0/1
        proba = 1.0 if prediction == 1 else 0.0

    st.subheader(f"Model Prediction: {prediction} ({'Fraud' if prediction == 1 else 'Legit'})")

    if prediction == 1:
        st.error(f"The transaction is predicted to be Fraudulent (probability ~ {proba:.2%}).")
    else:
        st.success(f"The transaction is predicted to be Legitimate (probability ~ {proba:.2%}).")

    # Build a dict of the raw feature values for the AI explanation
    feature_dict = {
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
        # If later you add 'step', include it here as well
    }

    st.markdown("### AI Explanation")
    explanation = generate_ai_explanation(feature_dict, prediction, proba)
    st.info(explanation)