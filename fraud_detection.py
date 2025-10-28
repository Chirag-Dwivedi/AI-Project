import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('fraud_detection_model1.pkl')
st.title("Credit Card Fraud Detection")
st.markdown("Enter the transaction details to predict if it's fraudulent or not.")
st.divider()

transaction_type = st.selectbox("Transaction Type", ["Payment", "Transfer", "Cash Out", "Deposit"])
amount = st.number_input("Transaction Amount", min_value=0.0, value = 1000.0)
oldbalanceOrg = st.number_input("Old Balance of Origin Account", min_value=0.0, value = 10000.0)
newbalanceOrig = st.number_input("New Balance of Origin Account", min_value=0.0, value = 9000.0)
oldbalanceDest = st.number_input("Old Balance of Destination Account", min_value=0.0, value = 0.0)
newbalanceDest = st.number_input("New Balance of Destination Account", min_value=0.0, value = 0.0)

if st.button("Predict"):
    input_data = pd.DataFrame({
        'type': [transaction_type],
        'amount': [amount],
        'oldbalanceOrg': [oldbalanceOrg],
        'newbalanceOrig': [newbalanceOrig],
        'oldbalanceDest': [oldbalanceDest],
        'newbalanceDest': [newbalanceDest]
    })

    # One-hot encode the transaction type
    input_data = pd.get_dummies(input_data, columns=['type'], drop_first=True)

    # Ensure all expected columns are present
    expected_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
                     'type_Cash Out', 'type_Deposit', 'type_Payment', 'type_Transfer']
    for col in expected_cols:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[expected_cols]

    prediction = model.predict(input_data)
    result = "Fraudulent Transaction" if prediction[0] == 1 else "Legitimate Transaction"
    st.success(f"The transaction is predicted to be: {result}")