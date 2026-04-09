import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Fraud Detection App")

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details below:")

# Create inputs for V1 to V27
inputs = []

for i in range(1, 28):
    val = st.number_input(f"V{i}", value=0.0)
    inputs.append(val)

# Amount input
amount = st.number_input("Amount", value=0.0)
inputs.append(amount)

# Time input
time = st.number_input("Time", value=0.0)
inputs.append(time)

# Convert to numpy array
input_data = np.array(inputs).reshape(1, -1)

# Prediction button
if st.button("Predict"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")
