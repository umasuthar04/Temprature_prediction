import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Next-Day Temperature Prediction", layout="centered")
st.title("Predict Next Day Temperature using LSTM")

uploaded_file = st.file_uploader("Upload your temperature CSV (with 'Date' and 'Temp')", type=["csv"])

if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file, parse_dates=["Date"], index_col="Date")
    st.write("Data Preview:", df.tail())

    if "Temp" not in df.columns:
        st.error("CSV must contain a 'Temp' column.")
    else:
        df["Temp"] = pd.to_numeric(df["Temp"], errors="coerce")
        df.dropna(inplace=True)

        try:
            # Load model and scaler
            model = joblib.load("temperature_prediction_model.pkl")       # LSTM model
            scaler = joblib.load("temperature_scaler.pkl")     # Scaler

            # Sequence length used during training
            seq_length = 30

            # Prepare input
            last_sequence = df["Temp"].values[-seq_length:].reshape(-1, 1)
            if len(last_sequence) < seq_length:
                st.error(f"Need at least {seq_length} days of data to predict.")
            else:
                scaled_seq = scaler.transform(last_sequence)
                X_input = np.expand_dims(scaled_seq, axis=0)  # shape: (1, seq_len, 1)

                # Predict next value
                pred_scaled = model.predict(X_input)
                pred_temp = scaler.inverse_transform(pred_scaled)

                st.subheader("Predicted Temperature for Next Day:")
                st.success(f"{pred_temp[0][0]:.2f} °C")


        except Exception as e:
            st.error(f"Prediction failed: {e}")
else:
    st.info("Please upload a CSV file with 'Date' and 'Temp' columns.")
