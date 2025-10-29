

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import datetime

# ------------------------------
# 1️ Page setup
# ------------------------------
st.set_page_config(page_title="Stock Price Predictor", layout="centered")
st.title(" Stock Price Prediction App")
st.write("Predict future stock prices using Machine Learning (Random Forest Regressor).")

# ------------------------------
# 2️ User Inputs
# ------------------------------
ticker = st.text_input("Enter Stock Symbol (e.g., TCS.NS, INFY.NS, RELIANCE.NS):", "TCS.NS")
start_date = st.date_input("Start Date", datetime.date(2020, 1, 1))
end_date = st.date_input("End Date", datetime.date(2024, 12, 31))

if st.button("Run Prediction"):
    st.info("⏳ Downloading and training model... please wait.")

    # ------------------------------
    # 3️Data Collection
    # ------------------------------
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        st.error("No data found for this symbol or date range.")
        st.stop()

    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    st.success(f" Data downloaded successfully! Total records: {len(data)}")

    # ------------------------------
    # 4️ Feature Engineering
    # ------------------------------
    data['Prev_Close'] = data['Close'].shift(1)
    data['MA_5'] = data['Close'].rolling(5).mean()
    data['MA_10'] = data['Close'].rolling(10).mean()
    data.dropna(inplace=True)

    # ------------------------------
    # 5️ Train-Test Split
    # ------------------------------
    X = data[['Open', 'High', 'Low', 'Volume', 'Prev_Close', 'MA_5', 'MA_10']]
    y = data['Close']
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # ------------------------------
    # 6️ Feature Scaling
    # ------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ------------------------------
    # 7️ Model Training (Random Forest)
    # ------------------------------
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # ------------------------------
    # 8️ Model Evaluation
    # ------------------------------
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    st.subheader(" Model Performance")
    st.write(f"**R² Score:** {r2:.3f}")
    st.write(f"**Mean Absolute Error:** {mae:.2f}")
    st.write(f"**Root Mean Squared Error:** {rmse:.2f}")

    # ------------------------------
    # 9️ Visualization
    # ------------------------------
    st.subheader(" Actual vs Predicted Stock Prices")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y_test.values, label="Actual Prices", color="blue")
    ax.plot(y_pred, label="Predicted Prices", color="red")
    ax.set_title(f"{ticker} Stock Price Prediction (Random Forest)")
    ax.set_xlabel("Days")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    # ------------------------------
    #  Prediction Summary
    # ------------------------------
    future_price = y_pred[-1]
    st.success(f"Predicted Next Close Price for {ticker}: **₹{future_price:.2f}**")
    st.balloons()

st.caption(" Data Science Mini Project | Vardhaman College of Engineering")
