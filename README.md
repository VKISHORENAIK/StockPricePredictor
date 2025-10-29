# StockPricePredictor
# 📈 Stock Price Prediction App

A Streamlit web app that predicts future stock prices using **Random Forest Regressor** and **Yahoo Finance** data.

## 🚀 Features
- Fetches live stock data using `yfinance`
- Calculates moving averages and previous close as features
- Trains a Random Forest model to predict prices
- Shows evaluation metrics (R², MAE, RMSE)
- Displays Actual vs Predicted price graph
- Simple Streamlit interface for any NSE stock symbol

## 🧰 Technologies Used
- Python
- Streamlit
- scikit-learn
- yfinance
- pandas, numpy, matplotlib

## ▶️ How to Run
1. Clone this repo  
   ```bash
   git clone https://github.com/<your-username>/stock-price-predictor.git
   cd stock-price-predictor
## install dependencies
pip install -r requirements.txt
## Run the app
streamlit run app.py
