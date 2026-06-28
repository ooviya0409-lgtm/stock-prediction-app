# my_stock_predictor_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("📈 NIFTY 50 Stock Price Prediction App")

# ---------------- Login Section ----------------
st.subheader("Login")
username = st.text_input("Enter your name")
dob = st.text_input("Enter your date of birth (DD/MM/YYYY)")
bank = st.text_input("Enter your bank details")

if st.button("Login"):
    if username and dob and bank:
        st.success(f"Welcome, {username}! You can now predict stock prices.")

        # ---------------- Stock Symbol Input ----------------
        st.subheader("Enter Stock Symbol")
        symbol_input = st.text_input("Example: ^NSEI for NIFTY 50", value="^NSEI")
        if st.button("Predict"):
            with st.spinner("Fetching data and predicting..."):
                try:
                    # Load real-time stock data
                    df = yf.download(symbol_input, period='6mo', interval='1d')
                    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                    df.dropna(inplace=True)

                    # Feature Engineering
                    df['Range'] = df['High'] - df['Low']
                    features = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Range']]
                    target = df['Open'].shift(-1)
                    features = features[:-1]
                    target = target[:-1]

                    # Preprocessing
                    scaler = MinMaxScaler()
                    X_scaled = scaler.fit_transform(features)
                    X_train, X_test, y_train, y_test = train_test_split(X_scaled, target.values, test_size=0.2, random_state=42)

                    # Model Setup
                    models = {
                        "Linear Regression": LinearRegression(),
                        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                        "Support Vector Regressor": SVR()
                    }

                    results = {}
                    predictions = {}

                    for name, model in models.items():
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        predictions[name] = y_pred
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = mse ** 0.5
                        r2 = r2_score(y_test, y_pred)
                        results[name] = {"RMSE": round(rmse, 2), "R2 Score": round(r2, 2)}

                    # Display Evaluation Table
                    st.subheader("📊 Model Evaluation Table")
                    results_df = pd.DataFrame(results).T
                    st.dataframe(results_df)

                    # Plot Actual vs Random Forest
                    st.subheader("📈 Actual vs Random Forest Predicted Prices (Last 30 Days)")
                    fig, ax = plt.subplots(figsize=(10,5))
                    ax.plot(y_test[:30], label='Actual', marker='o', color='blue')
                    ax.plot(predictions["Random Forest"][:30], label='Predicted (Random Forest)', linestyle='--', marker='x', color='green')
                    ax.set_xlabel("Sample Index")
                    ax.set_ylabel("Price (INR)")
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)

                    # Predict Next Day's Opening Price
                    latest_input = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Range']].iloc[-1:]
                    latest_scaled = scaler.transform(latest_input)

                    today = datetime.now().date()
                    next_day = today + timedelta(days=1)
                    while next_day.weekday() >= 5:
                        next_day += timedelta(days=1)

                    st.subheader(f"📅 Predicted Opening Price for {next_day.strftime('%A, %d %B %Y')}")
                    for name, model in models.items():
                        next_price = model.predict(latest_scaled)
                        if isinstance(next_price, np.ndarray):
                            next_price = next_price[0]
                        st.write(f"{name}: ₹{float(next_price):.2f}")

                except Exception as e:
                    st.error(f"Could not fetch data for symbol '{symbol_input}'. Please check the symbol or your internet connection.")
                    st.write("Error:", e)

    else:
        st.error("Please fill all login fields before proceeding.")