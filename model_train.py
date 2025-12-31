import yfinance as yf
import pandas as pd
import numpy as np
import os
import joblib  # Correct way to save the scaler
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Fetch stock data
def fetch_data(stock_symbol, start="2020-01-01", end="2025-01-01"):
    try:
        data = yf.download(stock_symbol, start=start, end=end)
        if data.empty:
            raise ValueError(f"Error: No data found for {stock_symbol}. Check the stock symbol or try later.")
        return data[['Close']]
    except Exception as e:
        print(f"Failed to fetch data: {e}")
        return None  # Return None if data isn't available

# Preprocess the data
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    x_train, y_train = [], []
    for i in range(60, len(scaled_data)):  
        x_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return x_train, y_train, scaler

# Build LSTM Model
def build_lstm_model():
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(60, 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train and save model
def train_model(stock_symbol):
    data = fetch_data(stock_symbol)
    if data is None:
        print("No data available. Exiting...")
        return

    x_train, y_train, scaler = preprocess_data(data)

    model = build_lstm_model()
    model.fit(x_train, y_train, epochs=50, batch_size=32)

    # Save model and scaler correctly
    os.makedirs("backend/model", exist_ok=True)
    model.save("backend/model/lstm_model.h5")
    joblib.dump(scaler, "backend/model/scaler.pkl")  # Save the entire scaler object
    print("✅ Model and Scaler saved successfully!")

if __name__ == "__main__":
    stock_symbol = input("Enter Stock Symbol (e.g., AAPL): ")
    train_model(stock_symbol)
