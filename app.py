from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load trained model and scaler
try:
    model = load_model("backend/model/lstm_model.h5")
    scaler = joblib.load("backend/model/scaler.pkl")
    print("✅ Model and scaler loaded successfully")
except Exception as e:
    print(f"❌ Error loading model/scaler: {e}")

# Predict future stock prices
def predict_price(stock_symbol):
    try:
        print(f"📊 Fetching data for: {stock_symbol}")
        data = yf.download(stock_symbol, period="70d")["Close"].values

        if len(data) < 60:
            return None, "Not enough data to make a prediction."
        
        print("Raw Data:", data[-10:])  # Debug print
        
        data = scaler.transform(data.reshape(-1, 1))
        print("Scaled Data:", data[-10:])  # Debug print
        
        X_test = np.array([data[-60:]])
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        predicted_price = model.predict(X_test)
        predicted_price = scaler.inverse_transform(predicted_price)[0][0]
        
        print(f"✅ Predicted Price: {predicted_price}")  # Debug print

        return predicted_price, scaler.inverse_transform(data[-10:]).flatten()
    except Exception as e:
        return None, str(e)

# Serve HTML page
@app.route("/")
def home():
    return render_template("index.html")

# API endpoint for predictions
@app.route("/predict", methods=["POST"])
def predict():
    stock_symbol = request.form.get("stock")
    if not stock_symbol:
        return jsonify({"error": "Stock symbol is required"}), 400

    price, past_prices = predict_price(stock_symbol)
    if price is None:
        return jsonify({"error": past_prices}), 400

    # Generate Matplotlib graph
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 11), past_prices, marker='o', linestyle='-', color='b', label='Past Prices')
    plt.axhline(y=price, color='r', linestyle='--', label='Predicted Price')
    plt.xlabel("Days")
    plt.ylabel("Stock Price")
    plt.title(f"Stock Price Prediction for {stock_symbol}")
    plt.legend()

    # Save the plot to a base64-encoded image
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return render_template("result.html", stock=stock_symbol, predicted_price=price, plot_url=plot_url)

if __name__ == "__main__":
    app.run()

