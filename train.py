import pandas as pd
import yfinance as yf
import joblib
import os
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

MODEL_PATH = "models/stock_model.pkl"

def fetch_data(ticker):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Day'] = range(len(data))
    return data[['Day', 'Close']]

def train_model(ticker):
    data = fetch_data(ticker)
    X = data[['Day']]
    y = data['Close']
    
    model = LinearRegression()
    model.fit(X, y)
    
    predictions = model.predict(X)
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)
    
    return model, mae, r2

def train_and_save_model(ticker):
    model, mae, r2 = train_model(ticker)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model, mae, r2

if __name__ == "__main__":
    ticker = "HDFCBANK.NS"
    train_and_save_model(ticker)
