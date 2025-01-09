import joblib
import numpy as np
import os

MODEL_PATH = "models/stock_model.pkl"

def load_model():
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
        return model
    else:
        raise FileNotFoundError("No trained model found. Train the model first!")

def predict_price(model, days_ahead):
    current_day = len(model.coef_)
    future_day = current_day + days_ahead
    prediction = model.predict(np.array([[future_day]]))
    return prediction[0]

if __name__ == "__main__":
    model = load_model()
    days = 30
    predicted_price = predict_price(model, days)
    print(f"Predicted price for {days} days ahead: {predicted_price}")
