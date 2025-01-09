from flask import Flask, render_template, request
from predict import load_model, predict_price

app = Flask(__name__)

# Load the model at startup
try:
    model = load_model()
except FileNotFoundError as e:
    model = None
    print(str(e))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    if not model:
        return render_template('index.html', error="Model not available. Train the model first.")
    
    days = request.args.get('days', default=30, type=int)
    try:
        predicted_price = predict_price(model, days)
        return render_template('result.html', predicted_price=predicted_price, days=days)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
