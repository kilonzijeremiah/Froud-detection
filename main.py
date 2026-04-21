import os
import pickle
import numpy as np
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse
from pathlib import Path

# 1. Initialize App
app = FastAPI(title="Fraud Detection API")

# 2. Robust Model Loading
# This finds the absolute path of the model folder relative to this script
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "model.pkl"

model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        if MODEL_PATH.exists():
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            print("✅ Model loaded successfully.")
        else:
            print(f"❌ Error: Model file not found at {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")

# 3. Routes
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>Fraud Detection</title>
            <style>
                body { font-family: sans-serif; margin: 40px; line-height: 1.6; }
                form { border: 1px solid #ccc; padding: 20px; border-radius: 8px; max-width: 400px; }
                input { margin-bottom: 10px; width: 100%; padding: 8px; }
                input[type="submit"] { background: #007bff; color: white; border: none; cursor: pointer; }
            </style>
        </head>
        <body>
            <h2>🛡️ Fraud Detection System</h2>
            <form action="/predict" method="post">
                <label>Transaction Amount ($):</label>
                <input type="number" step="0.01" name="amount" required placeholder="e.g. 99.99">
                <label>Seconds since first transaction (Time):</label>
                <input type="number" name="time" required placeholder="e.g. 3600">
                <input type="submit" value="Check Transaction">
            </form>
        </body>
    </html>
    """

@app.post("/predict", response_class=HTMLResponse)
def predict(amount: float = Form(...), time: float = Form(...)):
    if model is None:
        return "<h3>Error: Model not loaded on server.</h3>"

    try:
        # Prepare features: 
        # Most fraud models expect 30 features (Time, V1-V28, Amount)
        features = np.zeros(30)
        features[0] = time
        features[-1] = amount
        
        # Reshape for a single prediction
        input_data = features.reshape(1, -1)

        # Get Prediction
        prediction = model.predict(input_data)[0]
        
        # Get Probability (if the model supports it)
        try:
            prob = model.predict_proba(input_data)[0][1]
            probability = f"{round(prob * 100, 2)}%"
        except AttributeError:
            probability = "N/A (Model doesn't support probability)"

        result_text = "🚨 Fraudulent Transaction!" if prediction == 1 else "✅ Safe Transaction"
        color = "red" if prediction == 1 else "green"

        return f"""
        <div style="font-family: sans-serif; margin: 40px;">
            <h3 style="color: {color};">Result: {result_text}</h3>
            <p><strong>Fraud Probability:</strong> {probability}</p>
            <hr>
            <p>Input Amount: ${amount}</p>
            <p>Input Time: {time}s</p>
            <br>
            <a href='/'>← Back to Home</a>
        </div>
        """
    except Exception as e:
        return f"<h3>Processing Error: {str(e)}</h3>"
