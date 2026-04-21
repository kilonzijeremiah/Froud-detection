import os
import pickle
import numpy as np
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse
from pathlib import Path

# --- 1. Initialize App ---
app = FastAPI(title="Fraud Detection API")

# --- 2. Smart Model Loading ---
BASE_DIR = Path(__file__).resolve().parent

# This list checks common naming variations to prevent "File Not Found" errors
possible_paths = [
    BASE_DIR / "model.pkl",
    BASE_DIR / "Model.pkl",
    BASE_DIR / "model" / "model.pkl"
]

model = None

@app.on_event("startup")
def load_model():
    global model
    for path in possible_paths:
        if path.exists():
            try:
                with open(path, "rb") as f:
                    model = pickle.load(f)
                print(f"✅ SUCCESS: Model loaded from {path}")
                return # Exit once found
            except Exception as e:
                print(f"❌ Error loading {path}: {e}")
    
    print(f"❌ CRITICAL: No model file found in any expected location.")

# --- 3. Routes ---

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>Fraud Detection</title>
            <style>
                body { font-family: sans-serif; margin: 50px; background-color: #f8f9fa; text-align: center; }
                .card { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); display: inline-block; text-align: left; }
                input { display: block; width: 100%; padding: 10px; margin: 10px 0; border: 1px solid #ddd; border-radius: 5px; }
                input[type="submit"] { background-color: #28a745; color: white; border: none; cursor: pointer; font-weight: bold; }
                input[type="submit"]:hover { background-color: #218838; }
            </style>
        </head>
        <body>
            <div class="card">
                <h2>🛡️ Fraud Detection System</h2>
                <form action="/predict" method="post">
                    <label>Transaction Amount ($):</label>
                    <input type="number" step="0.01" name="amount" required>
                    <label>Seconds Since First Transaction:</label>
                    <input type="number" name="time" required>
                    <input type="submit" value="Verify Transaction">
                </form>
            </div>
        </body>
    </html>
    """

@app.post("/predict", response_class=HTMLResponse)
def predict(amount: float = Form(...), time: float = Form(...)):
    if model is None:
        return "<h3 style='color:red;'>Error: Model not found. Check server logs for file path errors.</h3>"

    try:
        # Standard 30-feature shape (Time, V1...V28, Amount)
        features = np.zeros(30)
        features[0] = time
        features[-1] = amount
        input_data = features.reshape(1, -1)

        prediction = model.predict(input_data)[0]
        
        try:
            prob = model.predict_proba(input_data)[0][1]
            probability = f"{round(prob * 100, 2)}%"
        except:
            probability = "N/A"

        result_text = "🚨 FRAUD DETECTED" if prediction == 1 else "✅ TRANSACTION SAFE"
        result_color = "red" if prediction == 1 else "green"

        return f"""
        <div style="font-family: sans-serif; margin: 50px; text-align: center;">
            <h2 style="color: {result_color};">{result_text}</h2>
            <p>Probability: {probability}</p>
            <p>Amount: ${amount}</p>
            <br>
            <a href="/">← Go Back</a>
        </div>
        """
    except Exception as e:
        return f"<h3>Processing Error: {e}</h3>"
