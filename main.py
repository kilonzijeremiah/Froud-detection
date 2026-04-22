import os
import pickle
import numpy as np
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from pathlib import Path

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
model = None

@app.on_event("startup")
def load_model():
    global model
    # We are looking for exactly 'model.pkl' in the main folder
    model_path = BASE_DIR / "model(1).pkl"
    
    print(f"🔍 DEBUG: Looking for model at {model_path}")
    
    if not model_path.exists():
        print(f"❌ ERROR: 'model.pkl' does not exist at {model_path}")
        return

    try:
        with open(model_path, "rb") as f:
            data = pickle.load(f)
            # If the file contains a list, extract the model
            model = data[0] if isinstance(data, list) else data
        print("✅ SUCCESS: Model loaded and variable is assigned.")
    except Exception as e:
        print(f"❌ ERROR: Failed to read the file: {e}")

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <body style="font-family:sans-serif; text-align:center; padding:50px;">
        <h2>🛡️ Fraud Detection System</h2>
        <form action="/predict" method="post" style="display:inline-block; text-align:left;">
            Amount: <input type="number" step="0.01" name="amount" required><br><br>
            Time: <input type="number" name="time" required><br><br>
            <input type="submit" value="Analyze">
        </form>
    </body>
    """

@app.post("/predict", response_class=HTMLResponse)
def predict(amount: float = Form(...), time: float = Form(...)):
    if model is None:
        return "<h3>Error: Model not loaded on server. Check Render logs.</h3>"
    
    # Prepare features (30 total)
    features = np.zeros(30)
    features[0] = time
    features[-1] = amount
    
    prediction = model.predict(features.reshape(1, -1))[0]
    result = "🚨 FRAUD" if prediction == 1 else "✅ SAFE"
    return f"<h1>Result: {result}</h1><a href='/'>Back</a>"
