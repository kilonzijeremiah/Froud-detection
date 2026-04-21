import os
import pickle
import numpy as np
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from pathlib import Path

# --- 1. INITIALIZE ---
app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
model = None

@app.on_event("startup")
def load_model():
    global model
    # We look for the exact name 'model.pkl' in the root folder
    model_path = BASE_DIR / "model.pkl"
    
    print(f"--- 🔍 Checking for model at: {model_path} ---")
    
    if model_path.exists():
        try:
            with open(model_path, "rb") as f:
                raw_data = pickle.load(f)
            
            # If the file contains a list [model], take the first item
            if isinstance(raw_data, list):
                model = raw_data[0]
                print("✅ SUCCESS: Model extracted from list.")
            else:
                model = raw_data
                print("✅ SUCCESS: Model loaded directly.")
        except Exception as e:
            print(f"❌ ERROR: Pickle file is corrupted or invalid: {e}")
    else:
        print(f"❌ ERROR: File 'model.pkl' not found at {model_path}")

# --- 2. ROUTES ---

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <body style="font-family:sans-serif; text-align:center; padding:50px; background:#f4f7f6;">
        <div style="display:inline-block; background:white; padding:30px; border-radius:10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h2>🛡️ Fraud Detector</h2>
            <form action="/predict" method="post">
                <input type="number" step="0.01" name="amount" placeholder="Amount ($)" required style="display:block; width:100%; margin:10px 0; padding:10px;">
                <input type="number" name="time" placeholder="Time (Seconds)" required style="display:block; width:100%; margin:10px 0; padding:10px;">
                <input type="submit" value="Check Transaction" style="width:100%; padding:10px; background:#1877f2; color:white; border:none; cursor:pointer; font-weight:bold;">
            </form>
        </div>
    </body>
    """

@app.post("/predict", response_class=HTMLResponse)
def predict(amount: float = Form(...), time: float = Form(...)):
    if model is None:
        return "<h3 style='color:red;'>Error: Model not loaded on server. Check Render logs.</h3>"
    try:
        # Assuming 30 features: Time is index 0, Amount is index 29
        features = np.zeros(30)
        features[0] = time
        features[-1] = amount
        
        prediction = model.predict(features.reshape(1, -1))[0]
        
        res = "🚨 FRAUD ALERT" if prediction == 1 else "✅ TRANSACTION SAFE"
        color = "#d93025" if prediction == 1 else "#188038"
        
        return f"""
        <div style="text-align:center; font-family:sans-serif; margin-top:100px;">
            <h1 style="color:{color};">{res}</h1>
            <p>Analyzed Amount: ${amount:,.2f}</p>
            <a href="/" style="color:#1877f2; font-weight:bold;">← Try Another</a>
        </div>
        """
    except Exception as e:
        return f"<h3 style='color:red;'>Prediction Error: {e}</h3>"
