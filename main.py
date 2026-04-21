import os
import pickle
import numpy as np
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse
from pathlib import Path

# --- 1. INITIALIZE APP ---
app = FastAPI(title="Fraud Detection API")

# --- 2. MODEL LOADING ---
BASE_DIR = Path(__file__).resolve().parent
model = None

@app.on_event("startup")
def load_model():
    global model
    # We are targeting the exact name found in your logs
    target_file = BASE_DIR / "model (1).pkl"
    
    print(f"--- 🔍 Attempting to load: {target_file} ---")
    
    if target_file.exists():
        try:
            with open(target_file, "rb") as f:
                loaded_data = pickle.load(f)
            
            # Handle the 'list' wrapper if it exists
            if isinstance(loaded_data, list):
                model = loaded_data[0]
                print("✅ SUCCESS: Model extracted from list.")
            else:
                model = loaded_data
                print("✅ SUCCESS: Model loaded directly.")
        except Exception as e:
            print(f"❌ Error during pickle load: {e}")
    else:
        print(f"❌ ERROR: {target_file} not found. Searching for any pkl...")
        # Fallback search if the name changes again
        for file in os.listdir(BASE_DIR):
            if file.endswith(".pkl"):
                print(f"📄 Found alternative: {file}")

# --- 3. UI & ROUTES ---

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>Fraud Detector</title>
            <style>
                body { font-family: sans-serif; background-color: #f0f2f5; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
                .card { background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 10px 25px rgba(0,0,0,0.1); width: 100%; max-width: 400px; }
                h2 { color: #1c1e21; text-align: center; }
                input { width: 100%; padding: 12px; margin: 10px 0; border: 1px solid #ddd; border-radius: 6px; box-sizing: border-box; }
                input[type="submit"] { background-color: #1877f2; color: white; border: none; font-weight: bold; cursor: pointer; margin-top: 10px; }
            </style>
        </head>
        <body>
            <div class="card">
                <h2>🛡️ Fraud Detector</h2>
                <form action="/predict" method="post">
                    <label>Amount ($):</label><input type="number" step="0.01" name="amount" required>
                    <label>Time (Seconds):</label><input type="number" name="time" required>
                    <input type="submit" value="Analyze Transaction">
                </form>
            </div>
        </body>
    </html>
    """

@app.post("/predict", response_class=HTMLResponse)
def predict(amount: float = Form(...), time: float = Form(...)):
    if model is None:
        return "<h3 style='color:red; text-align:center;'>Error: Model not loaded. Check Render logs.</h3>"

    try:
        # Prepare 30 features
        data = np.zeros(30)
        data[0] = time
        data[-1] = amount
        
        prediction = model.predict(data.reshape(1, -1))[0]
        
        try:
            prob = model.predict_proba(data.reshape(1, -1))[0][1]
            score = f"{round(prob * 100, 2)}%"
        except:
            score = "N/A"

        res_text = "🚨 FRAUD ALERT" if prediction == 1 else "✅ TRANSACTION SAFE"
        res_color = "#d93025" if prediction == 1 else "#188038"

        return f"""
        <div style="text-align: center; margin-top: 100px; font-family: sans-serif;">
            <h1 style="color: {res_color};">{res_text}</h1>
            <p style="font-size: 1.2rem;">Fraud Probability: <strong>{score}</strong></p>
            <p>Analyzed Amount: ${amount}</p>
            <br>
            <a href="/" style="text-decoration: none; color: #1877f2; font-weight: bold;">← Try Another</a>
        </div>
        """
    except Exception as e:
        return f"<p style='color:red; text-align:center;'>Prediction Error: {str(e)}</p>"
