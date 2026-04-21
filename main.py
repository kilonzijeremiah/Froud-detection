import os
import pickle
import numpy as np
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse
from pathlib import Path

# --- 1. INITIALIZE APP ---
app = FastAPI(title="Fraud Detection API")

# --- 2. SMART PATH LOGIC ---
BASE_DIR = Path(__file__).resolve().parent
model = None

@app.on_event("startup")
def load_model():
    global model
    
    print("--- START DEBUG: FILE LISTING ---")
    file_path = None
    for root, dirs, files in os.walk(BASE_DIR):
        if ".git" in root: continue 
        for file in files:
            if file.lower().endswith(".pkl"):
                file_path = os.path.join(root, file)
                print(f"📄 Found model file: {file_path}")
                break

    if file_path:
        try:
            with open(file_path, "rb") as f:
                loaded_data = pickle.load(f)
            
            # Fix for the 'list' object error:
            if isinstance(loaded_data, list):
                model = loaded_data[0]
                print("✅ SUCCESS: Extracted model from list.")
            else:
                model = loaded_data
                print("✅ SUCCESS: Model loaded directly.")
                
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
    else:
        print("❌ CRITICAL ERROR: No .pkl file found anywhere!")
    print("--- END DEBUG ---")

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
                input { width: 100%; padding: 12px; margin-bottom: 20px; border: 1px solid #ddd; border-radius: 6px; box-sizing: border-box; }
                input[type="submit"] { background-color: #1877f2; color: white; border: none; font-weight: bold; cursor: pointer; }
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
        return "<h3 style='color:red;'>Error: Model not loaded.</h3>"

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
        res_color = "red" if prediction == 1 else "green"

        return f"""
        <div style="text-align: center; margin-top: 100px; font-family: sans-serif;">
            <h1 style="color: {res_color};">{res_text}</h1>
            <p>Probability: {score}</p>
            <a href="/">← Try Another</a>
        </div>
        """
    except Exception as e:
        return f"<p style='color:red;'>Prediction Error: {str(e)}</p>"
