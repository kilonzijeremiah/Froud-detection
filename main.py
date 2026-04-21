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
    
    # --- 🔍 DEBUG BLOCK: This will tell us EXACTLY what files are on Render ---
    print("--- START DEBUG: FILE LISTING ---")
    file_found = False
    for root, dirs, files in os.walk(BASE_DIR):
        # Skip hidden git folders
        if ".git" in root: continue 
        for file in files:
            full_path = os.path.join(root, file)
            print(f"📄 Found file: {full_path}")
            # If we find a pkl file, try to load it regardless of name
            if file.lower().endswith(".pkl"):
                file_found = True
                try:
                    with open(full_path, "rb") as f:
                        loaded_data = pickle.load(f)
# If the pickle file contains a list, grab the first item
if isinstance(loaded_data, list):
    model = loaded_data[0]
else:
    model = loaded_data
                    print(f"✅ SUCCESS: Loaded model from {full_path}")
                except Exception as e:
                    print(f"❌ Error loading {full_path}: {e}")
    
    if not file_found:
        print("❌ CRITICAL ERROR: No .pkl file found anywhere in the project!")
    print("--- END DEBUG: FILE LISTING ---")

# --- 3. UI & ROUTES ---

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>Fraud Detection System</title>
            <style>
                body { font-family: 'Segoe UI', sans-serif; background-color: #f0f2f5; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
                .card { background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 10px 25px rgba(0,0,0,0.1); width: 100%; max-width: 400px; }
                h2 { color: #1c1e21; text-align: center; margin-bottom: 1.5rem; }
                label { font-weight: 600; color: #4b4f56; display: block; margin-bottom: 5px; }
                input { width: 100%; padding: 12px; margin-bottom: 20px; border: 1px solid #dddfe2; border-radius: 6px; box-sizing: border-box; font-size: 16px; }
                input[type="submit"] { background-color: #1877f2; color: white; border: none; font-weight: bold; cursor: pointer; transition: 0.2s; }
                input[type="submit"]:hover { background-color: #166fe5; }
            </style>
        </head>
        <body>
            <div class="card">
                <h2>🛡️ Fraud Detector</h2>
                <form action="/predict" method="post">
                    <label>Transaction Amount ($):</label>
                    <input type="number" step="0.01" name="amount" required placeholder="e.g. 150.50">
                    
                    <label>Time (Seconds):</label>
                    <input type="number" name="time" required placeholder="e.g. 3600">
                    
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
        # Prepare 30 features: [Time, V1...V28, Amount]
        data = np.zeros(30)
        data[0] = time
        data[-1] = amount
        prediction = model.predict(data.reshape(1, -1))[0]
        
        try:
            prob = model.predict_proba(data.reshape(1, -1))[0][1]
            score = f"{round(prob * 100, 2)}%"
        except:
            score = "N/A"

        res_color = "#d93025" if prediction == 1 else "#188038"
        res_text = "🚨 FRAUD ALERT" if prediction == 1 else "✅ TRANSACTION SAFE"

        return f"""
        <div style="font-family: sans-serif; text-align: center; margin-top: 100px;">
            <h1 style="color: {res_color};">{res_text}</h1>
            <p style="font-size: 1.2rem;">Fraud Probability: <strong>{score}</strong></p>
            <p>Amount: ${amount:,.2f} | Time: {time}s</p>
            <br>
            <a href="/" style="color: #1877f2; text-decoration: none; font-weight: bold;">← Try Another</a>
        </div>
        """
    except Exception as e:
        return f"<p style='color:red;'>Prediction Error: {str(e)}</p>"
