from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import numpy as np
import pickle

# ✅ CREATE APP HERE (top of file)
app = FastAPI()

# Load model
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

# Home route
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h2>Fraud Detection System</h2>
    <form action="/predict" method="post">
        Amount: <input type="text" name="amount"><br><br>
        Time: <input type="text" name="time"><br><br>
        <input type="submit" value="Check Transaction">
    </form>
    """

# Prediction route
@app.post("/predict", response_class=HTMLResponse)
def predict(amount: float = Form(...), time: float = Form(...)):
    
    features = np.zeros(30)
    features[0] = time
    features[-1] = amount

    prediction = model.predict([features])[0]

    prob = model.predict_proba([features])[0][1]
    probability = round(prob * 100, 2)

    result = "🚨 Fraudulent Transaction!" if prediction == 1 else "✅ Safe Transaction"

    return f"""
    <h3>Result: {result}</h3>
    <p>Fraud Probability: {probability}%</p>
    <a href='/'>Go Back</a>
    """
