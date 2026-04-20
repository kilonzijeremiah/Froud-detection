@app.post("/predict", response_class=HTMLResponse)
def predict(amount: float = Form(...), time: float = Form(...)):
    
    # Create feature vector
    features = np.zeros(30)
    features[0] = time
    features[-1] = amount

    # Prediction
    prediction = model.predict([features])[0]

    # ✅ Add probability calculation
    prob = model.predict_proba([features])[0][1]
    probability = round(prob * 100, 2)

    # Result text
    result = "🚨 Fraudulent Transaction!" if prediction == 1 else "✅ Safe Transaction"

    return f"""
    <h3>Result: {result}</h3>
    <p>Fraud Probability: {probability}%</p>
    <a href='/'>Go Back</a>
    """
