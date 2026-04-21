import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

# Load dataset
df = pd.read_csv("creditcard.csv")

# Features & labels
X = df.drop("Class", axis=1)
y = df["Class"]

# Scale Amount
scaler = StandardScaler()
X["Amount"] = scaler.fit_transform(X[["Amount"]])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved!")
