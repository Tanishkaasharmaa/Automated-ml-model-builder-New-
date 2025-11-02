import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

# Load your trained pipeline
pipeline = joblib.load("trained_model.pkl")

# Load your test dataset (with actual labels)
data = pd.read_csv("diabetes.csv")

# Separate features and target
y_true = data["Outcome"]        # <-- replace with actual target name
X = data.drop(columns=["Outcome"])

# Predict
y_pred = pipeline.predict(X)

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"✅ Model Accuracy: {accuracy:.4f}")
