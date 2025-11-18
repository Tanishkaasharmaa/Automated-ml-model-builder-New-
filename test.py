import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

# Load model (pipeline)
model = joblib.load("trained_model (12).pkl")

# Load test dataset
df = pd.read_csv("diabetes.csv")

# Detect target column automatically (last column)
target_column = "Outcome"

y_true = df[target_column]
X = df.drop(columns=[target_column])

# Predict
y_pred = model.predict(X)

results = {}

# If classification
if hasattr(model.named_steps["model"], "predict_proba"):
    results["task_type"] = "classification"
    results["accuracy"] = accuracy_score(y_true, y_pred)
    results["precision"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    results["recall"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    results["f1_score"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)

# Else regression
else:
    results["task_type"] = "regression"
    results["mse"] = mean_squared_error(y_true, y_pred)
    results["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
    results["mae"] = mean_absolute_error(y_true, y_pred)
    results["r2"] = r2_score(y_true, y_pred)

print("\n===== Evaluation Results =====")
for k, v in results.items():
    print(f"{k}: {v}")

# Add predictions to CSV
df["Prediction"] = y_pred
df.to_csv("evaluated_output.csv", index=False)

print("\nPredictions saved to evaluated_output.csv")
