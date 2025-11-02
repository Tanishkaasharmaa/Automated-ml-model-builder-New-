import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    r2_score
)
from typing import Dict, Any


def do_evaluate(session: Dict[str, Any]) -> Dict[str, float]:
    """
    Evaluate the trained model on the test set stored in session.
    Supports both regression and classification metrics.

    Returns:
        dict: Dictionary with either regression (rmse, r2)
              or classification (accuracy, precision, recall, f1) metrics.
    """
    pipeline = session.get("pipeline")
    X_test = session.get("X_test")
    y_test = session.get("y_test")

    if pipeline is None or X_test is None or y_test is None:
        raise ValueError("No trained model or test set found.")

    preds = pipeline.predict(X_test)

    # Determine if it's a regression task
    if pd.api.types.is_numeric_dtype(y_test) and y_test.nunique() > 20:
        rmse = mean_squared_error(y_test, preds, squared=False)
        r2 = r2_score(y_test, preds)
        return {"rmse": float(rmse), "r2": float(r2)}

    # Otherwise classification
    acc = accuracy_score(y_test, preds)
    if len(y_test.unique()) <= 2:
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
    else:
        prec = precision_score(y_test, preds, average="macro", zero_division=0)
        rec = recall_score(y_test, preds, average="macro", zero_division=0)
        f1 = f1_score(y_test, preds, average="macro", zero_division=0)

    return {
        "test_accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1)
    }