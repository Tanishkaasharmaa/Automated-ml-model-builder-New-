import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, r2_score
import joblib
from .config import MODEL_DIR

def do_train(session, target_column, model_type="auto", test_size=0.2, random_state=42):
    df = session.get("df_clean") or session.get("df")
    if df is None:
        raise ValueError("No dataset in session.")
    if target_column not in df.columns:
        raise ValueError("target column missing")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Handle categorical columns
    X = pd.get_dummies(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Determine task type
    task = session.get("task_type")
    if not task:
        if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
            task = "regression"
        else:
            task = "classification"
        session["task_type"] = task

    # Choose model
    if model_type == "auto":
        if task == "regression":
            model = RandomForestRegressor(random_state=random_state)
        else:
            model = RandomForestClassifier(random_state=random_state)
    else:
        # Custom model choice
        if model_type == "linear":
            model = LinearRegression() if task == "regression" else LogisticRegression(max_iter=500)
        elif model_type == "decision_tree":
            model = DecisionTreeRegressor() if task == "regression" else DecisionTreeClassifier()
        elif model_type == "random_forest":
            model = RandomForestRegressor() if task == "regression" else RandomForestClassifier()
        elif model_type == "svm":
            model = SVR() if task == "regression" else SVC()
        else:
            raise ValueError("Unknown model_type")

    # Train
    model.fit(X_train, y_train)

    # Save model
    model_path = os.path.join(MODEL_DIR, f"{target_column}_{task}_model.pkl")
    joblib.dump(model, model_path)
    session["model_path"] = model_path
    session["X_test"] = X_test
    session["y_test"] = y_test
    session["X_train"] = X_train
    session["y_train"] = y_train
    session["model"] = model

    return {"status": "success", "model_type": model_type, "task_type": task, "model_path": model_path}
