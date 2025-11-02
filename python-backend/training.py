import os
import uuid
from typing import Dict, Any, List
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from config import MODEL_DIR, TEST_SIZE, RANDOM_STATE


SUPPORTED_CLASSIFIERS = {
    "random_forest": RandomForestClassifier,
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
}


def build_preprocessor(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="mean"))])
    cat_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    preprocessor = ColumnTransformer([("num", numeric_pipeline, numeric_cols), ("cat", cat_pipeline, cat_cols)], remainder="drop")
    return preprocessor
def do_train(session: Dict[str, Any], target_column: str, task_type: str, model_type: str, session_id: str) -> Dict[str, Any]:
    df = session.get("df_clean") or session.get("df")
    if df is None:
        raise ValueError("No dataset in session.")
    if target_column not in df.columns:
        raise ValueError("target column missing")

    # Use selected features if available
    selected_features: List[str] = session.get("selected_features") or []
    if selected_features:
        X = df[selected_features]
    else:
        X = df.drop(columns=[target_column])
    y = df[target_column]

    # Build preprocessor
    preprocessor = build_preprocessor(X)

    # Choose classifier
    if model_type not in SUPPORTED_CLASSIFIERS:
        raise ValueError(f"Unsupported model_type: {model_type}")
    ModelClass = SUPPORTED_CLASSIFIERS[model_type]
    # default params - keep simple
    if model_type == "logistic_regression":
        model = ModelClass(max_iter=500)
    else:
        model = ModelClass()

    pipeline = Pipeline([("pre", preprocessor), ("model", model)])

    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    pipeline.fit(X_train, y_train)

    session["pipeline"] = pipeline
    session["X_test"] = X_test.reset_index(drop=True)
    session["y_test"] = y_test.reset_index(drop=True)

    model_id = str(uuid.uuid4())
    path = os.path.join(MODEL_DIR, f"model_{model_id}.pkl")
    joblib.dump(pipeline, path)
    session["model_path"] = path
    session["model_id"] = model_id

    return {"model_type": model_type, "task_type": task_type, "training_complete": True, "sessionId": session_id, "used_features": selected_features or list(X.columns)}
