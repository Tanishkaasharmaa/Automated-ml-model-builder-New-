# app/modeling.py
import os, uuid, joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, precision_score, recall_score

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def do_detect_task(session, target_column):
    df = session.get("df_clean") or session.get("df")
    print(df)
    if df is None:
        raise ValueError("No dataset in session.")
    target = df[target_column]
    unique = int(target.nunique())
    is_numeric = pd.api.types.is_numeric_dtype(target)
    task = "regression" if (is_numeric and unique > 20) else "classification"
    return {"task_type": task, "target_column": target_column, "unique_values": unique}

def do_select_features(session, target_column, n_features=10):
    df = session.get("df_clean") or session.get("df")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    scorer = f_regression if y.nunique() > 20 else f_classif
    k = min(n_features, len(numeric_cols))
    selector = SelectKBest(scorer, k=k)
    selector.fit(X[numeric_cols].fillna(0), y.fillna(0))
    scores = selector.scores_
    selected = [numeric_cols[i] for i in np.argsort(scores)[-k:][::-1]]
    session["selected_features"] = selected
    return {"selected_features": selected}

# def do_train(session, target_column, task_type, model_type, session_id):
#     df = session.get("df_clean") or session.get("df")
#     X = df.drop(columns=[target_column])
#     y = df[target_column]
#     num_cols = X.select_dtypes(include=[np.number]).columns
#     cat_cols = X.select_dtypes(exclude=[np.number]).columns
#     numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="mean"))])
#     cat_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
#     preprocessor = ColumnTransformer([("num", numeric_pipeline, num_cols), ("cat", cat_pipeline, cat_cols)])
#     model = RandomForestClassifier() if task_type == "classification" else RandomForestRegressor()
#     pipeline = Pipeline([("pre", preprocessor), ("model", model)])
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#     pipeline.fit(X_train, y_train)
#     session["pipeline"] = pipeline
#     session["X_test"], session["y_test"] = X_test, y_test
#     model_id = str(uuid.uuid4())
#     path = os.path.join(MODEL_DIR, f"model_{model_id}.pkl")
#     joblib.dump(pipeline, path)
#     session["model_path"] = path
#     print(session)
#     return {"model_saved": True, "model_id": model_id}
def do_train(session, target_column, task_type, model_type, session_id):
    df = session.get("df_clean") or session.get("df")
    if df is None:
        raise ValueError("No dataset in session.")
    if target_column not in df.columns:
        raise ValueError("target column missing")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="mean"))])
    cat_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    preprocessor = ColumnTransformer([("num", numeric_pipeline, numeric_cols), ("cat", cat_pipeline, cat_cols)], remainder="drop")

    if task_type == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    pipeline = Pipeline([("pre", preprocessor), ("model", model)])

    # small train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    # save pipeline and test set to session
    session["pipeline"] = pipeline
    session["X_test"] = X_test.reset_index(drop=True)
    session["y_test"] = y_test.reset_index(drop=True)

    model_id = str(uuid.uuid4())
    path = os.path.join(MODEL_DIR, f"model_{model_id}.pkl")
    joblib.dump(pipeline, path)
    session["model_path"] = path
    session["model_id"] = model_id

    return {"model_type": model_type, "task_type": task_type, "training_complete": True, "sessionId": session_id}

# def do_evaluate(session):
#     # pipeline = session["pipeline"]
#     pipeline = session.get("pipeline")
#     # print(session.get("pipeline"))
#     # if pipeline is None:
#     #     model_path = session.get("model_path")
#     #     if model_path and os.path.exists(model_path):
#     #         pipeline = joblib.load(model_path)
#     #         session["pipeline"] = pipeline
#     #     else:
#     #         raise ValueError("No trained model found. Please train a model first.")
#     print(session)
#     X_test, y_test = session["X_test"], session["y_test"]
#     preds = pipeline.predict(X_test)
#     if y_test.nunique() > 20:
#         return {"rmse": mean_squared_error(y_test, preds, squared=False), "r2": r2_score(y_test, preds)}
#     else:
#         return {"accuracy": accuracy_score(y_test, preds), "f1": f1_score(y_test, preds, average="macro")}
def do_evaluate(session):
    pipeline = session.get("pipeline")
    print(session)
    X_test = session.get("X_test")
    y_test = session.get("y_test")
    if pipeline is None or X_test is None or y_test is None:
        raise ValueError("No trained model or test set found.")
    preds = pipeline.predict(X_test)
    # classification vs regression decision (mirrors TypeScript heuristics)
    if pd.api.types.is_numeric_dtype(y_test) and y_test.nunique() > 20:
        rmse = mean_squared_error(y_test, preds, squared=False)
        r2 = r2_score(y_test, preds)
        return {"rmse": float(rmse), "r2": float(r2)}
    else:
        acc = accuracy_score(y_test, preds)
        # handle binary vs multiclass
        if len(y_test.unique()) <= 2:
            prec = precision_score(y_test, preds, zero_division=0)
            rec = recall_score(y_test, preds, zero_division=0)
            f1 = f1_score(y_test, preds, zero_division=0)
        else:
            prec = precision_score(y_test, preds, average="macro", zero_division=0)
            rec = recall_score(y_test, preds, average="macro", zero_division=0)
            f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        return {"test_accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1_score": float(f1)}

# def do_predict(session, input_data):
#     pipeline = session.get("pipeline")
#     print(session) 
#     if pipeline is None:
#         model_path = session.get("model_path")
#         if model_path and os.path.exists(model_path):
#             pipeline = joblib.load(model_path)
#             session["pipeline"] = pipeline
#         else:
#             raise ValueError("No trained model available for prediction. Train a model first.")
#     # pipeline = session.get("pipeline")
#     df_in = pd.DataFrame([input_data]) if isinstance(input_data, dict) else pd.DataFrame(input_data)
#     preds = pipeline.predict(df_in)
#     return {"prediction": preds.tolist()}

def do_predict(session, input_data):
    pipeline = session.get("pipeline")
    if pipeline is None:
        # attempt to load saved model if model_path exists
        mp = session.get("model_path")
        if mp and os.path.exists(mp):
            pipeline = joblib.load(mp)
            session["pipeline"] = pipeline
        else:
            raise ValueError("No trained model available for prediction")

    # accept a single dict, or list of dicts
    if isinstance(input_data, dict):
        df_in = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        df_in = pd.DataFrame(input_data)
    else:
        raise ValueError("Unsupported inputData format")

    preds = pipeline.predict(df_in)
    confidence = None
    try:
        # if classifier has predict_proba
        if hasattr(pipeline.named_steps["model"], "predict_proba"):
            probs = pipeline.predict_proba(df_in)
            # return confidence for first row (max prob)
            confidence = float(probs[0].max())
    except Exception:
        confidence = None

    # if multiple outputs, return first
    first = preds[0]
    try:
        # convert numpy types to python native
        val = first.item()
    except Exception:
        val = first
    return {"prediction": val, "confidence": confidence}
