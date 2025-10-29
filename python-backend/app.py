# python-backend/app.py
import os
import uuid
from typing import Any, Dict
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Automated ML Model Builder - Backend")

# allow all origins during dev; tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# in-memory session store: maps sessionId -> dict with keys: 'data', 'cleaned', 'pipeline', 'model_path', 'X_test','y_test', ...
SESSIONS: Dict[str, Dict[str, Any]] = {}

def rows_to_df(rows):
    """Convert frontend row-array (list of dicts) to pandas DataFrame."""
    if rows is None:
        return None
    df = pd.DataFrame(rows)
    # coerce numeric-ish columns where possible
    for c in df.columns:
        # if all values can be numeric, convert
        try:
            df[c] = pd.to_numeric(df[c], errors="ignore")
        except Exception:
            pass
    return df

@app.post("/ml")
async def ml_endpoint(req: Request):
    payload = await req.json()
    action = payload.get("action")
    rows = payload.get("data")  # optional
    session_id = payload.get("sessionId") or payload.get("session_id") or str(uuid.uuid4())
    params = {k: v for k, v in payload.items() if k not in ("action", "data", "sessionId")}

    # create or get session
    session = SESSIONS.setdefault(session_id, {})
    # if a dataset is provided, update the session data (mimics TypeScript route behavior)
    if rows:
        session["data"] = rows
        session["df"] = rows_to_df(rows)
        # clear any cleaned/pipeline if new raw data provided
        if(action not in ("evaluate","predict")):
            session.pop("df_clean", None)
            session.pop("pipeline", None)
            session.pop("model_path", None)

        session.pop("df_clean", None)
        session.pop("pipeline", None)
        session.pop("model_path", None)

    try:
        if action == "load":
            # return session id and quick info
            df = session.get("df")
            return JSONResponse({"sessionId": session_id, "dataLoaded": bool(df is not None), "rowCount": 0 if df is None else int(df.shape[0])})

        if action == "eda":
            return JSONResponse(do_eda(session))
        if action == "validate":
            return JSONResponse(do_validate(session))
        if action == "clean":
            strategy = params.get("strategy", "mean")
            return JSONResponse(do_clean(session, strategy))
        if action == "detect_task" or action == "detect_task":
            target = params.get("targetColumn") or params.get("target_column")
            return JSONResponse(do_detect_task(session, target))
        if action == "select_features":
            target = params.get("targetColumn") or params.get("target_column")
            n = int(params.get("nFeatures") or params.get("n_features") or 10)
            return JSONResponse(do_select_features(session, target, n))
        if action == "train":
            target = params.get("targetColumn") or params.get("target_column")
            task_type = params.get("taskType") or params.get("task_type") or "classification"
            model_type = params.get("modelType") or params.get("model_type") or "random_forest"
            return JSONResponse(do_train(session, target, task_type, model_type, session_id))
        if action == "evaluate":
            return JSONResponse(do_evaluate(session))
        if action == "predict":
            input_data = params.get("inputData") or params.get("input_data")
            return JSONResponse(do_predict(session, input_data))
        return JSONResponse({"error": f"Unknown action: {action}"}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ---- action implementations (match TS shapes) ----

def do_eda(session):
    df: pd.DataFrame = session.get("df")
    if df is None:
        raise ValueError("No dataset in session.")
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
    missing_values = df.isnull().sum().to_dict()
    missing_percentage = (df.isnull().mean() * 100).round(2).to_dict()
    summary_stats = {}
    for c in numeric_columns:
        s = df[c].describe()
        summary_stats[c] = {
            "mean": None if pd.isna(s.get("mean")) else float(s["mean"]),
            "std": None if pd.isna(s.get("std")) else float(s["std"]),
            "min": None if pd.isna(s.get("min")) else float(s["min"]),
            "max": None if pd.isna(s.get("max")) else float(s["max"]),
            "median": None if pd.isna(df[c].median()) else float(df[c].median()),
            "q1": None if pd.isna(df[c].quantile(0.25)) else float(df[c].quantile(0.25)),
            "q3": None if pd.isna(df[c].quantile(0.75)) else float(df[c].quantile(0.75))
        }
    return {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "missing_values": {k: int(v) for k, v in missing_values.items()},
        "missing_percentage": missing_percentage,
        "summary_stats": summary_stats,
    }

def do_validate(session):
    df: pd.DataFrame = session.get("df")
    if df is None:
        raise ValueError("No dataset in session.")
    duplicates = int(df.duplicated().sum())
    total_missing = int(df.isnull().sum().sum())
    columns_with_missing = [c for c in df.columns if df[c].isnull().any()]
    outliers = {}
    for c in df.select_dtypes(include=[np.number]).columns:
        q1 = df[c].quantile(0.25)
        q3 = df[c].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            outliers[c] = 0
            continue
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        outliers[c] = int(((df[c] < low) | (df[c] > high)).sum())
    return {
        "duplicates": duplicates,
        "total_missing": total_missing,
        "columns_with_missing": columns_with_missing,
        "outliers": outliers
    }

def do_clean(session, strategy="mean"):
    df: pd.DataFrame = session.get("df")
    if df is None:
        raise ValueError("No dataset in session.")
    original_rows = int(df.shape[0])
    cols = list(df.columns)
    cleaned_df = df.copy()

    if strategy == "drop":
        cleaned_df = cleaned_df.dropna()
    else:
        # mean or median
        num_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        if strategy == "mean":
            for c in num_cols:
                cleaned_df[c] = pd.to_numeric(cleaned_df[c], errors="coerce")
                cleaned_df[c].fillna(cleaned_df[c].mean(), inplace=True)
        elif strategy == "median":
            for c in num_cols:
                cleaned_df[c] = pd.to_numeric(cleaned_df[c], errors="coerce")
                cleaned_df[c].fillna(cleaned_df[c].median(), inplace=True)
        else:
            raise ValueError("Unknown cleaning strategy")

    # update session with cleaned data (mimic TypeScript behavior: store cleaned_data back in session)
    session["df_clean"] = cleaned_df
    session["data"] = cleaned_df.to_dict(orient="records")
    session["df"] = cleaned_df

    return {
        "original_shape": [original_rows, len(cols)],
        "new_shape": [int(cleaned_df.shape[0]), len(cols)],
        "rows_removed": int(original_rows - int(cleaned_df.shape[0])),
        "strategy_used": strategy,
        "cleaned_data": session["data"],   # return the array-of-objects (frontend expects this)
    }

def do_detect_task(session, target_column):
    df = session.get("df_clean") or session.get("df")
    if df is None:
        raise ValueError("No dataset in session.")
    if not target_column or target_column not in df.columns:
        raise ValueError("targetColumn missing or not in dataset")
    target = df[target_column].dropna()
    unique = int(target.nunique(dropna=True))
    is_numeric = bool(pd.api.types.is_numeric_dtype(target))
    if is_numeric and unique > 20:
        task = "regression"
    elif is_numeric and unique <= 20:
        task = "classification"
    else:
        task = "classification"
    return {"task_type": task, "target_column": target_column, "unique_values": unique, "is_numeric": is_numeric}

def do_select_features(session, target_column, n_features=10):
    df = session.get("df_clean") or session.get("df")
    if df is None:
        raise ValueError("No dataset in session.")
    if target_column not in df.columns:
        raise ValueError("target column missing")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        selected = X.columns.tolist()[:n_features]
        feature_scores = [{"feature": f, "score": 1.0} for f in selected]
        session["selected_features"] = selected
        return {"selected_features": selected, "feature_scores": feature_scores, "n_features_selected": len(selected)}
    # choose scorer
    scorer = f_regression if (pd.api.types.is_numeric_dtype(y) and y.nunique() > 20) else f_classif
    k = min(n_features, len(numeric_cols))
    try:
        selector = SelectKBest(scorer, k=k)
        selector.fit(X[numeric_cols].fillna(0), y.fillna(0))
        scores_arr = selector.scores_
        pairs = sorted(list(zip(numeric_cols, scores_arr)), key=lambda x: (x[1] if x[1] is not None else 0), reverse=True)
        selected = [p[0] for p in pairs[:k]]
        feature_scores = [{"feature": p[0], "score": float(p[1] if p[1] is not None else 0)} for p in pairs]
    except Exception:
        selected = numeric_cols[:k]
        feature_scores = [{"feature": f, "score": 1.0} for f in selected]
    session["selected_features"] = selected
    return {"selected_features": selected, "feature_scores": feature_scores, "n_features_selected": len(selected)}

# def do_train(session, target_column, task_type, model_type, session_id):
#     df = session.get("df_clean") or session.get("df")
#     if df is None:
#         raise ValueError("No dataset in session.")
#     if target_column not in df.columns:
#         raise ValueError("target column missing")
#     X = df.drop(columns=[target_column])
#     y = df[target_column]
#     numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
#     cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

#     numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="mean"))])
#     cat_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
#     preprocessor = ColumnTransformer([("num", numeric_pipeline, numeric_cols), ("cat", cat_pipeline, cat_cols)], remainder="drop")

#     if task_type == "classification":
#         model = RandomForestClassifier(n_estimators=100, random_state=42)
#     else:
#         model = RandomForestRegressor(n_estimators=100, random_state=42)

#     pipeline = Pipeline([("pre", preprocessor), ("model", model)])

#     # small train/test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     pipeline.fit(X_train, y_train)

#     # save pipeline and test set to session
#     session["pipeline"] = pipeline
#     session["X_test"] = X_test.reset_index(drop=True)
#     session["y_test"] = y_test.reset_index(drop=True)

#     model_id = str(uuid.uuid4())
#     path = os.path.join(MODEL_DIR, f"model_{model_id}.pkl")
#     joblib.dump(pipeline, path)
#     session["model_path"] = path
#     session["model_id"] = model_id

#     return {"model_type": model_type, "task_type": task_type, "training_complete": True, "sessionId": session_id}


def do_train(session, target_column, task_type, model_type, session_id):
    df = session.get("df_clean") or session.get("df")
    if df is None:
        raise ValueError("No dataset in session.")
    if target_column not in df.columns:
        raise ValueError("target column missing")

    #  Use selected features if available
    selected_features = session.get("selected_features")
    if selected_features:
        X = df[selected_features]
    else:
        X = df.drop(columns=[target_column])

    y = df[target_column]

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean"))
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", cat_pipeline, cat_cols)
    ], remainder="drop")

    # Model selection
    X = df.drop(columns=[target_column])
    y = df[target_column]
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="mean"))])
    cat_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))])
    preprocessor = ColumnTransformer([("num", numeric_pipeline, numeric_cols), ("cat", cat_pipeline, cat_cols)], remainder="drop")

    if task_type == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    pipeline = Pipeline([
        ("pre", preprocessor),
        ("model", model)
    ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train
    pipeline.fit(X_train, y_train)

    # Save to session
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

    return {
        "model_type": model_type,
        "task_type": task_type,
        "training_complete": True,
        "sessionId": session_id,
        "used_features": selected_features or list(X.columns)
    }
    return {"model_type": model_type, "task_type": task_type, "training_complete": True, "sessionId": session_id}

def do_evaluate(session):
    pipeline = session.get("pipeline")
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


# def do_predict(session, new_data=None):
#     pipeline = session.get("pipeline")
#     df = session.get("df_clean") or session.get("df")
#     target_column = session.get("target_column")
#     print(session)
#     if pipeline is None:
#         raise ValueError("No trained model found in session.")

#     # Use the same selected features as training
#     selected_features = session.get("selected_features")
#     print(selected_features)
#     if new_data is not None:
#         # Convert to DataFrame if it’s new input
#         X = pd.DataFrame(new_data)
#     else:
#         # Predict on test split or entire dataset
#         if df is None:
#             raise ValueError("No dataset found for prediction.")
#         X = df.drop(columns=[target_column])

#     #  Restrict to selected features if available
#     if selected_features:
#         missing = [f for f in selected_features if f not in X.columns]
#         if missing:
#             raise ValueError(f"Missing features in input: {missing}")
#         X = X[selected_features]

#     # Predict
#     predictions = pipeline.predict(X)

#     # If it’s the test data, store the results
#     if "y_test" in session and len(X) == len(session["y_test"]):
#         session["predictions"] = predictions.tolist()

#     return predictions.tolist()
