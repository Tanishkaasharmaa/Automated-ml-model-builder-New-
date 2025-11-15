# python-backend/app.py
import os
import uuid
import matplotlib
matplotlib.use("Agg")  # <-- Forces non-GUI backend
import seaborn as sns
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, root_mean_squared_error, r2_score
from sklearn.metrics import (
    root_mean_squared_error,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Automated ML Model Builder - Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSIONS: Dict[str, Dict[str, Any]] = {}

def rows_to_df(rows):
    """Convert frontend row-array (list of dicts) to pandas DataFrame safely."""
    if rows is None:
        return None
    df = pd.DataFrame(rows)
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except Exception:
            pass
    return df

@app.post("/ml")
async def ml_endpoint(req: Request):
    payload = await req.json()
    action = payload.get("action")
    rows = payload.get("data")
    session_id = payload.get("sessionId") or payload.get("session_id") or str(uuid.uuid4())
    params = {k: v for k, v in payload.items() if k not in ("action", "data", "sessionId")}

    session = SESSIONS.setdefault(session_id, {})
    if rows:
        session["data"] = rows
        session["df"] = rows_to_df(rows)
        if action not in ("evaluate","predict"):
            session.pop("df_clean", None)
            session.pop("pipeline", None)
            session.pop("model_path", None)

    try:
        if action == "load":
            df = session.get("df")
            return JSONResponse({"sessionId": session_id, "dataLoaded": bool(df is not None), "rowCount": 0 if df is None else int(df.shape[0])})
        if action == "eda":
            return JSONResponse(do_eda(session))
        if action == "validate":
            return JSONResponse(do_validate(session))
        if action == "clean":
            strategy = params.get("strategy", "mean")
            return JSONResponse(do_clean(session, strategy))
        if action == "detect_task":
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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import io
import base64

def fig_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    return img_base64


def do_eda(session):
    df: pd.DataFrame = session.get("df")
    if df is None:
        raise ValueError("No dataset in session.")
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()

    missing_values = df.isnull().sum().to_dict()
    missing_percentage = (df.isnull().mean() * 100).round(2).to_dict()

    # ===== Summary Stats =====
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

    # ========= Visualizations ========= #
    visualizations = {
        "general": {},
        "numeric": {},
        "categorical": {},
        "bivariate": {}
    }

    # --- General Plots ---
    # Data Types Count
    plt.figure()
    df.dtypes.value_counts().plot(kind="bar")
    plt.title("Column Data Types Count")
    visualizations["general"]["dtype_count"] = fig_to_base64()

    # Missing Value Pie Chart
    total_vals = df.size
    missing_total = df.isnull().sum().sum()
    plt.figure()
    plt.pie([total_vals - missing_total, missing_total], labels=["Non-Missing", "Missing"], autopct='%1.1f%%')
    plt.title("Missing Value Distribution")
    visualizations["general"]["missing_pie"] = fig_to_base64()

    # --- Numeric (Histogram + Boxplot) ---
    for col in numeric_columns:
        plt.figure()
        sns.histplot(df[col].dropna(), kde=True, bins=30)
        plt.title(f"Histogram: {col}")
        visualizations["numeric"][col + "_hist"] = fig_to_base64()

        plt.figure()
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot: {col}")
        visualizations["numeric"][col + "_box"] = fig_to_base64()

    # --- Categorical Countplots ---
    for col in categorical_columns:
        plt.figure(figsize=(6,4))
        df[col].value_counts().head(15).plot(kind="bar")
        plt.title(f"Count Plot: {col}")
        visualizations["categorical"][col + "_count"] = fig_to_base64()

    # --- Bivariate (Correlation Heatmap) ---
    if len(numeric_columns) > 1:
        plt.figure(figsize=(8,6))
        sns.heatmap(df[numeric_columns].corr(), annot=False, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        visualizations["bivariate"]["correlation_heatmap"] = fig_to_base64()

    # --- Bivariate: First Pair Scatter ---
    if len(numeric_columns) >= 2:
        col1, col2 = numeric_columns[:2]
        plt.figure()
        sns.scatterplot(x=df[col1], y=df[col2])
        plt.title(f"Scatter: {col1} vs {col2}")
        visualizations["bivariate"][f"{col1}_vs_{col2}_scatter"] = fig_to_base64()

    return {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "missing_values": {k: int(v) for k, v in missing_values.items()},
        "missing_percentage": missing_percentage,
        "summary_stats": summary_stats,
        "visualizations": visualizations
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
        num_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for c in num_cols:
            try:
                cleaned_df[c] = pd.to_numeric(cleaned_df[c])
            except Exception:
                pass
            if strategy == "mean":
                cleaned_df[c] = cleaned_df[c].fillna(cleaned_df[c].mean())
            elif strategy == "median":
                cleaned_df[c] = cleaned_df[c].fillna(cleaned_df[c].median())
            else:
                raise ValueError("Unknown cleaning strategy")

    session["df_clean"] = cleaned_df
    session["data"] = cleaned_df.to_dict(orient="records")
    session["df"] = cleaned_df

    return {
        "original_shape": [original_rows, len(cols)],
        "new_shape": [int(cleaned_df.shape[0]), len(cols)],
        "rows_removed": int(original_rows - int(cleaned_df.shape[0])),
        "strategy_used": strategy,
        "cleaned_data": session["data"],
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

    session["target_column"] = target_column
    session["task_type"] = task

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

def do_train(session, target_column, task_type, model_type, session_id):
    df = session.get("df_clean") or session.get("df")
    if df is None:
        raise ValueError("No dataset in session.")
    if target_column not in df.columns:
        raise ValueError("target column missing")

    selected_features = session.get("selected_features")
    X = df[selected_features] if selected_features else df.drop(columns=[target_column])
    y = df[target_column]

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="mean"))])
    cat_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), 
                             ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    preprocessor = ColumnTransformer([("num", numeric_pipeline, numeric_cols), ("cat", cat_pipeline, cat_cols)], remainder="drop")

    model = RandomForestClassifier(n_estimators=100, random_state=42) if task_type=="classification" else RandomForestRegressor(n_estimators=100, random_state=42)
    pipeline = Pipeline([("pre", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    session["pipeline"] = pipeline
    session["X_test"] = X_test.reset_index(drop=True)
    session["y_test"] = y_test.reset_index(drop=True)

    model_id = str(uuid.uuid4())
    path = os.path.join(MODEL_DIR, f"model_{model_id}.pkl")
    joblib.dump(pipeline, path)
    session["model_path"] = path
    session["model_id"] = model_id

    joblib.dump(pipeline, "trained_model.pkl")
    return {"model_type": model_type, "task_type": task_type, "training_complete": True, "sessionId": session_id, "used_features": selected_features or list(X.columns)}


def do_evaluate(session):
    pipeline = session.get("pipeline")
    X_test = session.get("X_test")
    y_test = session.get("y_test")

    if pipeline is None or X_test is None or y_test is None:
        raise ValueError("No trained model or test set found.")

    preds = pipeline.predict(X_test)

    # Determine task type
    is_regression = pd.api.types.is_numeric_dtype(y_test) and y_test.nunique() > 20

    if is_regression:
        rmse = root_mean_squared_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        print(rmse)
        print(mse)
        print(mae)
        print(r2)
        return {
            "rmse": float(rmse),
            "mse": float(mse),
            "mae": float(mae),
            "r2": float(r2),
            "sample_predictions": preds[:10].tolist()
        }
    else:
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
            "f1_score": float(f1),
            "sample_predictions": preds[:10].tolist()
        }

def do_predict(session, input_data):
    pipeline = session.get("pipeline")
    if pipeline is None:
        mp = session.get("model_path")
        if mp and os.path.exists(mp):
            pipeline = joblib.load(mp)
            session["pipeline"] = pipeline
        else:
            raise ValueError("No trained model available for prediction")

    if isinstance(input_data, dict):
        df_in = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        df_in = pd.DataFrame(input_data)
    else:
        raise ValueError("Unsupported inputData format")

    preds = pipeline.predict(df_in)
    confidence = None
    try:
        if hasattr(pipeline.named_steps["model"], "predict_proba"):
            probs = pipeline.predict_proba(df_in)
            confidence = float(probs[0].max())
    except Exception:
        confidence = None

    first = preds[0]
    try:
        val = first.item()
    except Exception:
        val = first
    return {"prediction": val, "confidence": confidence}

from fastapi.responses import FileResponse

@app.get("/download_model")
def download_model():
    file_path = "trained_model.pkl"
    return FileResponse(path=file_path, filename="model.pkl", media_type='application/octet-stream')


from fastapi import UploadFile, File
import pandas as pd

@app.post("/predict_from_file")
async def predict_from_file(file: UploadFile = File(...)):
    # Read CSV
    df = pd.read_csv(file.file)

    # Load model
    model = joblib.load("trained_model.pkl")

    # Do prediction
    predictions = model.predict(df)

    # Add predictions to the dataset
    df["Prediction"] = predictions

    # Save result file
    output_path = "prediction_output.csv"
    df.to_csv(output_path, index=False)

    return FileResponse(output_path, filename="predictions.csv", media_type='text/csv')

from fastapi import UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

@app.post("/evaluate_from_file")
async def evaluate_from_file(target_column: str, file: UploadFile = File(...)):
    try:
        # Load pipeline
        pipeline = joblib.load("trained_pipeline.pkl")

        # Load uploaded dataset
        df = pd.read_csv(file.file)

        # Separate features & target
        y_true = df[target_column]
        X = df.drop(columns=[target_column])

        # Predict
        y_pred = pipeline.predict(X)

        # Compute metrics
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)
        matrix = confusion_matrix(y_true, y_pred).tolist()

        return JSONResponse({
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": matrix
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)



from fastapi import UploadFile, File
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
)

@app.post("/test-model")
def test_model(file: UploadFile = File(...), session_id: str = None):
    # Load the saved pipeline directly
    model = joblib.load("trained_model.pkl")  # this is already a pipeline

    # Load the uploaded CSV
    df = pd.read_csv(file.file)

    # Retrieve session info
    session = SESSIONS.get(session_id, {})

    # Get target column from session or assume last column
    target_column = session.get("target_column") or df.columns[-1]

    if target_column in df.columns:
        y = df[target_column]
        X = df.drop(columns=[target_column])
        preds = model.predict(X)

        # Determine task type by model step
        model_step = model.named_steps["model"]
        model_type = model_step.__class__.__name__.lower()

        if "classifier" in model_type:
            metrics = {
                "accuracy": float(accuracy_score(y, preds)),
                "precision": float(precision_score(y, preds, average='weighted')),
                "recall": float(recall_score(y, preds, average='weighted')),
                "f1_score": float(f1_score(y, preds, average='weighted')),
            }
            cm = confusion_matrix(y, preds).tolist()
            return {
                "task_type": "classification",
                "metrics": metrics,
                "confusion_matrix": cm,
                "sample_predictions": preds[:10].tolist()
            }
        else:
            metrics = {
                "mse": float(mean_squared_error(y, preds)),
                "mae": float(mean_absolute_error(y, preds)),
                "r2_score": float(r2_score(y, preds)),
            }
            return {
                "task_type": "regression",
                "metrics": metrics,
                "sample_predictions": preds[:10].tolist()
            }

    else:
        # No target column detected, just return predictions
        preds = model.predict(df)
        return {
            "message": "No target column detected. Returning predictions only.",
            "sample_predictions": preds[:10].tolist()
        }
