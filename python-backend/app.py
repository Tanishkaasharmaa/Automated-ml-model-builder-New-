
# python-backend/app.py
import os
import uuid
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from typing import Any, Dict

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

import pandas as pd
import numpy as np
import joblib
import io, base64
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix
)

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Use this filename everywhere (you selected option 2)
TRAINED_PIPELINE_PATH = os.path.join(MODEL_DIR, "trained_pipeline.pkl")

app = FastAPI(title="Automated ML Model Builder - Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSIONS: Dict[str, Dict[str, Any]] = {}

# -------------------------
# Helpers
# -------------------------
def fig_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    return img

def rows_to_df(rows):
    if rows is None:
        return None
    df = pd.DataFrame(rows)
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except Exception:
            pass
    return df

def get_active_df(session):
    return session.get("df_clean") or session.get("df")

# -------------------------
# Main API: POST /ml used by frontend
# -------------------------
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
        if action not in ("evaluate", "predict"):
            session.pop("df_clean", None)
            session.pop("pipeline", None)
            session.pop("model_path", None)

    try:
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
            n = int(params.get("nFeatures") or 10)
            return JSONResponse(do_select_features(session, target, n))
        if action == "train":
            target = params.get("targetColumn")
            task_type = params.get("taskType")
            model_type = params.get("modelType")
            return JSONResponse(do_train(session, target, task_type, model_type, session_id))
        if action == "evaluate":
            return JSONResponse(do_evaluate(session))
        if action == "predict":
            input_data = params.get("inputData")
            return JSONResponse(do_predict(session, input_data))

        return JSONResponse({"error": f"Unknown action {action}"}, status_code=400)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)




@app.post("/clean-data")
async def clean_data(request: Request, 
                     session_id: str = Form(...), 
                     strategy: str = Form(...)):
    """
    Clean dataset using missing-value strategy selected by user.
    Strategies: mean, median, mode, drop.
    """
    try:
        # Get session
        session = SESSIONS.get(session_id)
        if session is None:
            raise ValueError("Invalid session ID")

        # Run cleaning
        result = do_clean(session, strategy)

        return JSONResponse({"status": "success", "cleaning_result": result})

    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": str(e)}, status_code=400
        )


def do_clean(session, strategy="mean"):
    df = get_active_df(session)
    if df is None:
        raise ValueError("No dataset loaded.")

    cleaned_df = df.copy()
    original_rows = df.shape[0]

    # If drop strategy - drop all rows containing ANY NaN
    if strategy == "drop":
        cleaned_df = cleaned_df.dropna()

    else:
        # Clean Numeric Columns
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if cleaned_df[col].isna().sum() > 0:
                if strategy == "mean":
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
                elif strategy == "median":
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
                elif strategy == "mode":
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0])

        # Clean Categorical Columns
        categorical_cols = cleaned_df.select_dtypes(exclude=[np.number]).columns
        for col in categorical_cols:
            if cleaned_df[col].isna().sum() > 0:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0])

    # ⭐⭐ CRITICAL STEP: Ensure absolutely no NaN left ⭐⭐
    if cleaned_df.isna().sum().sum() > 0:
        cleaned_df = cleaned_df.fillna(method="ffill").fillna(method="bfill")

    # Save back to session
    session["df_clean"] = cleaned_df
    print(session.get("df_clean"))
    session["df"] = cleaned_df
    session["data"] = cleaned_df.to_dict(orient="records")

    return {
    "original_shape": list(df.shape),
    "new_shape": list(cleaned_df.shape),
    "rows_removed": original_rows - cleaned_df.shape[0],
    "strategy_used": strategy,
    "cleaned_data": cleaned_df.to_dict(orient="records")   # ADD THIS
}



    
# -------------------------
# EDA / Validate / Clean
# -------------------------
def do_eda(session):
    df = get_active_df(session)
    if df is None:
        raise ValueError("No dataset in session.")

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Move low-unique numeric → categorical
    for col in numeric_columns.copy():
        if df[col].nunique() <= 10:
            numeric_columns.remove(col)
            categorical_columns.append(col)

    # Summary statistics
    summary_stats = {}
    for c in numeric_columns:
        s = df[c].describe()
        summary_stats[c] = {
            "mean": float(s["mean"]) if not pd.isna(s["mean"]) else None,
            "std": float(s["std"]) if not pd.isna(s["std"]) else None,
            "min": float(s["min"]) if not pd.isna(s["min"]) else None,
            "max": float(s["max"]) if not pd.isna(s["max"]) else None,
            "median": float(df[c].median()) if not pd.isna(df[c].median()) else None
        }

    visualizations = {"numeric": {}, "categorical": {}, "bivariate": {}}

    # Color palettes
    pal = sns.color_palette("Set2")
    heat_pal = sns.color_palette("coolwarm")

    # -------------------------------
    # Numeric plots
    # -------------------------------
    for col in numeric_columns:
        # Histogram
        plt.figure()
        sns.histplot(df[col].dropna(), kde=True, color=pal[0])
        plt.title(f"Histogram of {col}")
        visualizations["numeric"][col + "_hist"] = fig_to_base64()

        # Boxplot
        plt.figure()
        sns.boxplot(x=df[col].dropna(), color=pal[1])
        plt.title(f"Boxplot of {col}")
        visualizations["numeric"][col + "_box"] = fig_to_base64()

    # -------------------------------
    # Categorical plots
    # -------------------------------
    for col in categorical_columns:
        plt.figure()
        sns.countplot(x=df[col], palette="Set2")
        plt.xticks(rotation=45)
        plt.title(f"Countplot of {col}")
        visualizations["categorical"][col + "_count"] = fig_to_base64()

    # -------------------------------
    # Heatmap
    # -------------------------------
    if len(numeric_columns) > 1:
        plt.figure(figsize=(8, 6))
        sns.heatmap(df[numeric_columns].corr(),
                    cmap="coolwarm",
                    annot=False,
                    linewidths=0.5,
                    cbar=True)
        plt.title("Correlation Heatmap")
        visualizations["bivariate"]["correlation_heatmap"] = fig_to_base64()

    # -------------------------------
    # Distribution shapes
    # -------------------------------
    distribution_shapes = {}
    def shape(sk):
        if sk > 1: return "Highly right-skewed"
        if sk > 0.5: return "Right-skewed"
        if sk < -1: return "Highly left-skewed"
        if sk < -0.5: return "Left-skewed"
        return "Approximately normal"

    for col in numeric_columns:
        sk = df[col].skew()
        distribution_shapes[col] = {"skewness": float(sk), "shape": shape(sk)}

    missing_values = int(df.isnull().sum().sum())

    # Missing values per column with percentages
    missing_info = {}
    for col in df.columns:
        total = df[col].isnull().sum()
        pct = (total / len(df)) * 100
        missing_info[col] = {
            "missing_count": int(total),
            "missing_percentage": float(round(pct, 2))
        }

        # Missing values table (for UI)
    missing_table = [
        {"column": col,
        "missing_count": missing_info[col]["missing_count"],
        "missing_percentage": missing_info[col]["missing_percentage"]}
        for col in df.columns
    ]

    # Pie chart for missing percentage
    missing_percentages = {col: info["missing_percentage"]
                        for col, info in missing_info.items()
                        if info["missing_percentage"] > 0}

    pie_chart_img = None
    if sum(missing_percentages.values()) > 0:
        plt.figure(figsize=(6, 6))
        plt.pie(
            missing_percentages.values(),
            labels=missing_percentages.keys(),
            autopct="%1.1f%%",
            colors = plt.cm.Set3.colors

        )
        # plt.title("Missing Value Percentage by Column")
        pie_chart_img = fig_to_base64()

    return {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "total_missing_values": missing_values,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "summary_stats": summary_stats,
        "visualizations": visualizations,
        "heatmap": visualizations["bivariate"].get("correlation_heatmap"),
        "distribution_shapes": distribution_shapes,
        "missing_values_per_column": missing_info,
        "missing_values_per_column": missing_info,
        "missing_values_table": missing_table,
        "missing_values_pie_chart": pie_chart_img

    }


def do_validate(session):
    df = get_active_df(session)
    if df is None:
        raise ValueError("No dataset in session.")
    duplicates = int(df.duplicated().sum())
    total_missing = int(df.isnull().sum().sum())
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
        "columns_with_missing": [c for c in df.columns if df[c].isnull().any()],
        "outliers": outliers
    }

# def do_clean(session, strategy="mean"):
#     df = get_active_df(session)
#     if df is None:
#         raise ValueError("No dataset loaded.")
#     original_rows = df.shape[0]
#     cleaned_df = df.copy()
#     if strategy == "drop":
#         cleaned_df = cleaned_df.dropna()
#     else:
#         for col in cleaned_df.select_dtypes(include=[np.number]).columns:
#             if strategy == "mean":
#                 cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
#             elif strategy == "median":
#                 cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
#     session["df_clean"] = cleaned_df
#     session["df"] = cleaned_df
#     session["data"] = cleaned_df.to_dict(orient="records")
#     return {
#         "original_shape": list(df.shape),
#         "new_shape": list(cleaned_df.shape),
#         "rows_removed": int(original_rows - cleaned_df.shape[0]),
#         "strategy_used": strategy
#     }

# -------------------------
# Task detection / feature selection / train / evaluate / predict
# -------------------------
def do_detect_task(session, target_column):
    df = get_active_df(session)
    if df is None:
        raise ValueError("No dataset.")
    if target_column not in df.columns:
        raise ValueError("targetColumn missing or not in dataset")
    y = df[target_column]
    unique = y.nunique()
    if y.dtype.kind in "iuf" and unique > 20:
        task = "regression"
    else:
        task = "classification"
    session["task_type"] = task
    session["target_column"] = target_column
    return {"task_type": task, "target_column": target_column, "unique_values": int(unique), "is_numeric": y.dtype.kind in "iuf"}

def do_select_features(session, target_column, k=10):
    df = get_active_df(session)
    if df is None:
        raise ValueError("No dataset in session.")
    if target_column not in df.columns:
        raise ValueError("target column missing")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        selected = X.columns.tolist()[:k]
        feature_scores = [{"feature": f, "score": 1.0} for f in selected]
        session["selected_features"] = selected
        return {"selected_features": selected, "feature_scores": feature_scores, "n_features_selected": len(selected)}
    scorer = f_regression if (pd.api.types.is_numeric_dtype(y) and y.nunique() > 20) else f_classif
    k = min(k, len(numeric_cols))
    selector = SelectKBest(scorer, k=k)
    selector.fit(X[numeric_cols].fillna(0), y.fillna(0))
    scores_arr = selector.scores_
    pairs = sorted(list(zip(numeric_cols, scores_arr)), key=lambda x: (x[1] if x[1] is not None else 0), reverse=True)
    selected = [p[0] for p in pairs[:k]]
    feature_scores = [{"feature": p[0], "score": float(p[1] if p[1] is not None else 0)} for p in pairs]
    session["selected_features"] = selected
    return {"selected_features": selected, "feature_scores": feature_scores, "n_features_selected": len(selected)}

def do_train(session, target_column, task_type, model_type, session_id):
    df = get_active_df(session)
    if df is None:
        raise ValueError("No dataset in session.")
    if target_column not in df.columns:
        raise ValueError("target column missing")

    selected_features = session.get("selected_features")
    X = df[selected_features] if selected_features else df.drop(columns=[target_column])
    y = df[target_column]

    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X.select_dtypes(exclude=[np.number]).columns.tolist()

    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="mean"), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical)
    ], remainder="drop")

    if task_type == "classification":
        models = {
            "random_forest": RandomForestClassifier(n_estimators=120),
            "logistic_regression": LogisticRegression(max_iter=500),
            "svm": SVC(probability=True),
            "decision_tree": DecisionTreeClassifier(),
            "knn": KNeighborsClassifier()
        }
    else:
        models = {
            "random_forest": RandomForestRegressor(n_estimators=120),
            "linear_regression": LinearRegression(),
            "svm": SVR(),
            "decision_tree": DecisionTreeRegressor(),
            "knn": KNeighborsRegressor()
        }

    model = models.get(model_type)
    if model is None:
        raise ValueError(f"Unsupported model type: {model_type}")

    pipeline = Pipeline([("pre", pre), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    session["pipeline"] = pipeline
    session["X_test"] = X_test.reset_index(drop=True)
    session["y_test"] = y_test.reset_index(drop=True)

    # Persist used feature names so test-model can align uploaded CSV
    used_features = list(X.columns)
    session["used_features"] = used_features
    session["selected_features"] = selected_features or used_features
    session["target_column"] = target_column
    session["task_type"] = task_type

    # Save pipeline file (consistent name)
    joblib.dump(pipeline, TRAINED_PIPELINE_PATH)

    return {
        "model_type": model_type,
        "task_type": task_type,
        "training_complete": True,
        "used_features": used_features
    }

def do_train(session, target_column, task_type, model_type, session_id):

    # Always use cleaned dataset if available
    df = session.get("df_clean") or session.get("df")
    print(session.get("df"))
    if df is None:
        raise ValueError("No dataset in session.")

    if target_column not in df.columns:
        raise ValueError("target column missing")

    # Select features
    selected_features = session.get("selected_features")
    X = df[selected_features] if selected_features else df.drop(columns=[target_column])
    y = df[target_column]

    # Identify column types
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Preprocessing
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="mean"), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), categorical)
    ], remainder="drop")

    # Available Models
    if task_type == "classification":
        models = {
            "random_forest": RandomForestClassifier(n_estimators=120),
            "logistic_regression": LogisticRegression(max_iter=500),
            "svm": SVC(probability=True),
            "decision_tree": DecisionTreeClassifier(),
            "knn": KNeighborsClassifier()
        }
    else:
        models = {
            "random_forest": RandomForestRegressor(n_estimators=120),
            "linear_regression": LinearRegression(),
            "svm": SVR(),
            "decision_tree": DecisionTreeRegressor(),
            "knn": KNeighborsRegressor()
        }

    model = models.get(model_type)
    if model is None:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Build pipeline
    pipeline = Pipeline([
        ("pre", pre),
        ("model", model)
    ])

    # Split & train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipeline.fit(X_train, y_train)

    # Save to session
    session["pipeline"] = pipeline
    session["X_test"] = X_test.reset_index(drop=True)
    session["y_test"] = y_test.reset_index(drop=True)

    # Save features
    used_features = list(X.columns)
    session["used_features"] = used_features
    session["selected_features"] = selected_features or used_features
    session["target_column"] = target_column
    session["task_type"] = task_type

    # Save trained model
    joblib.dump(pipeline, TRAINED_PIPELINE_PATH)

    return {
        "model_type": model_type,
        "task_type": task_type,
        "training_complete": True,
        "used_features": used_features
    }

def do_evaluate(session):
    pipeline = session.get("pipeline")
    X_test = session.get("X_test")
    y_test = session.get("y_test")
    if pipeline is None or X_test is None or y_test is None:
        raise ValueError("No trained model or test set found.")

    preds = pipeline.predict(X_test)

    # regression detection
    if y_test.nunique() > 20 and pd.api.types.is_numeric_dtype(y_test):
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mse = float(mean_squared_error(y_test, preds))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))
        return {"rmse": rmse, "mse": mse, "mae": mae, "r2": r2, "sample_predictions": preds[:10].tolist()}

    acc = float(accuracy_score(y_test, preds))
    prec = float(precision_score(y_test, preds, average="weighted", zero_division=0))
    rec = float(recall_score(y_test, preds, average="weighted", zero_division=0))
    f1 = float(f1_score(y_test, preds, average="weighted", zero_division=0))
    cm = confusion_matrix(y_test, preds).tolist()
    return {"test_accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1, "confusion_matrix": cm, "sample_predictions": preds[:10].tolist()}

def do_predict(session, input_data):
    pipeline = session.get("pipeline")
    if pipeline is None:
        if os.path.exists(TRAINED_PIPELINE_PATH):
            pipeline = joblib.load(TRAINED_PIPELINE_PATH)
            session["pipeline"] = pipeline
        else:
            raise ValueError("No trained model available for prediction")

    if isinstance(input_data, dict):
        df_in = pd.DataFrame([input_data])
    else:
        df_in = pd.DataFrame(input_data)

    # If session has used_features, align columns (extra columns in df_in will be ignored)
    used = session.get("used_features")
    if used:
        df_in = df_in.reindex(columns=used, fill_value=0)

    preds = pipeline.predict(df_in)
    pred_value = preds[0]
    if isinstance(pred_value, (np.generic, np.ndarray)):
        try:
            pred_value = pred_value.item()
        except Exception:
            pred_value = pred_value.tolist()

    confidence = None
    try:
        if hasattr(pipeline.named_steps["model"], "predict_proba"):
            probs = pipeline.predict_proba(df_in)
            confidence = float(probs[0].max())
    except Exception:
        confidence = None

    return {"prediction": pred_value, "confidence": confidence}

# -------------------------
# Test endpoint: fixed & robust
# -------------------------
@app.post("/test-model")
def test_model(
    file: UploadFile = File(...),
    session_id: str = Form(None),
    target_column: str = Form(None)
):
    # Load trained pipeline
    if not os.path.exists(TRAINED_PIPELINE_PATH):
        raise ValueError("Trained pipeline not found. Train a model first.")

    pipeline = joblib.load(TRAINED_PIPELINE_PATH)

    # Read uploaded CSV
    df = pd.read_csv(file.file)

    # Retrieve session (may be empty if frontend didn't pass session_id)
    session = SESSIONS.get(session_id, {}) if session_id else {}

    # Determine target column: prefer session target, then provided form param, then last column
    sess_target = session.get("target_column")
    target = sess_target or target_column or df.columns[-1]

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in uploaded file.")

    # If session saved which features were used for training, use that exact set.
    used_features = session.get("used_features")
    if used_features:
        missing = [c for c in used_features if c not in df.columns]
        if missing:
            # If uploaded file doesn't contain the used features, that's an error.
            raise ValueError(f"Uploaded file missing features the model was trained on: {missing}")
        X = df[used_features].copy()
    else:
        # Fallback: use all columns except target
        X = df.drop(columns=[target])

    y_true = df[target]

    # Predict
    preds = pipeline.predict(X)

    # Convert numpy scalars to native Python types
    preds_list = [p.item() if isinstance(p, (np.generic, np.ndarray)) else p for p in preds]

    # Detect regression/classification
    is_regression = pd.api.types.is_numeric_dtype(y_true) and y_true.nunique() > 20

    if is_regression:
        rmse = float(np.sqrt(mean_squared_error(y_true, preds_list)))
        mse = float(mean_squared_error(y_true, preds_list))
        mae = float(mean_absolute_error(y_true, preds_list))
        r2 = float(r2_score(y_true, preds_list))
        return {
            "task_type": "regression",
            "metrics": {"rmse": rmse, "mse": mse, "mae": mae, "r2": r2},
            "sample_predictions": preds_list[:10]
        }

    # classification metrics
    acc = float(accuracy_score(y_true, preds_list))
    if y_true.nunique() <= 2:
        prec = float(precision_score(y_true, preds_list, zero_division=0))
        rec = float(recall_score(y_true, preds_list, zero_division=0))
        f1 = float(f1_score(y_true, preds_list, zero_division=0))
    else:
        prec = float(precision_score(y_true, preds_list, average="macro", zero_division=0))
        rec = float(recall_score(y_true, preds_list, average="macro", zero_division=0))
        f1 = float(f1_score(y_true, preds_list, average="macro", zero_division=0))

    cm = confusion_matrix(y_true, preds_list).tolist()

    return {
        "task_type": "classification",
        "metrics": {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1},
        "confusion_matrix": cm,
        "sample_predictions": preds_list[:10]
    }

# -------------------------
# Download model
# -------------------------
@app.get("/download_model")
def download_model():
    if not os.path.exists(TRAINED_PIPELINE_PATH):
        raise ValueError("Trained pipeline not found.")
    return FileResponse(TRAINED_PIPELINE_PATH, filename="trained_pipeline.pkl")
