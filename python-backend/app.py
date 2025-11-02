from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import uuid
import os
import io

from config import UPLOAD_DIR, MODEL_DIR
from session import get_session, set_session
from typing import Any, Dict, cast
from utils import rows_to_df
from eda import do_eda
from validation import do_validate
from cleaning import do_clean
from task_detection import do_detect_task
from feature_selection import do_select_features
from training import do_train
from evaluation import do_evaluate
from prediction import do_predict

app = FastAPI(
    title="AutoML Backend",
    description="Backend API for automated machine learning workflow",
    version="1.0.0"
)

# Allow frontend CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------
#       ROUTES
# ---------------------------

'''@app.post("/upload")
async def upload_file(file: UploadFile):
    """Upload CSV and initialize a session."""
    session_id = str(uuid.uuid4())
    session = {}

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    session["dataset"] = df
    get_session[session_id] = session

    return {"session_id": session_id, "columns": df.columns.tolist()}'''
@app.post("/upload")
async def upload_file(file: UploadFile):
    """Upload CSV and initialize a session."""
    session_id = str(uuid.uuid4())
    session = get_session(session_id)  # correctly get or create session

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    # store dataframe under the expected key 'df' (other modules expect this)
    session["df"] = df
    # keep a JSON-serializable copy as records for convenience
    session["data"] = df.to_dict(orient="records")
    # persist sessions to disk
    set_session(session_id, session)

    return {"session_id": session_id, "columns": df.columns.tolist()}



@app.post("/ml")
async def ml_action(
    action: str = Form(...),
    session_id: str = Form(...),
    payload: str = Form(None)
):
    """
    Central endpoint for performing ML actions.
    action: 'eda', 'validate', 'clean', 'detect_task', 'feature_select',
            'train', 'evaluate', 'predict'
    """
    try:
        session = get_session(session_id)
        if session is None:
            return JSONResponse(status_code=400, content={"error": "Invalid session ID"})

        # Convert payload from JSON if needed
        data = None
        if payload:
            import json
            try:
                data = json.loads(payload)
            except Exception:
                data = payload

        # --- ACTION DISPATCHER ---
        if action == "eda":
            # eda expects the session dict (uses 'df' key)
            return do_eda(session)

        elif action == "validate":
            return do_validate(session)

        elif action == "clean":
            # cleaning expects (session, strategy)
            strategy = None
            if isinstance(data, dict):
                strategy = data.get("strategy")
            result = do_clean(session, strategy or "mean")
            # cleaned dataframe is stored in session by do_clean; persist
            set_session(session_id, session)
            return result

        elif action == "detect_task":
            target = data.get("target") if isinstance(data, dict) else None
            if not isinstance(target, str):
                return JSONResponse(status_code=400, content={"error": "Missing required 'target' parameter for task detection"})
            result = do_detect_task(session, target)
            # store detected task type in session
            session["task_type"] = result.get("task_type") if isinstance(result, dict) else result
            set_session(session_id, session)
            return result

        elif action == "feature_select":
            target = data.get("target") if isinstance(data, dict) else None
            if not isinstance(target, str):
                return JSONResponse(status_code=400, content={"error": "Missing required 'target' parameter for feature selection"})
            try:
                n_features = int(data.get("n_features", 10)) if isinstance(data, dict) else 10
            except Exception:
                n_features = 10
            result = do_select_features(session, target, n_features)
            set_session(session_id, session)
            return result

        elif action == "train":
            target_col = data.get("target") if isinstance(data, dict) else None
            if not isinstance(target_col, str):
                return JSONResponse(status_code=400, content={"error": "Missing required 'target' parameter for training"})
            model_type = data.get("model_type", "random_forest") if isinstance(data, dict) else "random_forest"
            task_type = session.get("task_type") or (data.get("task_type") if isinstance(data, dict) else None)
            if not isinstance(task_type, str):
                # try to auto-detect task type
                try:
                    det = do_detect_task(session, target_col)
                    task_type = det.get("task_type") if isinstance(det, dict) else det
                    session["task_type"] = task_type
                except Exception as e:
                    return JSONResponse(status_code=400, content={"error": f"task_type missing and auto-detect failed: {e}"})
            # ensure task_type is a string for the train API
            task_type = str(task_type)
            metrics = do_train(session, target_col, task_type, model_type, session_id)
            set_session(session_id, session)
            return {"message": f"{model_type} trained", "metrics": metrics}

        elif action == "evaluate":
            result = do_evaluate(session)
            return result

        elif action == "predict":
            # Expect payload to be either a dict or list of records
            input_data = data if isinstance(data, (dict, list)) else None
            if input_data is None:
                return JSONResponse(status_code=400, content={"error": "Missing or invalid input data for prediction"})
            # cast to expected typing for static checkers: do_predict accepts Dict or list at runtime
            if isinstance(input_data, dict):
                result = do_predict(session, cast(Dict[str, Any], input_data))
            else:
                result = do_predict(session, cast(Any, input_data))
            return result

        else:
            return JSONResponse(status_code=400, content={"error": "Unknown action"})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/")
def root():
    return {"message": "AutoML Backend is running"}