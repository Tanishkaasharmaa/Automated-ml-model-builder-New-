from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uuid

from utils.session_utils import SESSIONS, rows_to_df
from services.eda_service import do_eda
from services.validate_service import do_validate
from services.clean_service import do_clean
from services.task_service import do_detect_task
from services.feature_service import do_select_features
from services.train_service import do_train
from services.evaluate_service import do_evaluate
from services.predict_service import do_predict

app = FastAPI(title="Automated ML Model Builder - Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
