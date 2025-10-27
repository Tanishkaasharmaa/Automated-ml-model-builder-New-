# app/routes.py
import uuid
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from .session_manager import SESSIONS
from .utils import rows_to_df
from . import eda, modeling

router = APIRouter()

# @router.post("/ml")
# async def ml_endpoint(req: Request):
#     payload = await req.json()
#     action = payload.get("action")
#     rows = payload.get("data")
#     session_id = payload.get("sessionId") or str(uuid.uuid4())
#     params = {k: v for k, v in payload.items() if k not in ("action", "data", "sessionId")}

#     session = SESSIONS.setdefault(session_id, {})
#     if rows and action in ("eda", "validate", "clean", "train"):
#         session["data"] = rows
#         session["df"] = rows_to_df(rows)
#         session.pop("df_clean", None)
#         session.pop("pipeline", None)
#         session.pop("model_path", None)

#     try:
#         if action == "eda":
#             return JSONResponse(eda.do_eda(session))
#         elif action == "validate":
#             return JSONResponse(eda.do_validate(session))
#         elif action == "clean":
#             return JSONResponse(eda.do_clean(session, params.get("strategy", "mean")))
#         elif action == "detect_task":
#             return JSONResponse(modeling.do_detect_task(session, params.get("targetColumn")))
#         elif action == "select_features":
#             return JSONResponse(modeling.do_select_features(session, params.get("targetColumn"), int(params.get("nFeatures", 10))))
#         elif action == "train":
#             return JSONResponse(modeling.do_train(session, params.get("targetColumn"), params.get("taskType"), params.get("modelType"), session_id))
#         elif action == "evaluate":
#             return JSONResponse(modeling.do_evaluate(session))
#         elif action == "predict":
#             return JSONResponse(modeling.do_predict(session, params.get("inputData")))
#         else:
#             return JSONResponse({"error": f"Unknown action: {action}"}, status_code=400)
#     except Exception as e:
#         return JSONResponse({"error": str(e)}, status_code=500)
@router.post("/ml")
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
    # if rows and action in ("eda", "validate", "clean", "train"):
        session["data"] = rows
        session["df"] = rows_to_df(rows)
        # clear any cleaned/pipeline if new raw data provided
        session.pop("df_clean", None)
        session.pop("pipeline", None)
        session.pop("model_path", None)

    try:
        if action == "load":
            # return session id and quick info
            df = session.get("df")
            return JSONResponse({"sessionId": session_id, "dataLoaded": bool(df is not None), "rowCount": 0 if df is None else int(df.shape[0])})

        if action == "eda":
            return JSONResponse(eda.do_eda(session))
        if action == "validate":
            return JSONResponse(eda.do_validate(session))
        if action == "clean":
            strategy = params.get("strategy", "mean")
            return JSONResponse(eda.do_clean(session, strategy))
        if action == "detect_task" :
            target = params.get("targetColumn") or params.get("target_column")
            return JSONResponse(modeling.do_detect_task(session, target))
        if action == "select_features":
            target = params.get("targetColumn") or params.get("target_column")
            n = int(params.get("nFeatures") or params.get("n_features") or 10)
            return JSONResponse(modeling.do_select_features(session, target, n))
        if action == "train":
            target = params.get("targetColumn") or params.get("target_column")
            task_type = params.get("taskType") or params.get("task_type") or "classification"
            model_type = params.get("modelType") or params.get("model_type") or "random_forest"
            return JSONResponse(modeling.do_train(session, target, task_type, model_type, session_id))
        if action == "evaluate":
            return JSONResponse(modeling.do_evaluate(session))
        if action == "predict":
            input_data = params.get("inputData") or params.get("input_data")
            return JSONResponse(modeling.do_predict(session, input_data))
        return JSONResponse({"error": f"Unknown action: {action}"}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

