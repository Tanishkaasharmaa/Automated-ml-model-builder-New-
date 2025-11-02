import os
import joblib
import pandas as pd
from typing import Dict, Any


def do_predict(session: Dict[str, Any], input_data: Dict[str, Any]):
    """
    Make predictions using the trained pipeline.
    If the pipeline isn't in memory, it will reload from saved model_path.

    Args:
        session: Session dict containing model info.
        input_data: Dict or list of records to predict.

    Returns:
        dict: { "prediction": value, "confidence": float or None }
    """
    pipeline = session.get("pipeline")

    # Load pipeline from saved file if needed
    if pipeline is None:
        mp = session.get("model_path")
        if mp and os.path.exists(mp):
            pipeline = joblib.load(mp)
            session["pipeline"] = pipeline
        else:
            raise ValueError("No trained model available for prediction")

    # Convert input into DataFrame
    if isinstance(input_data, dict):
        df_in = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        df_in = pd.DataFrame(input_data)
    else:
        raise ValueError("Unsupported inputData format (must be dict or list)")

    preds = pipeline.predict(df_in)

    # Confidence score (if model supports predict_proba)
    confidence = None
    try:
        model = pipeline.named_steps.get("model")
        if model and hasattr(model, "predict_proba"):
            probs = model.predict_proba(df_in)
            confidence = float(probs[0].max())
    except Exception:
        confidence = None

    # Convert numpy type to Python native
    first = preds[0]
    try:
        val = first.item()
    except Exception:
        val = first

    return {"prediction": val, "confidence": confidence}