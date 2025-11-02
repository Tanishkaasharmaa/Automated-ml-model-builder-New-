import pandas as pd
from typing import Dict, Any


def do_detect_task(session: Dict[str, Any], target_column: str):
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
