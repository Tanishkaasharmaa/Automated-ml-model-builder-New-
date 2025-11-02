import pandas as pd
import numpy as np
from typing import Dict, Any


def do_eda(session: Dict[str, Any]):
    df: pd.DataFrame = session["df"]
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
