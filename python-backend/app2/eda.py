# app/eda.py
import pandas as pd
import numpy as np

def do_eda(session):
    df = session.get("df")
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
            "mean": float(s["mean"]) if pd.notna(s["mean"]) else None,
            "std": float(s["std"]) if pd.notna(s["std"]) else None,
            "min": float(s["min"]) if pd.notna(s["min"]) else None,
            "max": float(s["max"]) if pd.notna(s["max"]) else None,
        }
    return {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "missing_values": {k: int(v) for k, v in missing_values.items()},
        "missing_percentage": missing_percentage,
        "summary_stats": summary_stats,
    }

def do_validate(session):
    df = session.get("df")
    if df is None:
        raise ValueError("No dataset in session.")
    duplicates = int(df.duplicated().sum())
    total_missing = int(df.isnull().sum().sum())
    columns_with_missing = [c for c in df.columns if df[c].isnull().any()]
    return {"duplicates": duplicates, "total_missing": total_missing, "columns_with_missing": columns_with_missing}

def do_clean(session, strategy="mean"):
    df = session.get("df")
    if df is None:
        raise ValueError("No dataset in session.")
    cleaned_df = df.copy()
    if strategy == "drop":
        cleaned_df = cleaned_df.dropna()
    else:
        num_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for c in num_cols:
            cleaned_df[c].fillna(cleaned_df[c].mean() if strategy == "mean" else cleaned_df[c].median(), inplace=True)
    session["df_clean"] = cleaned_df
    session["data"] = cleaned_df.to_dict(orient="records")
    return {"strategy_used": strategy, "new_shape": cleaned_df.shape}
