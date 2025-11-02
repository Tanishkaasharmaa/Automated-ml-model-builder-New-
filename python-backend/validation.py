import pandas as pd
import numpy as np
from typing import Dict, Any, cast


def do_validate(session: Dict[str, Any]):
    df_raw = session.get("df")
    if df_raw is None:
        raise ValueError("No dataset in session.")
    df = cast(pd.DataFrame, df_raw)
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
