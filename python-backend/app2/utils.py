# app/utils.py
import pandas as pd

def rows_to_df(rows):
    if rows is None:
        return None
    df = pd.DataFrame(rows)
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c], errors="ignore")
        except Exception:
            pass
    return df
