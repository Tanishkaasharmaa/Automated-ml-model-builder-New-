import pandas as pd

def rows_to_df(rows):
    """Convert frontend row-array (list of dicts) to pandas DataFrame safely."""
    if rows is None:
        return None
    df = pd.DataFrame(rows)
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except Exception:
            pass
    return df
