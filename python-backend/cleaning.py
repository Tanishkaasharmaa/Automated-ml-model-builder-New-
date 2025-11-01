import pandas as pd

def do_clean(session, strategy="mean"):
    df: pd.DataFrame = session.get("df")
    if df is None:
        raise ValueError("No dataset in session.")
    original_rows = int(df.shape[0])
    cols = list(df.columns)
    cleaned_df = df.copy()

    if strategy == "drop":
        cleaned_df = cleaned_df.dropna()
    else:
        num_cols = cleaned_df.select_dtypes(include=[pd.np.number]).columns
        for c in num_cols:
            try:
                cleaned_df[c] = pd.to_numeric(cleaned_df[c])
            except Exception:
                pass
            if strategy == "mean":
                cleaned_df[c] = cleaned_df[c].fillna(cleaned_df[c].mean())
            elif strategy == "median":
                cleaned_df[c] = cleaned_df[c].fillna(cleaned_df[c].median())
            else:
                raise ValueError("Unknown cleaning strategy")

    session["df_clean"] = cleaned_df
    session["data"] = cleaned_df.to_dict(orient="records")
    session["df"] = cleaned_df

    return {
        "original_shape": [original_rows, len(cols)],
        "new_shape": [int(cleaned_df.shape[0]), len(cols)],
        "rows_removed": int(original_rows - int(cleaned_df.shape[0])),
        "strategy_used": strategy,
        "cleaned_data": session["data"],
    }
