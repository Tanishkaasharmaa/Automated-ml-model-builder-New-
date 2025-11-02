import pandas as pd
import numpy as np
from typing import List, Dict, Any


def rows_to_df(rows: List[Dict[str, Any]]):
    """Convert frontend row-array (list of dicts) to pandas DataFrame."""
    if rows is None:
        return None
    df = pd.DataFrame(rows)
    # attempt numeric coercion for columns that can be numeric
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except Exception:
            pass
    return df
