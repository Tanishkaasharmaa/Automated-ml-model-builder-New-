import pandas as pd
from .utils import rows_to_df
from sklearn.preprocessing import OneHotEncoder

def do_predict(session, input_rows):
    model = session.get("model")
    if model is None:
        raise ValueError("No trained model found in session")

    df_input = rows_to_df(input_rows)
    if df_input is None or df_input.empty:
        raise ValueError("No input data provided")

    # Handle categorical columns using same columns as training
    X_train = session.get("X_train")
    if X_train is not None:
        df_input = pd.get_dummies(df_input)
        for c in X_train.columns:
            if c not in df_input.columns:
                df_input[c] = 0
        df_input = df_input[X_train.columns]

    predictions = model.predict(df_input)
    return {"predictions": predictions.tolist()}
