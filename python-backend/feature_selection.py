import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

def do_select_features(session, target_column, n_features=10):
    df = session.get("df_clean") or session.get("df")
    if df is None:
        raise ValueError("No dataset in session.")
    if target_column not in df.columns:
        raise ValueError("target column missing")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    numeric_cols = X.select_dtypes(include=[pd.np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        selected = X.columns.tolist()[:n_features]
        feature_scores = [{"feature": f, "score": 1.0} for f in selected]
        session["selected_features"] = selected
        return {"selected_features": selected, "feature_scores": feature_scores, "n_features_selected": len(selected)}
    scorer = f_regression if (pd.api.types.is_numeric_dtype(y) and y.nunique() > 20) else f_classif
    k = min(n_features, len(numeric_cols))
    try:
        selector = SelectKBest(scorer, k=k)
        selector.fit(X[numeric_cols].fillna(0), y.fillna(0))
        scores_arr = selector.scores_
        pairs = sorted(list(zip(numeric_cols, scores_arr)), key=lambda x: (x[1] if x[1] is not None else 0), reverse=True)
        selected = [p[0] for p in pairs[:k]]
        feature_scores = [{"feature": p[0], "score": float(p[1] if p[1] is not None else 0)} for p in pairs]
    except Exception:
        selected = numeric_cols[:k]
        feature_scores = [{"feature": f, "score": 1.0} for f in selected]
    session["selected_features"] = selected
    return {"selected_features": selected, "feature_scores": feature_scores, "n_features_selected": len(selected)}
