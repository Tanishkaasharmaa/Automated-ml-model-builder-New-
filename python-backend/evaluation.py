from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error

def do_evaluate(session):
    model = session.get("model")
    X_test = session.get("X_test")
    y_test = session.get("y_test")
    task = session.get("task_type")

    if model is None or X_test is None or y_test is None:
        raise ValueError("Model or test data missing")

    y_pred = model.predict(X_test)

    if task == "classification":
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        metrics = {"accuracy": float(accuracy), "f1_score": float(f1)}
    else:  # regression
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        metrics = {"r2_score": float(r2), "mse": float(mse)}

    session["last_metrics"] = metrics
    return {"metrics": metrics}
