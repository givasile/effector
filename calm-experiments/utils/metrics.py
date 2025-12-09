# utils/metrics.py
import numpy as np
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)


def compute_regression_metric(metric_name, y_true, y_pred, sc):
    metric_name = metric_name.lower()
    if metric_name == "r2":
        return r2_score(y_true, y_pred)
    elif metric_name == "mae":
        y_true = sc.inverse_transform(y_true.reshape(-1, 1))
        y_pred = sc.inverse_transform(y_pred.reshape(-1, 1))
        return mean_absolute_error(y_true, y_pred)
    elif metric_name == "rmse":
        y_true = sc.inverse_transform(y_true.reshape(-1, 1))
        y_pred = sc.inverse_transform(y_pred.reshape(-1, 1))
        return np.sqrt(mean_squared_error(y_true, y_pred))
    else:
        raise ValueError(f"Unknown regression metric: {metric_name}")


def compute_classification_metric(metric_name, y_true, y_pred):
    metric_name = metric_name.lower()
    if metric_name == "accuracy":
        return accuracy_score(y_true, y_pred)
    elif metric_name == "balanced_accuracy":
        return balanced_accuracy_score(y_true, y_pred)
    elif metric_name == "f1":
        return f1_score(y_true, y_pred, zero_division=0)
    else:
        raise ValueError(f"Unknown classification metric: {metric_name}")
