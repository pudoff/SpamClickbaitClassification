"""Метрики на отложенной выборке: общие для train.py и export_ensemble.py."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from app.labels import CLASS_IDS, LABELS


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, *, per_class: bool = True, digits: int | None = 4) -> dict:
    """per_class=True — формат model_meta.json тренера LR: f1_macro, f1_weighted, accuracy и метрики по классам.
    per_class=False — три сводные метрики экспорта ансамбля: accuracy, f1_macro, f1_weighted.
    digits=None — без округления.
    """
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    accuracy = accuracy_score(y_true, y_pred)
    if per_class:
        labels = list(CLASS_IDS)
        f1 = f1_score(y_true, y_pred, average=None, labels=labels)
        precision = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
        metrics = {"f1_macro": f1_macro, "f1_weighted": f1_weighted, "accuracy": accuracy}
        for idx, name in LABELS.items():
            metrics[f"f1_{name}"] = f1[idx]
            metrics[f"precision_{name}"] = precision[idx]
            metrics[f"recall_{name}"] = recall[idx]
    else:
        metrics = {"accuracy": accuracy, "f1_macro": f1_macro, "f1_weighted": f1_weighted}
    return {key: float(value) if digits is None else round(float(value), digits) for key, value in metrics.items()}
