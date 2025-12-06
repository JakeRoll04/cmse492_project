from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_basic_metrics(y_true, y_pred):
    """
    Compute accuracy, precision, recall, and F1
    for the positive class (>50K income).
    """
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

from typing import Dict, Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def compute_classification_metrics(
    y_true,
    y_pred,
    y_proba=None,
    average: str = "binary",
) -> Dict[str, Any]:
    """
    Compute standard classification metrics.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted class labels.
    y_proba : array-like or None
        Predicted probabilities for the positive class. If None,
        ROC-AUC will be returned as None.
    average : str
        Averaging method for precision/recall/F1 if multi-class.

    Returns
    -------
    metrics : dict
        Dictionary of accuracy, precision, recall, f1, auc, and confusion_matrix.
    """
    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, average=average, zero_division=0)

    if y_proba is not None:
        # assume binary classification; y_proba is probability of class 1
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)
    return metrics

