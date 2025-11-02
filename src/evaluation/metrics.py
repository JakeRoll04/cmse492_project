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
