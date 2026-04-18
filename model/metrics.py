"""
Fairness metrics computation:
- Demographic Parity Difference
- Equal Opportunity Difference
- Accuracy per group
- Overall accuracy
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix


def group_metrics(y_true, y_pred, sensitive):
    """
    Returns per-group accuracy, TPR, FPR, approval rate.
    sensitive: array-like with values 0 (Female) or 1 (Male)
    """
    results = {}
    groups = {0: "Female", 1: "Male"}

    for g_val, g_name in groups.items():
        mask = sensitive == g_val
        yt = np.array(y_true)[mask]
        yp = np.array(y_pred)[mask]

        if len(yt) == 0:
            continue

        acc = accuracy_score(yt, yp)
        approval_rate = yp.mean()

        # TPR (True Positive Rate / Recall for positive class)
        pos_mask = yt == 1
        tpr = yp[pos_mask].mean() if pos_mask.sum() > 0 else 0.0

        # FPR
        neg_mask = yt == 0
        fpr = yp[neg_mask].mean() if neg_mask.sum() > 0 else 0.0

        results[g_name] = {
            "accuracy": round(acc, 4),
            "approval_rate": round(approval_rate, 4),
            "tpr": round(tpr, 4),
            "fpr": round(fpr, 4),
            "count": int(mask.sum())
        }

    return results


def fairness_metrics(y_true, y_pred, sensitive):
    """
    Returns scalar fairness metrics.
    """
    gm = group_metrics(y_true, y_pred, sensitive)

    female = gm.get("Female", {})
    male = gm.get("Male", {})

    demographic_parity_diff = abs(
        male.get("approval_rate", 0) - female.get("approval_rate", 0)
    )
    equal_opportunity_diff = abs(
        male.get("tpr", 0) - female.get("tpr", 0)
    )
    accuracy_gap = abs(
        male.get("accuracy", 0) - female.get("accuracy", 0)
    )
    overall_acc = accuracy_score(y_true, y_pred)

    return {
        "demographic_parity_diff": round(demographic_parity_diff, 4),
        "equal_opportunity_diff": round(equal_opportunity_diff, 4),
        "accuracy_gap": round(accuracy_gap, 4),
        "overall_accuracy": round(overall_acc, 4),
        "group_metrics": gm
    }
