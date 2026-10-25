"""
Evaluation Metrics Module for Severely Imbalanced AML Classification
Computes:
- ROC-AUC & PR-AUC (Average Precision)
- Precision, Recall, F1-Score
- Precision@K (P@10, P@25, P@50, P@100)
- Confusion Matrix (TP, FP, TN, FN)
- Plottable ROC and Precision-Recall Curve coordinate series
- Strict NaN / Inf sanitization for JSON API compliance
"""

from typing import Dict, Any, List
import math
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)


def _sanitize_float(val: Any, default: float = 0.0) -> float:
    """Safely converts to float and replaces NaN/Inf with default."""
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return round(f, 4)
    except Exception:
        return default


def compute_precision_at_k(y_true: np.ndarray, y_prob: np.ndarray, k_values: List[int] = [10, 25, 50, 100]) -> Dict[str, float]:
    """
    Computes Precision@K: Fraction of top-K highest risk scored transactions that are true laundering.
    """
    n = len(y_true)
    if n == 0:
        return {f"P@{k}": 0.0 for k in k_values}

    sorted_indices = np.argsort(y_prob)[::-1]
    sorted_labels = y_true[sorted_indices]

    results = {}
    for k in k_values:
        actual_k = min(k, n)
        if actual_k > 0:
            top_k_labels = sorted_labels[:actual_k]
            precision_k = float(np.sum(top_k_labels) / actual_k)
            results[f"P@{k}"] = _sanitize_float(precision_k, 0.0)
        else:
            results[f"P@{k}"] = 0.0
    return results


def evaluate_aml_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Computes comprehensive forensic AML metrics designed for class-imbalanced evaluation.
    All outputs are sanitized against NaN and Inf for JSON compliance.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    has_both_classes = len(np.unique(y_true)) > 1

    if has_both_classes:
        try:
            roc_auc = _sanitize_float(roc_auc_score(y_true, y_prob), 0.5)
        except Exception:
            roc_auc = 0.5
        try:
            pr_auc = _sanitize_float(average_precision_score(y_true, y_prob), 0.0)
        except Exception:
            pr_auc = 0.0
    else:
        roc_auc = 0.5
        pr_auc = 0.0

    prec = _sanitize_float(precision_score(y_true, y_pred, zero_division=0), 0.0)
    rec = _sanitize_float(recall_score(y_true, y_pred, zero_division=0), 0.0)
    f1 = _sanitize_float(f1_score(y_true, y_pred, zero_division=0), 0.0)

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    elif cm.shape == (1, 1):
        if y_true[0] == 0:
            tn, fp, fn, tp = int(cm[0, 0]), 0, 0, 0
        else:
            tn, fp, fn, tp = 0, 0, 0, int(cm[0, 0])
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    p_at_k = compute_precision_at_k(y_true, y_prob, k_values=[10, 25, 50, 100])

    # ROC Curve downsampled for frontend chart
    roc_points = [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}]
    if has_both_classes:
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            indices = np.linspace(0, len(fpr) - 1, min(40, len(fpr))).astype(int)
            roc_points = [
                {"x": _sanitize_float(fpr[i], 0.0), "y": _sanitize_float(tpr[i], 0.0)}
                for i in indices
            ]
        except Exception:
            pass

    # PR Curve: interpolated precision (best precision at recall >= r) on a fixed recall grid,
    # so the plotted shape matches PR-AUC even when almost all thresholds sit at high recall
    pr_points = [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0}]
    if has_both_classes:
        try:
            p_curve, r_curve, _ = precision_recall_curve(y_true, y_prob)
            order = np.argsort(r_curve)
            r_sorted = r_curve[order]
            p_best = np.maximum.accumulate(p_curve[order][::-1])[::-1]
            grid = np.linspace(0.0, 1.0, 41)
            idx = np.clip(np.searchsorted(r_sorted, grid, side="left"), 0, len(r_sorted) - 1)
            pr_points = [
                {"x": _sanitize_float(r, 0.0), "y": _sanitize_float(p_best[i], 0.0)}
                for r, i in zip(grid, idx)
            ]
        except Exception:
            pass

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "precision_at_k": p_at_k,
        "threshold": threshold,
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "total_transactions": int(len(y_true)),
            "actual_laundering_count": int(np.sum(y_true))
        },
        "roc_curve": roc_points,
        "pr_curve": pr_points
    }
