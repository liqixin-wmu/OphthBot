from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve


def compute_binary_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    preds = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan")
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "sensitivity": float(tp / (tp + fn + 1e-8)),
        "specificity": float(tn / (tn + fp + 1e-8)),
        "auc": float(auc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def youden_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    if len(np.unique(labels)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(labels, probs)
    best_idx = int(np.argmax(tpr - fpr))
    best_threshold = float(thresholds[best_idx])
    return best_threshold if np.isfinite(best_threshold) else 0.5


def roc_points(labels: np.ndarray, probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan")
    return fpr, tpr, auc
