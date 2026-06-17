"""
evaluate.py — Metrics, plots, evaluation utilities
Project: ML-based Predictive Maintenance for Rural Borehole Water Pumps
Course:  COEN807 Machine Learning, ABU Zaria
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, average_precision_score,
                              precision_recall_curve, RocCurveDisplay)

PALETTE = {"NORMAL": "#2E75B6", "BROKEN": "#C00000", "RECOVERING": "#ED7D31"}
LABELS  = ["NORMAL", "BROKEN", "RECOVERING"]

def full_report(y_true, y_pred, model_name: str = "Model") -> dict:
    """Print and return classification metrics."""
    print(f"\n{'='*55}")
    print(f"  {model_name} — Classification Report")
    print(f"{'='*55}")
    print(classification_report(y_true, y_pred, target_names=LABELS, zero_division=0))
    report = classification_report(y_true, y_pred,
                                   target_names=LABELS,
                                   output_dict=True, zero_division=0)
    return report

def plot_confusion_matrix(y_true, y_pred, model_name: str,
                          save_path: str = None):
    """Plot and optionally save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABELS, yticklabels=LABELS,
                linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual",    fontsize=11)
    ax.set_title(f"Confusion Matrix — {model_name}",
                 fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

def compare_models_table(results: dict) -> pd.DataFrame:
    """Build a comparison DataFrame from a dict of classification reports."""
    rows = []
    for name, report in results.items():
        rows.append({
            "Model":        name,
            "F1-BROKEN":    round(report["BROKEN"]["f1-score"],    4),
            "F1-macro":     round(report["macro avg"]["f1-score"], 4),
            "Precision-BROKEN": round(report["BROKEN"]["precision"], 4),
            "Recall-BROKEN":    round(report["BROKEN"]["recall"],    4),
        })
    df = pd.DataFrame(rows).sort_values("F1-BROKEN", ascending=False)
    return df
