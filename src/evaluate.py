"""
evaluate.py
-----------
Evaluation utilities for the sickle cell detection pipeline.

Computes accuracy, AUC-ROC, F1, sensitivity, specificity, precision,
confusion matrix, and ROC curve for a trained binary classifier.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
    classification_report
)


def get_predictions(model, loader, device):
    """
    Run inference on a DataLoader and collect predictions, probabilities, and labels.

    Parameters
    ----------
    model  : nn.Module — must be already loaded with trained weights
    loader : DataLoader
    device : torch.device

    Returns
    -------
    tuple: (all_preds, all_probs, all_labels) — all numpy arrays
    """
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images  = images.to(device)
            outputs = model(images)
            probs   = torch.sigmoid(outputs).cpu().squeeze().numpy()
            preds   = (probs >= 0.5).astype(int)

            all_probs.extend(probs  if probs.ndim  > 0 else [probs.item()])
            all_preds.extend(preds  if preds.ndim  > 0 else [preds.item()])
            all_labels.extend(labels.numpy())

    return (
        np.array(all_preds),
        np.array(all_probs),
        np.array(all_labels)
    )


def compute_metrics(all_labels, all_preds, all_probs) -> dict:
    """
    Compute all evaluation metrics for binary classification.

    Parameters
    ----------
    all_labels : np.array — ground truth labels (0 or 1)
    all_preds  : np.array — predicted labels (0 or 1)
    all_probs  : np.array — predicted probabilities for positive class

    Returns
    -------
    dict with keys: accuracy, auc, f1, sensitivity, specificity, precision
    """
    accuracy    = (all_preds == all_labels).mean()
    auc         = roc_auc_score(all_labels, all_probs)
    f1          = f1_score(all_labels, all_preds)
    sensitivity = recall_score(all_labels, all_preds)
    specificity = recall_score(all_labels, all_preds, pos_label=0)
    precision   = precision_score(all_labels, all_preds)

    return {
        "accuracy":    accuracy,
        "auc":         auc,
        "f1":          f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision":   precision
    }


def print_results(model_name: str, metrics: dict, all_labels, all_preds):
    """
    Print a formatted results summary to stdout.

    Parameters
    ----------
    model_name : str
    metrics    : dict from compute_metrics()
    all_labels : np.array
    all_preds  : np.array
    """
    print("=" * 55)
    print(f"{model_name} — TEST SET RESULTS")
    print("=" * 55)
    print(f"  Accuracy:    {metrics['accuracy']*100:.2f}%")
    print(f"  AUC-ROC:     {metrics['auc']:.4f}")
    print(f"  F1 Score:    {metrics['f1']:.4f}")
    print(f"  Sensitivity: {metrics['sensitivity']*100:.2f}%")
    print(f"  Specificity: {metrics['specificity']*100:.2f}%")
    print(f"  Precision:   {metrics['precision']*100:.2f}%")
    print("=" * 55)
    print("\nClassification Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=["Negative", "Positive"]
    ))


def plot_confusion_and_roc(
    model_name: str,
    all_labels,
    all_preds,
    all_probs,
    metrics: dict,
    save_path: str = None
):
    """
    Plot confusion matrix and ROC curve side by side.

    Parameters
    ----------
    model_name : str
    all_labels : np.array
    all_preds  : np.array
    all_probs  : np.array
    metrics    : dict from compute_metrics()
    save_path  : str or None — if provided, saves the figure to this path
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        ax=axes[0]
    )
    axes[0].set_title(f"Confusion Matrix — {model_name}", fontweight="bold")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")
    tn, fp, fn, tp = cm.ravel()
    axes[0].text(
        0.5, -0.15, f"TN={tn}  FP={fp}  FN={fn}  TP={tp}",
        transform=axes[0].transAxes, ha="center", fontsize=11
    )

    # ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    axes[1].plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={metrics['auc']:.4f})")
    axes[1].plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")
    axes[1].fill_between(fpr, tpr, alpha=0.1)
    axes[1].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[1].set_ylabel("True Positive Rate (Sensitivity)")
    axes[1].set_title(f"ROC Curve — {model_name}", fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"{model_name} — Evaluation Results", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()


def evaluate_model(model, loader, device, model_name: str, save_path: str = None) -> dict:
    """
    Full evaluation pipeline: inference, metrics, print results, plot figures.

    Parameters
    ----------
    model      : nn.Module with trained weights loaded
    loader     : DataLoader (test set)
    device     : torch.device
    model_name : str
    save_path  : str or None — path to save confusion/ROC figure

    Returns
    -------
    dict with keys: preds, probs, labels, metrics
    """
    preds, probs, labels = get_predictions(model, loader, device)
    metrics              = compute_metrics(labels, preds, probs)

    print_results(model_name, metrics, labels, preds)
    plot_confusion_and_roc(model_name, labels, preds, probs, metrics, save_path)

    return {
        "preds":   preds,
        "probs":   probs,
        "labels":  labels,
        "metrics": metrics
    }
