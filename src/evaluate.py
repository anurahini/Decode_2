"""
Model Evaluation Module: Confusion Matrix and Classification Metrics
DecodeLabs AI Training Kit - Project 2: Data Classification Using AI
"""

import os
from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import joblib

try:
    from .data_loader import load_iris_data
    from .preprocess import prepare_data
except ImportError:
    from data_loader import load_iris_data
    from preprocess import prepare_data


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_names: list,
    plot_path: str = "plots/confusion_matrix.png"
) -> Dict[str, Any]:
    """
    Evaluates classification performance per DecodeLabs Project 2 specification:
    - Generates Confusion Matrix (TP, FP, FN, TN per class)
    - Computes Accuracy, Precision, Recall, and F1 Score (Harmonic Mean)
    - Saves styled Confusion Matrix visualization.

    Parameters:
        model: Trained classifier (e.g. KNeighborsClassifier).
        X_test (np.ndarray): Scaled test features.
        y_test (np.ndarray): Ground truth test labels.
        target_names (list): List of class names.
        plot_path (str): Filepath to save confusion matrix image.

    Returns:
        dict: Performance metrics dictionary.
    """
    y_pred = model.predict(X_test)

    # Core metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    rec_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=target_names)
    clf_report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    clf_report_text = classification_report(y_test, y_pred, target_names=target_names)

    # Plot styled confusion matrix
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=200)

    # DecodeLabs style palette
    cmap = sns.color_palette("Blues", as_cmap=True)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=target_names,
        yticklabels=target_names,
        cbar=True,
        linewidths=1.5,
        linecolor="white",
        annot_kws={"size": 13, "weight": "bold", "color": "#0F172A"}
    )

    ax.set_title("THE DIAGNOSTIC TOOL: CONFUSION MATRIX\nEvaluation on Test Set (20% Split)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="medium", labelpad=10)
    ax.set_ylabel("True Ground Truth Class", fontsize=11, fontweight="medium", labelpad=10)

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    metrics = {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "precision_weighted": prec_weighted,
        "recall_macro": rec_macro,
        "recall_weighted": rec_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm.tolist(),
        "classification_report": clf_report,
        "classification_report_text": clf_report_text
    }

    return metrics


def run_evaluation(
    model_path: str = "saved_models/knn_model.joblib",
    scaler_path: str = "saved_models/scaler.joblib",
    data_path: str = "data/iris.csv"
) -> Dict[str, Any]:
    """
    Loads saved model and evaluates on test set.
    """
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        from train import run_training_pipeline
        print("[Evaluate] Saved models not found. Running training first...")
        run_training_pipeline()

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    df = load_iris_data(data_path)
    X_train, X_test, y_train, y_test, _, feature_names, target_names = prepare_data(df)

    metrics = evaluate_model(model, X_test, y_test, target_names)

    print("=" * 65)
    print("DecodeLabs AI Project 2: Model Performance & Output Validation")
    print("=" * 65)
    print(f"Accuracy                 : {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro Precision          : {metrics['precision_macro']:.4f}")
    print(f"Macro Recall             : {metrics['recall_macro']:.4f}")
    print(f"Macro F1 Score (Harmonic): {metrics['f1_macro']:.4f}")
    print("\nDetailed Classification Report:")
    print(metrics["classification_report_text"])
    print(f"[Evaluate] Confusion matrix heatmap saved to plots/confusion_matrix.png")

    return metrics


if __name__ == "__main__":
    run_evaluation()
