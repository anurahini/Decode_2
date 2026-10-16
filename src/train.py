"""
Model Training & Hyperparameter Tuning Module
DecodeLabs AI Training Kit - Project 2: Data Classification Using AI
"""

import os
from typing import Tuple, List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

try:
    from .data_loader import load_iris_data
    from .preprocess import prepare_data
except ImportError:
    from data_loader import load_iris_data
    from preprocess import prepare_data


def find_optimal_k(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    max_k: int = 25,
    plot_path: str = "plots/elbow_method.png"
) -> Tuple[int, List[float]]:
    """
    Implements the Elbow Method to evaluate the optimal K parameter
    as specified in DecodeLabs Project 2 ('Tuning the Engine: Choosing K').

    Parameters:
        X_train (np.ndarray): Scaled training features.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Scaled test features.
        y_test (np.ndarray): Testing labels.
        max_k (int): Upper bound of K values to explore (default 25).
        plot_path (str): Destination path for the error rate curve plot.

    Returns:
        optimal_k (int): Selected optimal number of neighbors.
        error_rates (list): Computed error rate (1 - accuracy) for each k.
    """
    error_rates = []
    k_range = range(1, max_k + 1)

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        err = float(np.mean(pred_i != y_test))
        error_rates.append(err)

    # Determine optimal k:
    # Per Slide 12, K=1 is prone to noise/overfitting. If K >= 3 achieves the same
    # minimum error rate (or within 0.01 margin), prefer K in [3, 5, 7].
    min_err = min(error_rates)
    candidates_at_min = [k for k, err in zip(k_range, error_rates) if err <= min_err + 0.001]
    robust_candidates = [k for k in candidates_at_min if k >= 3 and k % 2 != 0]

    if robust_candidates:
        optimal_k = robust_candidates[0]
    else:
        optimal_k = candidates_at_min[0]

    # Generate and style the Elbow plot (matching DecodeLabs aesthetic)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200)

    # Theme colors
    bg_color = "#F8FAFC"
    line_color = "#1E3A8A"
    highlight_color = "#EA580C"
    grid_color = "#E2E8F0"

    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    ax.plot(
        list(k_range),
        error_rates,
        color=line_color,
        linestyle="--",
        marker="o",
        markerfacecolor=line_color,
        markersize=6,
        linewidth=2,
        label="Test Error Rate"
    )

    # Highlight optimal k (The Elbow)
    opt_err = error_rates[optimal_k - 1]
    ax.scatter(
        [optimal_k],
        [opt_err],
        color=highlight_color,
        s=180,
        zorder=5,
        edgecolors="black",
        linewidth=1.5,
        label=f"Optimal Elbow (K={optimal_k}, Error={opt_err:.3f})"
    )

    ax.annotate(
        f"The Elbow (K={optimal_k})",
        xy=(optimal_k, opt_err),
        xytext=(optimal_k + 1.5, opt_err + 0.03),
        arrowprops=dict(facecolor=highlight_color, shrink=0.08, width=1.5, headwidth=8),
        fontweight="bold",
        fontsize=10,
        color=highlight_color
    )

    ax.set_title("TUNING THE ENGINE: CHOOSING 'K'\nError Rate vs. K Value (Elbow Method)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("K Value (Number of Neighbors)", fontsize=11, fontweight="medium")
    ax.set_ylabel("Error Rate (1 - Accuracy)", fontsize=11, fontweight="medium")
    ax.set_xticks(list(k_range))
    ax.grid(True, linestyle=":", color=grid_color, alpha=0.9)
    ax.legend(frameon=True, facecolor="white", edgecolor="#CBD5E1")

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"[Train] Elbow method plot saved to {plot_path}")
    print(f"[Train] Optimal K discovered: K={optimal_k} with test error rate: {opt_err:.4f}")

    return optimal_k, error_rates


def train_knn_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_neighbors: int = 5
) -> KNeighborsClassifier:
    """
    Instantiates and fits the KNeighborsClassifier according to
    the Scikit-Learn 3-step workflow (Instantiate -> Fit -> Predict).

    Parameters:
        X_train (np.ndarray): Scaled training features.
        y_train (np.ndarray): Training labels.
        n_neighbors (int): Number of nearest neighbors.

    Returns:
        KNeighborsClassifier: Fitted classifier model.
    """
    model = KNeighborsClassifier(n_neighbors=n_neighbors, metric="minkowski", p=2)
    model.fit(X_train, y_train)
    return model


def save_artifacts(
    model: KNeighborsClassifier,
    scaler: Any,
    model_path: str = "saved_models/knn_model.joblib",
    scaler_path: str = "saved_models/scaler.joblib"
) -> None:
    """
    Saves trained model and fitted scaler for portable deployment and evaluation.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"[Train] Model saved to {model_path}")
    print(f"[Train] Scaler saved to {scaler_path}")


def run_training_pipeline() -> Dict[str, Any]:
    """
    Executes the end-to-end training pipeline.
    """
    print("=" * 60)
    print("DecodeLabs AI Project 2: Starting Model Training Pipeline")
    print("=" * 60)

    # 1. Load data
    df = load_iris_data("data/iris.csv")

    # 2. Preprocess & Scale
    X_train, X_test, y_train, y_test, scaler, feature_names, target_names = prepare_data(df)

    # 3. Find optimal K via Elbow Method
    optimal_k, error_rates = find_optimal_k(
        X_train, y_train, X_test, y_test, max_k=25, plot_path="plots/elbow_method.png"
    )

    # 4. Train final model with optimal K
    model = train_knn_model(X_train, y_train, n_neighbors=optimal_k)

    # 5. Evaluate on train and test
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"[Train] Training Accuracy: {train_acc * 100:.2f}%")
    print(f"[Train] Testing Accuracy : {test_acc * 100:.2f}%")

    # 6. Save model and scaler
    save_artifacts(model, scaler)

    return {
        "model": model,
        "scaler": scaler,
        "optimal_k": optimal_k,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "feature_names": feature_names,
        "target_names": target_names
    }


if __name__ == "__main__":
    run_training_pipeline()
