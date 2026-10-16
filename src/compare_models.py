"""
Algorithm Comparison Benchmark Module
DecodeLabs AI Training Kit - Project 2: Data Classification Using AI
"""

import os
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    from .data_loader import load_iris_data
    from .preprocess import prepare_data
except ImportError:
    from data_loader import load_iris_data
    from preprocess import prepare_data


def run_benchmark(plot_path: str = "plots/model_comparison.png") -> pd.DataFrame:
    """
    Benchmarks multiple supervised learning algorithms against KNN
    on identical 80/20 standardized train-test splits.
    """
    df = load_iris_data("data/iris.csv")
    X_train, X_test, y_train, y_test, _, _, _ = prepare_data(df)

    models = {
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=7),
        "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
        "Support Vector Classifier (SVC)": SVC(kernel="rbf", probability=True, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gaussian Naive Bayes": GaussianNB()
    }

    results = []

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        results.append({
            "Algorithm": name,
            "Accuracy": acc,
            "Precision (Macro)": prec,
            "Recall (Macro)": rec,
            "F1-Score (Macro)": f1
        })

    res_df = pd.DataFrame(results)

    # Plot model comparison chart
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200)

    # Palette
    colors = ["#2563EB", "#059669", "#7C3AED", "#D97706", "#DC2626", "#0891B2"]

    bars = ax.barh(res_df["Algorithm"], res_df["Accuracy"] * 100, color=colors, height=0.55, edgecolor="#1E293B", linewidth=0.8)

    ax.set_xlim(85, 102)
    ax.set_xlabel("Test Accuracy (%)", fontsize=11, fontweight="medium", labelpad=10)
    ax.set_title("ALGORITHM BENCHMARK COMPARISON\nEvaluating Supervised Classifiers on Iris Benchmark", fontsize=13, fontweight="bold", pad=15)
    ax.grid(axis="x", linestyle=":", alpha=0.7)

    # Add accuracy percentage labels on bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.3, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%",
                va="center", ha="left", fontsize=10, fontweight="bold", color="#1E293B")

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"[Benchmark] Comparison plot saved to {plot_path}")

    return res_df


if __name__ == "__main__":
    print("=" * 70)
    print("DecodeLabs AI Project 2: Algorithm Comparison Benchmark")
    print("=" * 70)
    df_results = run_benchmark()
    print(df_results.to_string(index=False))
