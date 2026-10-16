"""
Data Exploration and Feature Visualization Module
DecodeLabs AI Training Kit - Project 2: Data Classification Using AI
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

try:
    from .data_loader import load_iris_data
except ImportError:
    from data_loader import load_iris_data


def generate_eda_plots(df: pd.DataFrame, output_dir: str = "plots") -> None:
    """
    Generates Exploratory Data Analysis visualizations:
    1. Feature Distributions (Boxplots & KDE violin distributions)
    2. Pairplot of Petal and Sepal dimensions categorized by species
    """
    os.makedirs(output_dir, exist_ok=True)
    palette = {"setosa": "#3B82F6", "versicolor": "#10B981", "virginica": "#F97316"}

    # 1. Feature Distributions Boxplot
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=200)
    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    titles = ["Sepal Length (cm)", "Sepal Width (cm)", "Petal Length (cm)", "Petal Width (cm)"]

    for ax, feat, title in zip(axes.flatten(), features, titles):
        sns.boxplot(data=df, x="species", y=feat, hue="species", palette=palette, ax=ax, legend=False)
        sns.stripplot(data=df, x="species", y=feat, color="#1E293B", size=3.5, jitter=0.2, alpha=0.6, ax=ax)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("cm")
        ax.grid(axis="y", linestyle=":", alpha=0.6)

    plt.suptitle("THE IRIS BENCHMARK: FEATURE DISTRIBUTIONS PER CLASS", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    dist_path = os.path.join(output_dir, "feature_distributions.png")
    plt.savefig(dist_path)
    plt.close()
    print(f"[Visualize] Feature distributions saved to {dist_path}")

    # 2. Pairplot
    pairplot = sns.pairplot(
        df,
        hue="species",
        palette=palette,
        corner=False,
        diag_kind="kde",
        plot_kws={"alpha": 0.8, "s": 35, "edgecolor": "none"}
    )
    pairplot.fig.suptitle("IRIS BENCHMARK PAIRPLOT // MULTI-DIMENSIONAL FEATURE INTERACTION", y=1.02, fontsize=12, fontweight="bold")
    pairplot_path = os.path.join(output_dir, "pairplot.png")
    pairplot.savefig(pairplot_path, dpi=200)
    plt.close()
    print(f"[Visualize] Pairplot saved to {pairplot_path}")


if __name__ == "__main__":
    df = load_iris_data()
    generate_eda_plots(df)
