"""
Dataset Loader and Inspection Module
DecodeLabs AI Training Kit - Project 2: Data Classification Using AI
"""

import os
from typing import Tuple, Dict, Any
import pandas as pd
from sklearn.datasets import load_iris


def load_iris_data(csv_path: str = "data/iris.csv") -> pd.DataFrame:
    """
    Loads the Iris dataset from CSV. If the CSV does not exist,
    fetches from scikit-learn and caches it to disk.

    Parameters:
        csv_path (str): Relative or absolute path to the iris CSV file.

    Returns:
        pd.DataFrame: Formatted dataframe with 150 samples, 4 features, and target species.
    """
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        iris = load_iris()
        df = pd.DataFrame(
            data=iris.data,
            columns=["sepal_length", "sepal_width", "petal_length", "petal_width"]
        )
        df["species"] = [iris.target_names[i] for i in iris.target]
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"[DataLoader] Cached iris dataset to {csv_path}")

    return df


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes summary metrics for data understanding.

    Parameters:
        df (pd.DataFrame): Dataset dataframe.

    Returns:
        dict: Summary statistics including shapes, class counts, and null counts.
    """
    summary = {
        "total_samples": len(df),
        "features": [col for col in df.columns if col != "species"],
        "num_features": len(df.columns) - 1,
        "classes": df["species"].unique().tolist(),
        "num_classes": df["species"].nunique(),
        "class_distribution": df["species"].value_counts().to_dict(),
        "missing_values": int(df.isnull().sum().sum()),
        "is_balanced": bool(df["species"].value_counts().nunique() == 1)
    }
    return summary


if __name__ == "__main__":
    df = load_iris_data()
    summary = get_dataset_summary(df)
    print("=" * 60)
    print("DecodeLabs AI Project 2 - Dataset Inspection Summary")
    print("=" * 60)
    for key, value in summary.items():
        print(f"  {key:<20}: {value}")
    print("\nSample Data:")
    print(df.head())
