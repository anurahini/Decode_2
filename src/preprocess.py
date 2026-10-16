"""
Preprocessing Module: Train-Test Split and Feature Scaling
DecodeLabs AI Training Kit - Project 2: Data Classification Using AI
"""

from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
    scale_features: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, List[str], List[str]]:
    """
    Prepares the dataset according to DecodeLabs Project 2 architectural blueprint:
      1. Separates feature matrix X and target vector y.
      2. Applies 80/20 train-test split with shuffle=True and stratification.
      3. Fits StandardScaler (Mean = 0, Variance = 1) strictly on training data
         to avoid data leakage, and transforms both train and test splits.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        test_size (float): Proportion of dataset to include in the test split (default 0.20 = 20%).
        random_state (int): Seed for reproducible random shuffling.
        scale_features (bool): Whether to apply StandardScaler.

    Returns:
        X_train_scaled (np.ndarray): Scaled training features.
        X_test_scaled (np.ndarray): Scaled testing features.
        y_train (np.ndarray): Training labels.
        y_test (np.ndarray): Testing labels.
        scaler (StandardScaler): Fitted StandardScaler instance.
        feature_names (list): Names of feature columns.
        target_names (list): Unique class labels.
    """
    feature_names = [col for col in df.columns if col != "species"]
    target_names = sorted(df["species"].unique().tolist())

    X = np.array(df[feature_names].values, dtype=float)
    y = np.array(df["species"].tolist())

    # 80/20 train-test split with shuffling and stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=y
    )

    scaler = StandardScaler()
    if scale_features:
        # Fit on training data ONLY, transform both
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names, target_names


if __name__ == "__main__":
    from data_loader import load_iris_data
    df = load_iris_data()
    X_train, X_test, y_train, y_test, scaler, feat_names, target_names = prepare_data(df)
    print("=" * 60)
    print("DecodeLabs AI Project 2 - Preprocessing Verification")
    print("=" * 60)
    print(f"X_train shape: {X_train.shape} (80% split)")
    print(f"X_test shape : {X_test.shape} (20% split)")
    print(f"Scaled feature means (train): {np.round(X_train.mean(axis=0), 3)}")
    print(f"Scaled feature std   (train): {np.round(X_train.std(axis=0), 3)}")
    print(f"Target classes: {target_names}")
