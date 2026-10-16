"""
Unit and Integration Test Suite
DecodeLabs AI Training Kit - Project 2: Data Classification Using AI
"""

import os
import unittest
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import joblib

from src.data_loader import load_iris_data, get_dataset_summary
from src.preprocess import prepare_data
from src.train import train_knn_model, find_optimal_k
from src.evaluate import evaluate_model
from src.predict import predict_sample, load_inference_artifacts


class TestIrisClassificationPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_iris_data("data/iris.csv")
        cls.X_train, cls.X_test, cls.y_train, cls.y_test, cls.scaler, cls.feat_names, cls.target_names = prepare_data(cls.df)

    def test_01_dataset_integrity(self):
        """Verify Iris Benchmark specifications: 150 samples, 4 features, 3 balanced classes."""
        self.assertEqual(len(self.df), 150)
        self.assertEqual(len(self.feat_names), 4)
        self.assertEqual(len(self.target_names), 3)
        self.assertFalse(self.df.isnull().values.any(), "Dataset contains unexpected missing values.")

        # Verify class balance (50 samples per class)
        counts = self.df["species"].value_counts().to_dict()
        for cls_name in ["setosa", "versicolor", "virginica"]:
            self.assertEqual(counts[cls_name], 50, f"Class {cls_name} is not balanced.")

    def test_02_structural_split_80_20(self):
        """Verify Structural Split: 80% Train (120), 20% Test (30) with stratification."""
        self.assertEqual(self.X_train.shape, (120, 4))
        self.assertEqual(self.X_test.shape, (30, 4))
        self.assertEqual(len(self.y_train), 120)
        self.assertEqual(len(self.y_test), 30)

        # Test stratification in test set (10 per class)
        test_counts = pd.Series(self.y_test).value_counts().to_dict()
        for cls_name in self.target_names:
            self.assertEqual(test_counts[cls_name], 10, f"Stratification mismatch for {cls_name}")

    def test_03_standard_scaler_gatekeeper(self):
        """Verify The Gatekeeper Rule: Mean=0, Variance=1 on training data."""
        means = self.X_train.mean(axis=0)
        stds = self.X_train.std(axis=0)
        np.testing.assert_allclose(means, np.zeros(4), atol=1e-7, err_msg="Scaled train mean is not 0")
        np.testing.assert_allclose(stds, np.ones(4), atol=1e-7, err_msg="Scaled train std is not 1")

    def test_04_knn_training_and_accuracy(self):
        """Verify KNN model training achieves target benchmark accuracy >= 90%."""
        model = train_knn_model(self.X_train, self.y_train, n_neighbors=7)
        self.assertIsInstance(model, KNeighborsClassifier)
        y_pred = model.predict(self.X_test)
        acc = float(np.mean(y_pred == self.y_test))
        self.assertGreaterEqual(acc, 0.90, f"Model accuracy {acc} below benchmark threshold of 0.90")

    def test_05_evaluation_metrics_and_matrix(self):
        """Verify output diagnostic metrics: Accuracy, Precision, Recall, F1-Score."""
        model = train_knn_model(self.X_train, self.y_train, n_neighbors=7)
        metrics = evaluate_model(model, self.X_test, self.y_test, self.target_names, plot_path="plots/test_cm.png")

        self.assertIn("accuracy", metrics)
        self.assertIn("f1_macro", metrics)
        self.assertIn("confusion_matrix", metrics)
        self.assertGreaterEqual(metrics["f1_macro"], 0.90)

        # Clean up temporary plot
        if os.path.exists("plots/test_cm.png"):
            os.remove("plots/test_cm.png")

    def test_06_realtime_inference(self):
        """Verify single-sample inference and probability calibration."""
        res = predict_sample(5.1, 3.5, 1.4, 0.2)
        self.assertEqual(res["predicted_species"], "setosa")
        self.assertGreaterEqual(res["confidence"], 0.8)

        # Probabilities sum to 1.0
        prob_sum = sum(res["class_probabilities"].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=3)

    def test_07_saved_artifacts_integrity(self):
        """Verify saved serialized model and scaler exist and can be loaded."""
        model, scaler = load_inference_artifacts()
        self.assertIsInstance(model, KNeighborsClassifier)
        self.assertIsInstance(scaler, StandardScaler)


if __name__ == "__main__":
    unittest.main()
