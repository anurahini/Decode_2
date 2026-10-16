"""
Inference & Prediction Module: Classify New Flower Measurements
DecodeLabs AI Training Kit - Project 2: Data Classification Using AI
"""

import os
import argparse
from typing import Dict, Any, Union, List
import numpy as np
import joblib


def load_inference_artifacts(
    model_path: str = "saved_models/knn_model.joblib",
    scaler_path: str = "saved_models/scaler.joblib"
) -> tuple:
    """Loads pre-trained model and feature scaler."""
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Artifacts missing: {model_path} or {scaler_path}. Run 'python src/train.py' first."
        )
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_sample(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
    model=None,
    scaler=None,
    model_path: str = "saved_models/knn_model.joblib",
    scaler_path: str = "saved_models/scaler.joblib"
) -> Dict[str, Any]:
    """
    Takes 4 raw flower measurements, standardizes them via the fitted scaler,
    and returns the predicted species along with class probabilities.

    Parameters:
        sepal_length (float): Sepal length in cm.
        sepal_width (float): Sepal width in cm.
        petal_length (float): Petal length in cm.
        petal_width (float): Petal width in cm.
        model: Optional pre-loaded KNeighborsClassifier.
        scaler: Optional pre-loaded StandardScaler.

    Returns:
        dict: Prediction results including predicted species and class probabilities.
    """
    if model is None or scaler is None:
        model, scaler = load_inference_artifacts(model_path, scaler_path)

    raw_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]], dtype=float)
    scaled_features = scaler.transform(raw_features)

    prediction = model.predict(scaled_features)[0]
    probabilities = model.predict_proba(scaled_features)[0]

    classes = model.classes_.tolist()
    prob_dict = {cls: float(np.round(p, 4)) for cls, p in zip(classes, probabilities)}

    return {
        "inputs": {
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width
        },
        "scaled_features": scaled_features[0].tolist(),
        "predicted_species": prediction,
        "confidence": float(np.max(probabilities)),
        "class_probabilities": prob_dict
    }


def main():
    parser = argparse.ArgumentParser(description="DecodeLabs AI Project 2: Iris Species Predictor")
    parser.add_argument("--sepal_length", type=float, default=None, help="Sepal length in cm (e.g. 5.1)")
    parser.add_argument("--sepal_width", type=float, default=None, help="Sepal width in cm (e.g. 3.5)")
    parser.add_argument("--petal_length", type=float, default=None, help="Petal length in cm (e.g. 1.4)")
    parser.add_argument("--petal_width", type=float, default=None, help="Petal width in cm (e.g. 0.2)")
    args = parser.parse_args()

    # Pre-defined test cases if no arguments are provided
    test_cases = [
        {"name": "Setosa Benchmark Sample", "sl": 5.1, "sw": 3.5, "pl": 1.4, "pw": 0.2, "expected": "setosa"},
        {"name": "Versicolor Benchmark Sample", "sl": 6.0, "sw": 2.9, "pl": 4.5, "pw": 1.5, "expected": "versicolor"},
        {"name": "Virginica Benchmark Sample", "sl": 6.8, "sw": 3.2, "pl": 5.9, "pw": 2.3, "expected": "virginica"}
    ]

    model, scaler = load_inference_artifacts()

    if args.sepal_length is not None and args.sepal_width is not None and args.petal_length is not None and args.petal_width is not None:
        result = predict_sample(args.sepal_length, args.sepal_width, args.petal_length, args.petal_width, model, scaler)
        print("=" * 60)
        print("DecodeLabs AI Project 2: Real-Time Prediction Result")
        print("=" * 60)
        print(f"Inputs           : Sepal ({result['inputs']['sepal_length']}x{result['inputs']['sepal_width']}), Petal ({result['inputs']['petal_length']}x{result['inputs']['petal_width']})")
        print(f"Predicted Species: {result['predicted_species'].upper()}")
        print(f"Confidence       : {result['confidence'] * 100:.1f}%")
        print("Probabilities    :")
        for cls, p in result["class_probabilities"].items():
            print(f"  - {cls:<12}: {p * 100:.1f}%")
    else:
        print("=" * 60)
        print("DecodeLabs AI Project 2: Batch Verification on Benchmark Samples")
        print("=" * 60)
        for tc in test_cases:
            res = predict_sample(tc["sl"], tc["sw"], tc["pl"], tc["pw"], model, scaler)
            status = "MATCH" if res["predicted_species"] == tc["expected"] else "MISMATCH"
            print(f"\n[Case: {tc['name']}]")
            print(f"  Input        : SL={tc['sl']}, SW={tc['sw']}, PL={tc['pl']}, PW={tc['pw']}")
            print(f"  Predicted    : {res['predicted_species']} ({status}) [Confidence: {res['confidence']*100:.1f}%]")
            print(f"  Probabilities: {res['class_probabilities']}")


if __name__ == "__main__":
    main()
