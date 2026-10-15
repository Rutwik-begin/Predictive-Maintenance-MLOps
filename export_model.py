
import json
from pathlib import Path

import joblib
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "best_model.joblib"
METADATA_PATH = BASE_DIR / "artifacts" / "model_metadata.json"
OUTPUT_PATH = BASE_DIR / "artifacts" / "model.json"


def export_tree(estimator):
    """Convert one sklearn decision tree into JSON-compatible data."""
    tree = estimator.tree_

    return {
        "children_left": tree.children_left.tolist(),
        "children_right": tree.children_right.tolist(),
        "feature": tree.feature.tolist(),
        "threshold": tree.threshold.tolist(),
        "value": tree.value[:, 0, 0].tolist(),
    }


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing metadata: {METADATA_PATH}")

    model = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    if model.n_features_in_ != len(metadata["features"]):
        raise ValueError("Model feature count does not match metadata.")

    if model.n_classes_ != 2:
        raise ValueError("This export script expects a binary classifier.")

    # Obtain the initial raw score used by sklearn.
    dummy_input = np.zeros((1, model.n_features_in_), dtype=float)
    initial_prediction = model._raw_predict_init(dummy_input)
    initial_score = float(initial_prediction[0, 0])

    # GradientBoostingClassifier stores one regression tree per stage
    # for binary classification.
    trees = [
        export_tree(model.estimators_[stage, 0])
        for stage in range(model.n_estimators)
    ]

    payload = {
        "model_type": "GradientBoostingClassifier",
        "features": metadata["features"],
        "classes": model.classes_.tolist(),
        "n_features": int(model.n_features_in_),
        "n_estimators": int(model.n_estimators),
        "learning_rate": float(model.learning_rate),
        "initial_score": initial_score,
        "trees": trees,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(payload, file, separators=(",", ":"), allow_nan=False)

    print("Model exported successfully!")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Features: {payload['n_features']}")
    print(f"Boosting stages: {len(trees)}")
    print(f"Feature order: {payload['features']}")
    print(f"File size: {OUTPUT_PATH.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
