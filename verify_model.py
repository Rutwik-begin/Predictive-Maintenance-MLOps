
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data" / "processed"

model_path = ARTIFACT_DIR / "best_model.joblib"
metadata_path = ARTIFACT_DIR / "model_metadata.json"

# Load model and metadata
model = joblib.load(model_path)

with open(metadata_path, "r", encoding="utf-8") as file:
    metadata = json.load(file)

# Load test data
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
y_test = pd.read_csv(DATA_DIR / "y_test.csv").iloc[:, 0]

# Predict failure labels and probabilities
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("Model loaded successfully!")
print(f"Model: {metadata['best_model']}")
print(f"Features: {len(metadata['features'])}")
print(f"Test records: {len(X_test)}")
print(f"Predictions generated: {len(predictions)}")

print("\nFirst 10 failure probabilities:")
print(probabilities[:10].round(4))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["No Failure", "Failure"],
        zero_division=0,
    )
)

print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))
