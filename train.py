
import json
from pathlib import Path

import joblib
import mlflow
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
ARTIFACT_DIR = BASE_DIR / "artifacts"

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_DB = BASE_DIR / "mlflow.db"

X_TRAIN_PATH = DATA_DIR / "X_train.csv"
X_TEST_PATH = DATA_DIR / "X_test.csv"
Y_TRAIN_PATH = DATA_DIR / "y_train.csv"
Y_TEST_PATH = DATA_DIR / "y_test.csv"


# --------------------------------------------------
# 2. Load prepared data
# --------------------------------------------------

required_files = [
    X_TRAIN_PATH,
    X_TEST_PATH,
    Y_TRAIN_PATH,
    Y_TEST_PATH,
]

for file_path in required_files:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}\n"
            "Run prepare_data.py first."
        )

X_train = pd.read_csv(X_TRAIN_PATH)
X_test = pd.read_csv(X_TEST_PATH)

y_train = pd.read_csv(Y_TRAIN_PATH).iloc[:, 0]
y_test = pd.read_csv(Y_TEST_PATH).iloc[:, 0]

# Ensure the training and test columns match.
if list(X_train.columns) != list(X_test.columns):
    raise ValueError("Training and test feature columns do not match.")

if set(y_train.unique()) != {0, 1}:
    raise ValueError("Training target must contain both classes 0 and 1.")

if set(y_test.unique()) != {0, 1}:
    raise ValueError("Test target must contain both classes 0 and 1.")

print("=" * 60)
print("PREDICTIVE MAINTENANCE MODEL TRAINING")
print("=" * 60)
print(f"Training records: {len(X_train)}")
print(f"Testing records:  {len(X_test)}")
print(f"Training failures: {int(y_train.sum())}")
print(f"Testing failures:  {int(y_test.sum())}")
print(f"Features: {list(X_train.columns)}")


# --------------------------------------------------
# 3. Define models
# --------------------------------------------------

models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        ),
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42,
    ),
}


# --------------------------------------------------
# 4. Configure MLflow
# --------------------------------------------------

mlflow.set_tracking_uri(MLFLOW_DB.as_uri().replace(
    "file:///", "sqlite:///"
))
mlflow.set_experiment("Predictive Maintenance")

results = []
best_model = None
best_name = None
best_f1 = -1.0


# --------------------------------------------------
# 5. Train, evaluate, and log metrics
# --------------------------------------------------

for name, model in models.items():
    print("\n" + "=" * 60)
    print(f"TRAINING: {name}")
    print("=" * 60)

    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(
                precision_score(y_test, predictions, zero_division=0)
            ),
            "recall": float(
                recall_score(y_test, predictions, zero_division=0)
            ),
            "f1": float(
                f1_score(y_test, predictions, zero_division=0)
            ),
            "roc_auc": float(
                roc_auc_score(y_test, probabilities)
            ),
        }

        # Log model configuration and metrics to MLflow.
        mlflow.log_params({
            "model_name": name,
            "training_rows": len(X_train),
            "testing_rows": len(X_test),
            "feature_count": X_train.shape[1],
            "random_state": 42,
        })

        mlflow.log_metrics(metrics)

        # Print evaluation results.
        print("\nMetrics:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

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

        results.append({
            "model": name,
            **metrics,
        })

        # Select the model using failure-class F1-score.
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model = model
            best_name = name


# --------------------------------------------------
# 6. Save comparison results
# --------------------------------------------------

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="f1",
    ascending=False,
)

comparison_path = ARTIFACT_DIR / "model_comparison.csv"
results_df.to_csv(comparison_path, index=False)


# --------------------------------------------------
# 7. Save the best model and metadata
# --------------------------------------------------

model_path = ARTIFACT_DIR / "best_model.joblib"
joblib.dump(best_model, model_path)

metadata = {
    "project": "Predictive Maintenance",
    "best_model": best_name,
    "selection_metric": "failure_class_f1",
    "selection_score": float(best_f1),
    "features": list(X_train.columns),
    "target": "Machine failure",
    "positive_class": 1,
    "threshold": 0.5,
    "training_rows": int(len(X_train)),
    "testing_rows": int(len(X_test)),
    "metrics": next(
        row for row in results
        if row["model"] == best_name
    ),
}

metadata_path = ARTIFACT_DIR / "model_metadata.json"

with open(metadata_path, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)


# --------------------------------------------------
# 8. Final summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(results_df.to_string(index=False, float_format="%.4f"))

print("\nBEST MODEL")
print(f"Name: {best_name}")
print(f"Failure-class F1-score: {best_f1:.4f}")

print("\nSAVED FILES")
print(f"Model:      {model_path}")
print(f"Metadata:   {metadata_path}")
print(f"Comparison: {comparison_path}")
print(f"MLflow DB:  {MLFLOW_DB}")

print("\nTraining and evaluation completed successfully!")
