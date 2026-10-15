
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/ai4i2020.csv")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Predictors available for making a machine-failure prediction
feature_columns = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

target_column = "Machine failure"

X = df[feature_columns].copy()
y = df[target_column].copy()

# Convert machine type into numeric indicator columns
X = pd.get_dummies(X, columns=["Type"], dtype=int)

# Split before fitting any preprocessing or model
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Save consistent feature columns and labels for later training
X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)
y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)

print("Data preparation completed successfully!")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")
print(f"Training failures: {int(y_train.sum())}")
print(f"Testing failures: {int(y_test.sum())}")
print(f"Feature columns: {X_train.columns.tolist()}")
print(f"Saved files to: {OUTPUT_DIR.resolve()}")
