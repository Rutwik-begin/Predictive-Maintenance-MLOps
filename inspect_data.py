
import pandas as pd
from pathlib import Path

# Dataset location
DATA_PATH = Path("data/ai4i2020.csv")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH.resolve()}. "
        "Check the CSV filename and location."
    )

# Load dataset
df = pd.read_csv(DATA_PATH)

print("\n--- DATASET SHAPE ---")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- MACHINE FAILURE COUNTS ---")
if "Machine failure" in df.columns:
    print(df["Machine failure"].value_counts())
else:
    print("Target column 'Machine failure' was not found.")
