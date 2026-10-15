
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path("data/ai4i2020.csv")
OUTPUT_DIR = Path("artifacts/eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# 1. Machine failure distribution
failure_counts = df["Machine failure"].value_counts().sort_index()

print("Machine failure distribution:")
print(failure_counts)
print(f"\nFailure percentage: {df['Machine failure'].mean() * 100:.2f}%")

failure_counts.plot(kind="bar", rot=0)
plt.title("Machine Failure Distribution")
plt.xlabel("Machine failure (0 = No, 1 = Yes)")
plt.ylabel("Number of records")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "failure_distribution.png", dpi=150)
plt.close()

# 2. Numeric feature distributions by failure status
features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

for feature in features:
    df.boxplot(column=feature, by="Machine failure")
    plt.title(f"{feature} by Machine Failure")
    plt.suptitle("")
    plt.xlabel("Machine failure (0 = No, 1 = Yes)")
    plt.tight_layout()

    safe_name = (
        feature.lower()
        .replace(" ", "_")
        .replace("[", "")
        .replace("]", "")
    )
    plt.savefig(OUTPUT_DIR / f"{safe_name}.png", dpi=150)
    plt.close()

print(f"\nEDA plots saved to: {OUTPUT_DIR.resolve()}")
