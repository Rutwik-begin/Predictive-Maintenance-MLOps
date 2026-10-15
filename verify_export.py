
import json
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "artifacts" / "best_model.joblib")
X_test = pd.read_csv(BASE_DIR / "data" / "processed" / "X_test.csv")
y_test = pd.read_csv(BASE_DIR / "data" / "processed" / "y_test.csv").iloc[:, 0]

# Use the exact feature order saved with the exported model.
with open(BASE_DIR / "artifacts" / "model.json", encoding="utf-8") as f:
    exported = json.load(f)

X_test = X_test[exported["features"]].head(20)

python_predictions = model.predict(X_test).astype(int)
python_probabilities = model.predict_proba(X_test)[:, 1]

# Send test samples to the JavaScript prediction engine.
samples = []
for _, row in X_test.iterrows():
    samples.append({name: float(row[name]) for name in exported["features"]})

payload = json.dumps(samples)

js_code = r"""
const fs = require("fs");
const model = JSON.parse(fs.readFileSync("artifacts/model.json", "utf8"));
global.window = {};
eval(fs.readFileSync("predict.js", "utf8"));

const samples = JSON.parse(process.argv[1]);
const results = samples.map(sample => window.predictFailure(model, sample));
console.log(JSON.stringify(results));
"""

result = subprocess.run(
    ["node", "-e", js_code, payload],
    cwd=BASE_DIR,
    capture_output=True,
    text=True,
)

if result.returncode != 0:
    print("JavaScript execution failed.")
    print(result.stderr)
    print("\nMake sure Node.js is installed and available in PATH.")
    raise SystemExit(result.returncode)

js_results = json.loads(result.stdout)
js_predictions = np.array([r["prediction"] for r in js_results])
js_probabilities = np.array([r["failureProbability"] for r in js_results])

print("Samples checked:", len(X_test))
print("Python predictions:", python_predictions.tolist())
print("JavaScript predictions:", js_predictions.tolist())
print(
    "Maximum probability difference:",
    f"{np.max(np.abs(python_probabilities - js_probabilities)):.10f}",
)
print("Predictions match:", np.array_equal(python_predictions, js_predictions))
print(
    "Probabilities match:",
    np.allclose(python_probabilities, js_probabilities, atol=1e-6),
)
