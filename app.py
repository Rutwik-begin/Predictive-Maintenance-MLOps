
import json
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd


# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "best_model.joblib"
METADATA_PATH = BASE_DIR / "artifacts" / "model_metadata.json"
TRAIN_DATA_PATH = BASE_DIR / "data" / "processed" / "X_train.csv"


# Load trained model
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# Use the exact column names and order used during training.
if TRAIN_DATA_PATH.exists():
    FEATURES = pd.read_csv(TRAIN_DATA_PATH, nrows=0).columns.tolist()
elif METADATA_PATH.exists():
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)
    FEATURES = metadata["features"]
else:
    raise FileNotFoundError(
        "Could not find X_train.csv or model_metadata.json."
    )

if len(FEATURES) != model.n_features_in_:
    raise ValueError(
        f"Feature mismatch: model expects {model.n_features_in_} features, "
        f"but the feature list contains {len(FEATURES)}."
    )


def normalize_name(name):
    """Ignore spaces and punctuation when matching feature names."""
    return "".join(character.lower() for character in name
                   if character.isalnum())


def predict_failure(machine_type, air_temperature, process_temperature,
                    rotational_speed, torque, tool_wear):
    # Raw input values from the Gradio form
    values = {
        "Air temperature [K]": air_temperature,
        "Process temperature [K]": process_temperature,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "Type_H": int(machine_type == "H"),
        "Type_L": int(machine_type == "L"),
        "Type_M": int(machine_type == "M"),
    }

    # Match feature names robustly, including "Toolwear [min]"
    normalized_values = {
        normalize_name(key): value for key, value in values.items()
    }

    row = {}
    for feature in FEATURES:
        normalized_feature = normalize_name(feature)

        if normalized_feature not in normalized_values:
            raise ValueError(
                f"Cannot map model feature '{feature}' to an input value."
            )

        row[feature] = normalized_values[normalized_feature]

    # Preserve the model's training feature order
    input_df = pd.DataFrame([row], columns=FEATURES)

    prediction = int(model.predict(input_df)[0])

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(input_df)[0][1]) * 100
    else:
        probability = None

    if prediction == 1:
        result = "⚠️ **Machine failure predicted**"
    else:
        result = "✅ **No machine failure predicted**"

    if probability is not None:
        result += f"\n\nEstimated failure probability: **{probability:.2f}%**"

    result += (
        "\n\n*This is a model prediction, not a guarantee of machine condition.*"
    )

    return result


# Gradio interface
with gr.Blocks(title="Predictive Maintenance") as demo:
    gr.Markdown("# 🔧 Predictive Maintenance")
    gr.Markdown(
        "Enter machine measurements to estimate the likelihood of failure."
    )

    with gr.Row():
        machine_type = gr.Dropdown(
            choices=["L", "M", "H"],
            value="L",
            label="Machine Type",
        )
        air_temperature = gr.Number(
            value=298.1, label="Air Temperature (K)"
        )
        process_temperature = gr.Number(
            value=308.6, label="Process Temperature (K)"
        )

    with gr.Row():
        rotational_speed = gr.Number(
            value=1551, label="Rotational Speed (rpm)"
        )
        torque = gr.Number(value=42.8, label="Torque (Nm)")
        tool_wear = gr.Number(value=0, label="Tool Wear (min)")

    predict_button = gr.Button("Predict Failure", variant="primary")
    output = gr.Markdown()

    predict_button.click(
        fn=predict_failure,
        inputs=[
            machine_type,
            air_temperature,
            process_temperature,
            rotational_speed,
            torque,
            tool_wear,
        ],
        outputs=output,
    )

if __name__ == "__main__":
    demo.launch()
