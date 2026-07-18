"""
Flask app that serves the trained insurance claim model.
Two ways to use it:
  1. Web form at '/'          -> fill in vehicle/policy details, get a prediction
  2. JSON API at '/api/predict' -> POST a single record or a list of records
"""

import os

import pandas as pd
from flask import Flask, render_template, request, jsonify

from src.prediction_pipeline import PredictionPipeline

app = Flask(__name__)

pipeline = None
try:
    pipeline = PredictionPipeline(artifacts_dir="artifacts")
except FileNotFoundError:
    # artifacts not generated yet, /health will report this clearly
    pipeline = None

FORM_FIELDS = [
    "policy_tenure", "age_of_car", "age_of_policyholder", "area_cluster",
    "population_density", "make", "segment", "model", "fuel_type",
    "max_torque", "max_power", "engine_type", "airbags", "is_esc",
    "is_adjustable_steering", "is_tpms", "is_parking_sensors",
    "is_parking_camera", "rear_brakes_type", "displacement", "cylinder",
    "transmission_type", "gear_box", "steering_type", "turning_radius",
    "length", "width", "height", "gross_weight", "is_front_fog_lights",
    "is_rear_window_wiper", "is_rear_window_washer",
    "is_rear_window_defogger", "is_brake_assist", "is_power_door_locks",
    "is_central_locking", "is_power_steering",
    "is_driver_seat_height_adjustable", "is_day_night_rear_view_mirror",
    "is_ecw", "is_speed_alert", "ncap_rating",
]


@app.route("/health")
def health():
    status = "ready" if pipeline is not None else "artifacts not found"
    return jsonify({"status": status})


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        if pipeline is None:
            error = "Model artifacts not found. Run the training notebooks first."
        else:
            try:
                form_data = {field: request.form.get(field) for field in FORM_FIELDS}
                form_data["policy_id"] = "ID_WEB_INPUT"
                df = pd.DataFrame([form_data])

                numeric_fields = [
                    "policy_tenure", "age_of_car", "age_of_policyholder",
                    "population_density", "make", "airbags", "displacement",
                    "cylinder", "gear_box", "turning_radius", "length",
                    "width", "height", "gross_weight", "ncap_rating",
                ]
                for field in numeric_fields:
                    df[field] = pd.to_numeric(df[field])

                prediction = pipeline.predict(df)
                result = {
                    "claim_predicted": bool(prediction["prediction"][0]),
                    "claim_probability": prediction["claim_probability"][0],
                }
            except Exception as exc:
                error = f"Could not generate a prediction: {exc}"

    return render_template("index.html", result=result, error=error)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if pipeline is None:
        return jsonify({"error": "Model artifacts not found"}), 503

    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Send a JSON body with policy/vehicle fields"}), 400

    records = payload if isinstance(payload, list) else [payload]
    try:
        df = pd.DataFrame(records)
        prediction = pipeline.predict(df)
        return jsonify(prediction)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
