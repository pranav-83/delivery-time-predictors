import os

import joblib
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
CORS(app)

model = joblib.load(os.path.join(BASE_DIR, "delivery_time_model.pkl"))

CATEGORY_COLS = ["Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"]


@app.route("/")
def home():
    """Serve the web page, so one server handles both page and predictions."""
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/health")
def health():
    return "Delivery Time Prediction API is running"


@app.route("/options")
def options():
    """Send the real category values from the dataset to the web page."""
    try:
        df = pd.read_csv(os.path.join(BASE_DIR, "delivery.csv"))
        return jsonify({c: df[c].dropna().unique().tolist() for c in CATEGORY_COLS})
    except Exception:
        return jsonify({}), 404


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        order = pd.DataFrame([{
            "Distance_km": float(data["Distance_km"]),
            "Weather": data["Weather"],
            "Traffic_Level": data["Traffic_Level"],
            "Time_of_Day": data["Time_of_Day"],
            "Vehicle_Type": data["Vehicle_Type"],
            "Preparation_Time_min": float(data["Preparation_Time_min"]),
            "Courier_Experience_yrs": float(data["Courier_Experience_yrs"]),
        }])

        prediction = model.predict(order)[0]
        return jsonify({"estimated_time_min": round(float(prediction), 1)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)