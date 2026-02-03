# """
# app.py
# A simple Flask dashboard for the AI-Based IDS.

# - Shows a dashboard page with recent detections (in-memory).
# - Provides an API endpoint /api/predict to accept JSON feature vector and return classification.
# - If a saved model (model.pkl) is present (created by ai_ids.py), it will load it.
# - Otherwise, it will run a demo mode returning random predictions.

# Run:
#     pip install -r requirements.txt
#     flask run --port=5000
# or
#     python app.py
# """

# from flask import Flask, render_template, request, jsonify
# import os
# import joblib
# import random
# from datetime import datetime

# app = Flask(__name__)

# MODEL_PATH = "model.pkl"
# model_bundle = None
# if os.path.exists(MODEL_PATH):
#     try:
#         model_bundle = joblib.load(MODEL_PATH)
#         model = model_bundle['model']
#         scaler = model_bundle.get('scaler', None)
#         le_dict = model_bundle.get('label_encoders', {})
#         print("Model loaded for dashboard.")
#     except Exception as e:
#         print("Failed to load model:", e)
#         model = None
#         scaler = None
# else:
#     model = None
#     scaler = None

# # In-memory recent alerts (list of dicts)
# recent_alerts = []

# @app.route("/")
# def index():
#     # show up to last 20 alerts
#     return render_template("index.html", alerts=list(recent_alerts)[-20:][::-1])

# @app.route("/api/predict", methods=["POST"])
# def api_predict():
#     data = request.get_json()
#     if not data or 'features' not in data:
#         return jsonify({"error": "Send JSON with key 'features' containing feature dict"}), 400
#     features = data['features']
#     # If model present, attempt prediction; otherwise demo random
#     if model is not None:
#         try:
#             # Ensure all features exist in the right order
#             X = []
#             for fname in model.feature_names_in_:
#                 X.append(features.get(fname, 0))
#             import numpy as np
#             X = np.array([X])
#             if scaler is not None:
#                 X = scaler.transform(X)
#             pred = model.predict(X)[0]
#             label = "Attack" if int(pred)==1 else "Normal"
#         except Exception as e:
#             return jsonify({"error": f"Model prediction failed: {e}"}), 500
#     else:
#         label = random.choice(["Normal", "Attack"])
#     alert = {
#         "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         "src_ip": data.get("src_ip", "unknown"),
#         "dst_ip": data.get("dst_ip", "unknown"),
#         "label": label,
#         "details": data.get("meta", "")
#     }
#     recent_alerts.append(alert)
#     return jsonify({"label": label, "alert": alert})

# if __name__ == "__main__":
#     # bundled simple server run
#     app.run(host="0.0.0.0", port=5000, debug=True)


"""
app.py - Flask dashboard for AI-Based Network Intrusion Detection System (IDS)

Features:
- Loads model from 'model.pkl' (supports joblib bundle with {'model', 'scaler', 'label_encoders'})
  or a direct sklearn model saved with joblib/pickle.
- /               -> dashboard showing recent alerts (templates/index.html expected)
- /api/predict    -> POST JSON API: {"features": {...}, "src_ip":"", "dst_ip":"", "meta":""}
                     returns prediction, attack_type, alert
- Handles multi-class (attack name) models (like NSL-KDD labels) and maps attack names
  to higher-level categories: DoS, Probe, R2L, U2R.
- If model is binary (0/1), returns 'Attack' vs 'Normal' (attack type may be unknown).

Usage:
    pip install -r requirements.txt
    python app.py
"""

import os
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify
from datetime import datetime
import traceback

app = Flask(__name__)

MODEL_PATH = "model.pkl"
model = None
scaler = None
label_encoders = {}
model_loaded = False

# Try loading model (supports joblib bundle created by ai_ids.py)
if os.path.exists(MODEL_PATH):
    try:
        bundle = joblib.load(MODEL_PATH)
        # bundle might be a dict (our ai_ids.py saves a dict) or a raw model
        if isinstance(bundle, dict):
            model = bundle.get('model')
            scaler = bundle.get('scaler', None)
            # label encoders may be stored under different key names
            label_encoders = bundle.get('label_encoders', bundle.get('label_encoders', {}))
        else:
            model = bundle
        model_loaded = model is not None
        print("Model loaded from", MODEL_PATH)
    except Exception as e:
        print("Failed to load model.pkl:", e)
        traceback.print_exc()
        model = None
else:
    print("No model.pkl found. Dashboard will run in demo/random mode.")


# Mapping of many NSL-KDD attack names into categories (DoS, Probe, R2L, U2R)
DOS_SET = {
    "back", "land", "neptune", "pod", "smurf", "teardrop", "apache2", "mailbomb"
}
PROBE_SET = {
    "satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"
}
R2L_SET = {
    "ftp_write", "guess_passwd", "imap", "phf", "spy", "warezclient", "warezmaster", "multihop"
}
U2R_SET = {
    "buffer_overflow", "loadmodule", "perl", "rootkit", "httptunnel", "ps"
}

def map_attack_name_to_category(attack_name: str) -> str:
    """
    Given an attack name (e.g. 'neptune', 'satan') return the high-level category.
    If it's 'normal' or similar, return 'Normal Traffic'.
    """
    if attack_name is None:
        return "Unknown"
    name = str(attack_name).lower().strip()
    if name in ("normal", "normal."):
        return "Normal Traffic"
    if name in DOS_SET:
        return "DoS Attack (Denial of Service)"
    if name in PROBE_SET:
        return "Probe Attack (Network Scanning)"
    if name in R2L_SET:
        return "R2L Attack (Remote to Local)"
    if name in U2R_SET:
        return "U2R Attack (User to Root)"
    # fallback checks: many NSL-KDD variants have suffixes; check substring
    for s in DOS_SET:
        if s in name:
            return "DoS Attack (Denial of Service)"
    for s in PROBE_SET:
        if s in name:
            return "Probe Attack (Network Scanning)"
    for s in R2L_SET:
        if s in name:
            return "R2L Attack (Remote to Local)"
    for s in U2R_SET:
        if s in name:
            return "U2R Attack (User to Root)"
    # Unknown attack name -> return generic Attack
    return f"Attack (type: {attack_name})"


# In-memory recent alerts (for dashboard display)
recent_alerts = []  # list of dicts: {time, src_ip, dst_ip, raw_label, category_label, details}


@app.route("/")
def index():
    # Show last 20 alerts in reverse chronological order
    last_alerts = list(recent_alerts)[-20:][::-1]
    return render_template("index.html", alerts=last_alerts)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Expects JSON:
    {
      "features": { "f1": val1, "f2": val2, ... }   # values for each model feature (optional)
      "src_ip": "10.0.0.1",
      "dst_ip": "192.168.1.2",
      "meta": "optional details"
    }
    Returns JSON with prediction and attack type.
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "Send JSON payload with 'features' key."}), 400

        features = data.get("features", {})
        src_ip = data.get("src_ip", "unknown")
        dst_ip = data.get("dst_ip", "unknown")
        meta = data.get("meta", "")

        # If no model loaded, return demo random result
        if not model_loaded or model is None:
            import random
            demo_label = random.choice(["normal", "neptune", "satan", "guess_passwd", "buffer_overflow"])
            category = map_attack_name_to_category(demo_label)
            alert = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "raw_label": demo_label,
                "category": category,
                "details": meta
            }
            recent_alerts.append(alert)
            return jsonify({"label": demo_label, "category": category, "alert": alert})

        # When model is available:
        # Prepare input vector in the same order model expects.
        # If model has attribute feature_names_in_, use that ordering.
        if hasattr(model, "feature_names_in_"):
            feature_names = list(model.feature_names_in_)
        else:
            # fallback: use keys from provided features (order not guaranteed)
            feature_names = list(features.keys())

        # Build input array
        x_list = []
        for fname in feature_names:
            # if user provided, use it; else default to 0
            val = features.get(fname, 0)
            try:
                x_list.append(float(val))
            except:
                # non-numeric -> try to map with label encoder if present
                le = label_encoders.get(fname)
                if le:
                    try:
                        encoded = le.transform([str(val)])[0]
                        x_list.append(float(encoded))
                    except:
                        x_list.append(0.0)
                else:
                    x_list.append(0.0)
        X = np.array([x_list])

        # scale if scaler present
        if scaler is not None:
            try:
                X = scaler.transform(X)
            except Exception:
                # if scaler fails, continue with raw X
                pass

        # Predict
        raw_pred = model.predict(X)[0]

        # If model is multiclass with string labels (e.g., 'neptune'), raw_pred may be string
        # If model is numeric (0/1 or integer indexes), try to map via model.classes_
        raw_label = raw_pred
        category = "Unknown"

        # If the model uses classes_ and raw_pred is an index or class label, handle both
        try:
            # If raw_pred is a numeric index mapping to classes_, sklearn usually returns the class label itself.
            # So raw_pred may already be the label; try to map it to category:
            category = map_attack_name_to_category(raw_label)
        except Exception:
            category = "Unknown"

        # If the model is binary (0,1) and categories are not specific attack names, handle binary mapping
        if (isinstance(raw_label, (int, np.integer)) or str(raw_label).isdigit()) and hasattr(model, "classes_"):
            # sometimes classes_ = [0,1] - if so convert
            try:
                # if classes_ are strings, get the actual class string
                classes = list(model.classes_)
                # raw_label likely equals actual class label, not index; but handle index possibility:
                # if raw_label is numeric and not in classes, try to interpret as index
                if raw_label in classes:
                    label_value = raw_label
                else:
                    # if raw_label is like 1 and classes are [0,1], treat as label_value
                    label_value = raw_label
                # If label_value indicates normal (0) or attack (1)
                if str(label_value) in ("0", "0.0", "normal"):
                    category = "Normal Traffic"
                elif str(label_value) in ("1", "1.0"):
                    category = "Attack (type unknown)"
                else:
                    # fallback try to map as string
                    category = map_attack_name_to_category(label_value)
            except Exception:
                category = map_attack_name_to_category(raw_label)

        # Build alert and append to in-memory list
        alert = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "raw_label": str(raw_label),
            "category": category,
            "details": meta
        }
        recent_alerts.append(alert)

        return jsonify({
            "label": str(raw_label),
            "category": category,
            "alert": alert
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# Simple web form predict route (if you add a form in index.html that posts to /predict)
@app.route("/predict", methods=["POST"])
def predict_form():
    try:
        # Collect form fields: assume user posts a set of named fields
        form_dict = request.form.to_dict()
        # Prepare features by removing meta fields if present
        src_ip = form_dict.pop("src_ip", "unknown")
        dst_ip = form_dict.pop("dst_ip", "unknown")
        meta = form_dict.pop("meta", "")

        # Convert remaining fields to floats where possible
        features = {}
        for k, v in form_dict.items():
            try:
                features[k] = float(v)
            except:
                features[k] = v

        # Use same logic as API to predict
        response = api_predict_from_internal(features, src_ip, dst_ip, meta)
        # response is JSON-like dict
        return render_template("index.html", prediction_text=f"{response.get('category')} ({response.get('label')})", alerts=list(recent_alerts)[-20:][::-1])
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}", alerts=list(recent_alerts)[-20:][::-1])


def api_predict_from_internal(features: dict, src_ip: str, dst_ip: str, meta: str):
    """
    Helper to perform prediction programmatically (used by predict_form).
    Returns dict similar to api_predict response.
    """
    # call the same logic used in api_predict by creating a fake request JSON
    fake_json = {"features": features, "src_ip": src_ip, "dst_ip": dst_ip, "meta": meta}
    # Flask's test_request_context is an overkill; we just call the function logic directly (slightly hacky)
    with app.test_request_context(json=fake_json):
        resp = api_predict()
        # api_predict returns a Flask response, extract JSON
        if hasattr(resp, "get_json"):
            return resp.get_json()
        else:
            # In case of tuple (response, status)
            try:
                return resp[0].get_json()
            except:
                return {"label": "error", "category": "error", "alert": {}}


if __name__ == "__main__":
    # Run the app
    # In production, use a proper WSGI server; for demo this is fine.
    app.run(host="0.0.0.0", port=5000, debug=True)
