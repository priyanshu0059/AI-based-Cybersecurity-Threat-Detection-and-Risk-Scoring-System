# import sys
# import os
# import pandas as pd
# import json
# from flask import Blueprint, request, jsonify

# # Setup import path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# from utils.load_model import get_model
# from model.risk_scorer import calculate_risk_score
# from database.mongo_import import log_prediction  # Optional

# # === Setup Flask Blueprint ===
# predict_bp = Blueprint('predict', __name__, url_prefix='/predict')

# # === Load full pipeline model ===
# model = get_model()

# # === Load model metadata (including expected features and class names) ===
# MODEL_METADATA_PATH = os.path.join(os.path.dirname(__file__), '../../model/model_metadata.json')
# with open(MODEL_METADATA_PATH, 'r') as f:
#     metadata = json.load(f)

# expected_columns = metadata.get("categorical_features", [])
# class_names = metadata.get("class_names", [])

# @predict_bp.route('', methods=['POST'])
# def predict():
#     try:
#         # Step 1: Parse input JSON
#         data = request.get_json()
#         if not data:
#             return jsonify({"error": "No input data provided"}), 400

#         # Step 2: Convert input to DataFrame
#         input_df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
#         if not isinstance(input_df, pd.DataFrame):
#             return jsonify({"error": "Invalid input format"}), 400

#         # Step 3: Ensure all expected features are present
#         for col in expected_columns:
#             if col not in input_df.columns:
#                 input_df[col] = 0  # Always fill with 0 (numeric)
#         input_df = input_df[expected_columns]

#         # Force all columns to numeric (convert strings to float, set errors to 0.0)
#         input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0.0)

#         # Step 4: Predict class
#         pred_class_index = model.predict(input_df)[0]

#         # Step 5: Predict probability for all classes
#         proba_array = model.predict_proba(input_df)[0]

#         # Step 6: Map prediction to class label
#         try:
#             predicted_class = str(class_names[int(pred_class_index)])
#             probability = float(proba_array[int(pred_class_index)])
#         except Exception as e:
#             return jsonify({"error": f"Class mapping failed: {str(e)}"}), 500

#         # Step 7: Compute risk score
#         risk_score = calculate_risk_score(probability, pred_class_index, predicted_class)

#         # Step 8: Prepare response
#         result = {
#             "predicted_class": predicted_class,
#             "probability": round(probability, 6),
#             "risk_score": round(risk_score, 4)
#         }

#         # Step 9: Log to MongoDB (optional)
#         try:
#             log_prediction(data, predicted_class, probability, risk_score)
#         except Exception as log_error:
#             print("MongoDB log failed:", log_error)

#         return jsonify(result), 200

#     except Exception as e:
#         print("Prediction error:", e)
#         return jsonify({"error": str(e)}), 500
import sys
import os
import pandas as pd
import json
from flask import Blueprint, request, jsonify
import numpy as np

# Setup import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.load_model import get_model
from model.risk_scorer import calculate_risk_score
from database.mongo_import import log_prediction  # Optional

# === Setup Flask Blueprint ===
predict_bp = Blueprint('predict', __name__, url_prefix='/predict')

# === Load full pipeline model ===
model = get_model()

# === Load model metadata (including expected features and class names) ===
MODEL_METADATA_PATH = os.path.join(os.path.dirname(__file__), '../../model/model_metadata.json')
with open(MODEL_METADATA_PATH, 'r') as f:
    metadata = json.load(f)

expected_columns = metadata.get("features", [])
class_names = metadata.get("class_names", [])

@predict_bp.route('', methods=['POST'])
def predict():
    try:
        # Step 1: Parse input JSON
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        print(f"Input data: {data}")

        # Step 2: Convert input to DataFrame
        input_df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
        if not isinstance(input_df, pd.DataFrame):
            return jsonify({"error": "Invalid input format"}), 400
        print(f"Input DataFrame: {input_df.to_dict()}")

        # Step 3: Ensure all expected features are present
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0  # Always fill with 0 (numeric)
        input_df = input_df[expected_columns]
        input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
        print(f"Processed DataFrame: {input_df.to_dict()}")

        # Step 4: Predict class
        pred_class_index = model.predict(input_df)[0]
        print(f"pred_class_index: {pred_class_index}, type: {type(pred_class_index)}")

        # Ensure pred_class_index is an integer
        try:
            pred_class_index = int(np.round(pred_class_index))  # Convert to int, handle floats
        except (TypeError, ValueError) as e:
            return jsonify({"error": f"Invalid prediction index type: {str(e)}"}), 500

        # Step 5: Predict probability for all classes
        proba_array = model.predict_proba(input_df)[0]
        print(f"proba_array: {proba_array}, length: {len(proba_array)}")

        # Step 6: Map prediction to class label
        if not class_names:
            return jsonify({"error": "No class names provided in metadata"}), 500
        print(f"class_names: {class_names}, length: {len(class_names)}")

        if pred_class_index < 0 or pred_class_index >= len(class_names):
            return jsonify({"error": f"Prediction index {pred_class_index} out of range for class_names"}), 500

        try:
            predicted_class = str(class_names[pred_class_index])
            probability = float(proba_array[pred_class_index])
        except (IndexError, TypeError) as e:
            return jsonify({"error": f"Class mapping failed: {str(e)}"}), 500
        print(f"predicted_class: {predicted_class}, probability: {probability}")

        # Step 7: Compute risk score
        print(f"Calling calculate_risk_score with: probability={probability}, type={type(probability)}")
        print(f"pred_class_index={pred_class_index}, type={type(pred_class_index)}")
        print(f"predicted_class={predicted_class}, type={type(predicted_class)}")
        try:
            risk_score = calculate_risk_score(probability, pred_class_index, predicted_class)
        except Exception as e:
            return jsonify({"error": f"Risk score calculation failed: {str(e)}"}), 500
        print(f"risk_score: {risk_score}")

        # Step 8: Prepare response
        result = {
            "predicted_class": predicted_class,
            "probability": round(probability, 6),
            "risk_score": round(risk_score, 4)
        }

        # Step 9: Log to MongoDB (optional)
        try:
            log_prediction(data, predicted_class, probability, risk_score)
        except Exception as log_error:
            print("MongoDB log failed:", log_error)

        return jsonify(result), 200

    except Exception as e:
        print("Prediction error:", str(e))
        return jsonify({"error": str(e)}), 500