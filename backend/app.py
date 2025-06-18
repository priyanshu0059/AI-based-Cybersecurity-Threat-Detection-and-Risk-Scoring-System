from flask import Flask, jsonify
from flask_cors import CORS
from routes.predict import predict_bp


import logging

app = Flask(__name__)

# ✅ Enable CORS for all routes
CORS(app)

# ✅ Register prediction route blueprint
app.register_blueprint(predict_bp)

# ✅ Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "🛡️ Cybersecurity AI API is live", "status": "OK"}), 200

# ✅ Main entry point
if __name__ == "__main__":
    logger.info("🚀 Starting Flask server on http://localhost:5000 ...")
    app.run(debug=True, host="0.0.0.0", port=5000)
