from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load your ML model
# model = joblib.load("models/model.pkl")  ← uncomment when you have a model

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "ML API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        features = np.array(data["features"]).reshape(1, -1)

        # prediction = model.predict(features)  ← uncomment when you have a model
        # return jsonify({"prediction": prediction.tolist()})

        return jsonify({"message": "Prediction endpoint ready"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    app.run(debug=True)