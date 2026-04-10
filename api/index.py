from flask import Flask, request, jsonify
import subprocess
import sys

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "ML Project API is running!",
        "status": "healthy"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        return jsonify({
            "prediction": "success",
            "received": data
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# This is required by Vercel
handler = app