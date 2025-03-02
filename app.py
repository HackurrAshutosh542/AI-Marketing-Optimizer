 
from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load("xgboost_best_model.pkl")

# Initialize Flask App
app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to AI Marketing Optimizer API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(input_features)[0]
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
