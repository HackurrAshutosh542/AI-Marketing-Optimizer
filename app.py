from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained XGBoost model
model = joblib.load("xgboost_best_model.pkl")

# Initialize Flask App
app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to AI Marketing Optimizer API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Convert input data into a NumPy array
    input_features = np.array(data['features']).reshape(1, -1)
    
    # Check for feature mismatch
    expected_features = model.n_features_in_
    if input_features.shape[1] != expected_features:
        return jsonify({'error': f'Feature shape mismatch: expected {expected_features}, got {input_features.shape[1]}'})
    
    # Make prediction
    prediction = model.predict(input_features)[0]
    
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
