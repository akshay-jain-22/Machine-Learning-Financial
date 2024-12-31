from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained components
preprocessor = joblib.load('preprocessor.pkl')
pca = joblib.load('pca.pkl')
models = {
    'Logistic Regression': joblib.load('logistic_regression.pkl'),
    'Random Forest': joblib.load('random_forest.pkl'),
    'SVM': joblib.load('svm.pkl')
}

# Root route
@app.route('/')
def home():
    return 'Welcome to the Fraud Detection API!'

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input
        input_data = request.json

        # Validate input
        if not isinstance(input_data, dict):
            return jsonify({'error': 'Input data should be a JSON object.'}), 400

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Preprocess input
        processed_input = preprocessor.transform(input_df)
        pca_input = pca.transform(processed_input)

        # Predictions from all models
        predictions = {}
        for name, model in models.items():
            prediction = model.predict(pca_input)[0]  # Single prediction
            prediction_proba = model.predict_proba(pca_input)[0].tolist()  # Probabilities

            predictions[name] = {
                'prediction': int(prediction),  # Convert NumPy type to native Python type
                'probability': prediction_proba
            }

        # Return predictions
        return jsonify(predictions)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
