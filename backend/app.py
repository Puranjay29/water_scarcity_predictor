from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model, scaler, and label encoder
model = joblib.load('backend/water_stress_model.pkl')
scaler = joblib.load('backend/scaler.pkl')
le = joblib.load('backend/label_encoder.pkl')

# Load preloaded state data
state_data = pd.read_csv('backend/state_data.csv')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request
    data = request.json
    state = data['state']
    year = data['year']

    # Find the corresponding row in the preloaded data
    row = state_data[(state_data['State Name'] == state) & (state_data['Year'] == year)]
    if row.empty:
        return jsonify({'error': 'Data not found for the selected state and year'}), 404

    # Prepare the input data
    input_data = row.drop('Total Water Demand', axis=1).values
    input_data[0, 0] = le.transform([state])[0]  # Encode state name

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)

    # Return the result
    return jsonify({'prediction': float(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)