import joblib
import numpy as np
from flask import Flask, jsonify, request
import joblib
import logging
from datetime import date
import random
import os
import pickle
import pandas as pd

# Get the current directory of the Python file
base_dir = os.path.dirname(os.path.relpath(__file__))

model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model', 'logreg_model.joblib')
column_name_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model', 'col_names.pk1')

app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)


# Load the KNN model
logreg = joblib.load(model_path)
with open(column_name_path, 'rb') as f:
    col_names = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():

    today = date.today()  # get today's date
    jan_1 = date(today.year, 1, 1)  # create a date object for January 1st of this year
    delta = today - jan_1  # calculate the time difference between January 1st and today
    days_since_jan_1 = delta.days + 1  # add 1 to include January 1st

    odd_or_even = days_since_jan_1%2

    if odd_or_even == 0:
        # Return the predicted activity as a JSON response
        activity = ['BubblePopActivity', 'YogaActivity', 'ColoringActivity', 'RainbowActivity', 'SelfHugActivity',
         'WaterDrinkingActivity']
        print("Inside If", days_since_jan_1)
        return jsonify({'activity': random.choice(activity)})

    else:
        # Get the emotion and feedback values from the request
        print("Inside else", days_since_jan_1)

        emotion = request.json['emotion']
        feedback = request.json['feedback']

        # Create a new DataFrame with a single row containing the encoded input values
        input_data = pd.DataFrame({'emotion_' + emotion: [1], 'feedback_' + feedback: [1]})

        # Add any missing columns to the input_data DataFrame
        missing_cols = set(col_names) - set(input_data.columns)
        for c in missing_cols:
            input_data[c] = 0

        # Reorder the columns in the input_data DataFrame to match the order of the columns in X
        input_data = input_data[col_names]

        prediction = logreg.predict(input_data)

        # Return the predicted activity as a JSON response
        return jsonify({'activity': prediction[0]})

# Test api for outside access tested.
@app.route('/predict/test', methods=['GET'])
def predicthi():
    return "Predict Till Test"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
    # app.run(host='127.0.0.1', port=5000)