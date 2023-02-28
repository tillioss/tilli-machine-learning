import joblib
import numpy as np
from flask import Flask, jsonify, request
import joblib
import logging

app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)

# Load the LabelEncoders
le_act = joblib.load('../model/le_act.joblib')
le_emo = joblib.load('../model/le_emo.joblib')
le_fbk = joblib.load('../model/le_fbk.joblib')

# Load the KNN model
knn = joblib.load('../model/knn_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the emotion and feedback values from the request
    emotion = le_emo.transform([request.json['emotion']])
    feedback = le_fbk.transform([request.json['feedback']])

    # Predict the activity using the KNN model
    prediction = knn.predict(np.array([[emotion, feedback]]).reshape(-1, 2))

    # Decode the predicted activity value
    activity_pred = le_act.inverse_transform(prediction)

    # Return the predicted activity as a JSON response
    return jsonify({'activity': activity_pred[0]})

# Test api for outside access tested.
@app.route('/predict/test', methods=['GET'])
def predicthi():
    return "Predict Till Test"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    # app.run(host='127.0.0.1', port=5000)