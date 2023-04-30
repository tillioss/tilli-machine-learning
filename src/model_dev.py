# Import required libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import pickle

from src.util import data_etl

# Load the data
df = data_etl()

# Use pd.get_dummies to convert categorical columns to one-hot encoded columns
df = pd.get_dummies(df, columns=['emotion', 'feedback'])

# Separate the input features and target variable
X = df.drop('activity', axis=1)
y = df['activity']

# Create a logistic regression model
logreg = LogisticRegression()

# Fit the model on the training data
logreg.fit(X, y)

# Now, suppose you want to predict the activity for a given emotion and feedback:
# Encode the input values using the same one-hot encoding as the training data
emotion = 'happy'
feedback = 'satisfied'

original_col_order = X.columns
# Create a new DataFrame with a single row containing the encoded input values
input_data = pd.DataFrame({'emotion_' + emotion: [1], 'feedback_' + feedback: [1]})

# Add any missing columns to the input_data DataFrame
missing_cols = set(X.columns) - set(input_data.columns)
for c in missing_cols:
    input_data[c] = 0


# Reorder the columns in the input_data DataFrame to match the order of the columns in X
input_data = input_data[X.columns]

# Predict the activity using the trained logistic regression model
prediction = logreg.predict(input_data)
probabilities = logreg.predict_proba(input_data)

# Get the class names from the model
class_names = logreg.classes_

# Save the saved model from disk
joblib.dump(logreg, './model/logreg_model.joblib')
with open('./model/col_names.pk1', 'wb') as f:
    pickle.dump(original_col_order, f)

# Print the predicted activity and its probability
print(f"Prediction: {prediction[0]}")
print(f"Probabilities: {probabilities[0]}")
print(class_names)
