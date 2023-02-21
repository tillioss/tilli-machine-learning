import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from util import categorical_to_numeric_df, data_etl
import numpy as np
import joblib

# Load the data
processed_df = data_etl()
data = processed_df[processed_df['emotion'] !=""]
# data = categorical_to_numeric_df(processed_df)

#TODO Need to add a more generalised missing value handling mechanism

# Load the dataset
# data = pd.read_csv('your_dataset.csv')

# Encode categorical variables
le_act = LabelEncoder()
data['activity'] = le_act.fit_transform(data['activity'])

le_emo = LabelEncoder()
data['emotion'] = le_emo.fit_transform(data['emotion'])

le_fbk = LabelEncoder()
data['feedback'] = le_fbk.fit_transform(data['feedback'])

# Separate features and target variable
X = data[['emotion', 'feedback']]
y = data['activity']

X = X.values.reshape(-1, 2)
# y = y.values.reshape(-1, 1)
# Create KNN model with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Fit the model
knn.fit(X, y)

# Predict an activity for given emotion and satisfied feedback
emotion = le_emo.transform(['sad'])
feedback = le_fbk.transform(['satisfied'])

prediction = knn.predict(np.array([[emotion, feedback]]).reshape(-1, 2))

# Decode the prediction
activity_pred = le_act.inverse_transform(prediction)

# Print the predicted activity
print(f"The predicted activity is {activity_pred[0]}")

# Save the LabelEncoders
joblib.dump(le_act, '../model/le_act.joblib')
joblib.dump(le_emo, '../model/le_emo.joblib')
joblib.dump(le_fbk, '../model/le_fbk.joblib')

# Save the KNN model
joblib.dump(knn, '../model/knn_model.joblib')