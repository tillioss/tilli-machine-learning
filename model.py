import pandas as pd
from sklearn.neighbors import NearestNeighbors
from util import categorical_to_numeric_df, data_etl

# Load the data
processed_df = data_etl()
data = categorical_to_numeric_df(processed_df)

#TODO Need to add a more generalised missing value handling mechanism

data = data[data['emotion'] !=""]

# data = pd.read_csv('data.csv')

# Prepare the data for KNN
X = data.drop(['feedback'], axis=1)  # 'feedback' is the target variable
y = data['feedback']

# Instantiate the KNN model
knn = NearestNeighbors(n_neighbors=5)

# Fit the model to the data
knn.fit(X, y)

# Predict using the model
positive_feedback = [1]
new_emotion = [3,1]  # Replace with the emotion value to be predicted
distances, indices = knn.kneighbors([new_emotion])

# Get the top recommendations and calculate their feedback
top_indices = indices[0]
top_activities = data.loc[top_indices, 'activity']
top_feedbacks = data.loc[top_indices, 'feedback']
positive_activities = top_activities[top_feedbacks == 1]

# Print the recommended activities with positive feedback
print("Recommended activities with positive feedback:")
print(positive_activities)
