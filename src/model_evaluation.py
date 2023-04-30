from util import data_etl

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

import warnings
warnings.filterwarnings("ignore")

# Load the input data frame
df = data_etl()

# One-hot encode the 'emotion' and 'feedback' columns
encoder = OneHotEncoder()
X = encoder.fit_transform(df[['emotion', 'feedback']]).toarray()

# Get the 'activity' column as the target variable
y = df['activity']

# Define a list of machine learning models to try out
models = [
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier()),
    ('SVM', SVC()),
    ('MLP', MLPClassifier()),
    ('Deep Learning', Sequential([
        Dense(32, activation='relu', input_dim=X.shape[1]),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ]).compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']))]

# Evaluate each model using cross-validation
for name, model in models:
    scores = cross_val_score(model, X, y, cv=5)
    print(f'{name}: {scores.mean():.3f} (std={scores.std():.3f})')
