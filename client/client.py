import requests

# Define the endpoint URL
url = 'http://127.0.0.1:5000/predict'

# Define the emotion and feedback values
emotion = 'sad'
feedback = 'satisfied'

# Define the request payload as a JSON object
payload = {'emotion': emotion, 'feedback': feedback}

# Send the request to the server and get the response
response = requests.post(url, json=payload)

# Print the predicted activity value from the response
print(response.json()['activity'])