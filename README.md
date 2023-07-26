# Machine Learning Model Server

This repository contains code for running a machine learning model server using Docker and Python. The server serves predictions based on a trained logistic regression model. It also includes tests, a client server for manual testing, and additional files needed at runtime.

## Instructions

1. To run the server with the machine learning model, follow these steps:

   - Build the Docker image using the provided Dockerfile
     `docker build -t ml-model-server .`
   - Run the Docker container to start the server on port 8080.
     `docker run -p 8080:8080 ml-model-server`

2. The tests are located in the `tests` folder and are executed via the CI/CD pipeline set up in the `test.yml` file located in the `workflows` folder (using GitHub Actions). To add new tests, create a new file in the `tests` folder and add the relevant test cases. Currently, the file `tests/test_server.py` is created for testing the server.

3. To train a new model, create a folder named `data` and add a JSON file (`data_1.json`) with the required data. (Note: The file format may be updated in the future.)

4. The `client/client.py` file contains a client server that can be used for manual testing of the server. Modify the `url`, `emotion`, and `feedback` variables to match your server's endpoint and the desired input values.

5. The `models` folder contains additional information and files needed at runtime. Currently, the following files are included:

- `logreg_model.joblib`: A logistic regression model trained with the data.
- `col_names.pk1`: Column names of the original dataset.

6. The `src/model_dev.py` file demonstrates the model development process. It includes the following steps:

- Load the data using the `data_etl()` function from the `src/util.py` file.
- Convert categorical columns to one-hot encoded columns.
- Separate the input features and target variable.
- Create and train a logistic regression model.
- Predict the activity for a given emotion and feedback.
- Save the trained model and the column names for future use.

7. Utility functions are stored in the `src/util.py` file. These functions include:

- `categorical_to_numeric_df()`: Converts categorical values to a numeric range.
- `visualize()`: Visualizes the spread of user interactions.
- `data_etl()`: Loads and preprocesses the data.

8. The `src/model_evaluation.py` file is under development and contains code to evaluate the performance of different models.

Feel free to explore the repository and adapt it to your needs.

### Join our Slack Community

Join our Slack community to connect with other members, ask questions, and collaborate on projects. [Slack Invite Link](https://join.slack.com/t/tilliopensour-wyp9205/shared_invite/zt-206f4f11s-HoII8Kob45f6WK3GPIIi6g)
