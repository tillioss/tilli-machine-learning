import pandas as pd
import matplotlib.pyplot as plt


def categorical_to_numeric_df(training_df):
    """
    This function converts categorical values to a number value range
    :param training_df:
    :return:
    """
    training_df['feedback'].replace(['unsatisfied', 'satisfied', 'neutral'], [0, 1, 2], inplace=True)
    training_df['activity'].replace(
        ['BubblePopActivity', 'YogaActivity', 'ColoringActivity', 'RainbowActivity', 'SelfHugActivity',
         'WaterDrinkingActivity', 'FeedTheFrogActivity'], [0, 1, 2, 3, 4, 5, 6], inplace=True)
    training_df['emotion'].replace(['happy', 'sad', 'angry', 'surprised', 'scared', 'disgusted'], [0, 1, 2, 3, 4, 5],
                                   inplace=True)
    return training_df


def visualize(df):
    """
    This function visualizes the spread of user interactions
    :param df:
    :return:
    """
    df['feedback'].replace(['unsatisfied', 'satisfied', 'neutral'], ['red', 'blue', 'green'], inplace=True)
    plt.scatter(df.emotion, df.activity, s=200, c=df.feedback, cmap='gray')
    plt.tight_layout()
    plt.savefig("data/data_1.png")


def data_etl():
    df = pd.read_json('data/data_1.json').reset_index()
    journey_df = pd.DataFrame(list(df['users']))[['journeys']].dropna()
    training_df = pd.DataFrame()

    for i in range (len(journey_df)):
        if len(training_df) == 0:
            training_df = pd.DataFrame(list(journey_df.iloc[1])[0].values())
        else:
            training_df = training_df.append(pd.DataFrame(list(journey_df.iloc[i])[0].values()))
    training_df = training_df[training_df['activity'] != 'FeedTheFrogActivity']
    training_df = training_df[training_df['emotion'] != ""]
    return training_df

training_df = data_etl()
visualize(training_df)
# training_df = categorical_to_numeric_df(training_df)
