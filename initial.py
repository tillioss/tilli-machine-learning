import pandas as pd


def categorical_to_numeric_df(training_df):
    training_df['feedback'].replace(['unsatisfied', 'satisfied', 'neutral'], [0, 1, 2], inplace=True)
    training_df['activity'].replace(
        ['BubblePopActivity', 'YogaActivity', 'ColoringActivity', 'RainbowActivity', 'SelfHugActivity',
         'WaterDrinkingActivity', 'FeedTheFrogActivity'], [0, 1, 2, 3, 4, 5, 6], inplace=True)
    training_df['emotion'].replace(['happy', 'sad', 'angry', 'surprised', 'scared', 'disgusted'], [0, 1, 2, 3, 4, 5],
                                   inplace=True)
    return training_df


df = pd.read_json('data/data_1.json').reset_index()

journey_df = pd.DataFrame(list(df['users']))[['journeys']].dropna()

training_df = pd.DataFrame()

for i in range (len(journey_df)):
    if len(training_df) == 0:
        training_df = pd.DataFrame(list(journey_df.iloc[1])[0].values())
    else:
        training_df = training_df.append(pd.DataFrame(list(journey_df.iloc[i])[0].values()))

training_df = categorical_to_numeric_df(training_df)
