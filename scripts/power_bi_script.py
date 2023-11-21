# Power BI Python script

import pandas as pd
from football_prediction import predict_match_winner

# Load the dataset in Power BI
dataset = pd.read_csv("C:/Users/Izukanji/Desktop/project/data/matches.csv")

# Select the features for prediction
predictors = dataset[["venue_code", "opp_code", "hour", "day_code"]].iloc[-1].values.reshape(1, -1)

# Call the predict_match_winner function
predicted_winner_prob = predict_match_winner(predictors)

# Use the prediction results as needed in Power BI
