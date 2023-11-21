import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def predict_match_winner_logistic_regression(matches, team1, team2):
    # Filtering the data according to the teams provided
    team_data = matches[(matches["team"] == team1) | (matches["team"] == team2)].copy()

    # Preprocess the data
    team_data.loc[:, "target"] = (team_data["result"] == "W").astype("int")
    team_data.loc[:, "date"] = pd.to_datetime(team_data["date"])
    team_data.loc[:, "venue_code"] = team_data["venue"].astype("category").cat.codes
    team_data.loc[:, "opp_code"] = team_data["opponent"].astype("category").cat.codes
    team_data.loc[:, "hour"] = team_data["time"].str.replace(":.+", "", regex=True).astype("int")
    team_data.loc[:, "day_code"] = team_data["date"].dt.dayofweek

    # Split the data into training and testing sets
    train, test = train_test_split(team_data, test_size=0.2, random_state=1)

    # Specify predictors
    predictors = ["venue_code", "opp_code", "hour", "day_code"]

    # Initialize the Logistic Regression model
    logistic_reg = LogisticRegression()

    # Train the model
    logistic_reg.fit(train[predictors], train["target"])

    # Use the latest match data for prediction
    latest_match_data = team_data.iloc[-1][predictors].values.reshape(1, -1)

    # Predict the probability of winning for the latest match
    predicted_winner_prob = logistic_reg.predict_proba(latest_match_data)[:, 1]

    # Map predicted winner code to team name
    #predicted_winner = team1 if predicted_winner_prob > 0.5 else team2

    if predicted_winner_prob > 0.5:
        predicted_winner = team1
        probability_winner = predicted_winner_prob
    elif predicted_winner_prob < 0.5:
        predicted_winner = team2
        probability_winner = 1 - predicted_winner_prob  # Probability of the other team winning
    else:
        predicted_winner = "draw"
        probability_winner = 0.5  # Probability of a draw

    return predicted_winner, predicted_winner_prob[0]

# reading the matches.csv dataset:
matches_data = pd.read_csv("matches.csv", index_col=0)
team1 = "Manchester City"
team2 = "Manchester United"

#calling the predict_match_winner_logistic_regression(data,team1,team2) function
predicted_winner, probability_team1_win = predict_match_winner_logistic_regression(matches_data, team1, team2)
#printing the results from the function
print(f'The predicted winner between {team1} and {team2} is: {predicted_winner}')
print(f'The predicted probability that {team1} will win: {probability_team1_win:}')
