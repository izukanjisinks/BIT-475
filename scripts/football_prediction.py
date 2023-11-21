# football_prediction.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

def train_logistic_regression_model(dataset_path):
    # Load the dataset
    dataset = pd.read_csv(dataset_path)

    # Preprocess the data
    dataset["target"] = (dataset["result"] == "W").astype("int")
    dataset["date"] = pd.to_datetime(dataset["date"])
    dataset["venue_code"] = dataset["venue"].astype("category").cat.codes
    dataset["opp_code"] = dataset["opponent"].astype("category").cat.codes
    dataset["hour"] = dataset["time"].str.replace(":.+", "", regex=True).astype("int")
    dataset["day_code"] = dataset["date"].dt.dayofweek

    # Split the data into training and testing sets
    train, _ = train_test_split(dataset, test_size=0.2, random_state=1)

    # Specify predictors
    predictors = ["venue_code", "opp_code", "hour", "day_code"]

    # Initialize the Logistic Regression model
    logistic_reg = LogisticRegression()

    # Train the model
    logistic_reg.fit(train[predictors], train["target"])

    # Save the trained model using pickle
    with open("football_prediction_model.pkl", "wb") as model_file:
        pickle.dump(logistic_reg, model_file)

def predict_match_winner(predictors):
    # Load the trained model
    with open("football_prediction_model.pkl", "rb") as model_file:
        logistic_reg = pickle.load(model_file)

    # Predict the probability of winning
    predicted_winner_prob = logistic_reg.predict_proba(predictors)[:, 1]

    return predicted_winner_prob[0]

if __name__ == "__main__":
    # Example usage for training the model (replace with your dataset path)
    train_logistic_regression_model("C:/Users/Izukanji/Desktop/project/data/matches.csv")
