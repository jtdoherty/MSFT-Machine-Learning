#inpired by dataquest

import yfinance as yf
import os
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import precision_score

DATA_PATH = "msft_data.json"

if os.path.exists(DATA_PATH):
    # Read from file if we've already downloaded the data.
    with open(DATA_PATH) as f:
        msft_hist = pd.read_json(DATA_PATH)
else:
    msft = yf.Ticker("MSFT")
    msft_hist = msft.history(period="max")

    # Save file to json in case we need it later.  This prevents us from having to re-download it every time.
    msft_hist.to_json(DATA_PATH)

# Display microsoft stock price history so we can look at the structure of the data
msft_hist.head(5)

# Visualize microsoft stock prices
msft_hist.plot.line(y="Close", use_index=True)

#Preparing the data

# Ensure we know the actual closing price
data = msft_hist[["Close"]]
data = data.rename(columns = {'Close':'Actual_Close'})

# Setup our target.  This identifies if the price went up or down
data["Target"] = msft_hist.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["Close"]

# Shift stock prices forward one day, so we're predicting tomorrow's stock prices from today's prices.
msft_prev = msft_hist.copy()
msft_prev = msft_prev.shift(1)

# Create our training data
predictors = ["Close", "Volume", "Open", "High", "Low"]
data = data.join(msft_prev[predictors]).iloc[1:]

#Creating the machine learning model

# Create a random forest classification model.  Set min_samples_split high to ensure we don't overfit.
model = RandomForestClassifier(n_estimators=100, min_samples_split=200, random_state=1)

# Create a train and test set
train = data.iloc[:-100]
test = data.iloc[-100:]

model.fit(train[predictors], train["Target"])

# Evaluate error of predictions
preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index)

### precision_score(test["Target"], preds) #to see current results

combined = pd.concat({"Target": test["Target"],"Predictions": preds}, axis=1)
combined.plot()

##Backtesting


def backtest(data, model, predictors, start=1000, step=750):
    predictions = []
    # Loop over the dataset in increments
    for i in range(start, data.shape[0], step):
        # Split into train and test sets
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        
        # Fit the random forest model
        model.fit(train[predictors], train["Target"])
        
        # Make predictions
        preds = model.predict_proba(test[predictors])[:,1]
        preds = pd.Series(preds, index=test.index)
        preds[preds > .6] = 1
        preds[preds<=.6] = 0
        
        # Combine predictions and test values
        combined = pd.concat({"Target": test["Target"],"Predictions": preds}, axis=1)
        
        predictions.append(combined)
    
    return pd.concat(predictions)

predictions = backtest(data, model, predictors)

predictions["Predictions"].value_counts()

### precision_score(predictions["Target"], predictions["Predictions"]) #to see current results

##Improving accuracy
#We'll add in some rolling means, so the model can evaluate the current price against recent prices. We'll also look at the ratios between different indicators.

weekly_mean = data.rolling(7).mean()
quarterly_mean = data.rolling(90).mean()
annual_mean = data.rolling(365).mean()
weekly_trend = data.shift(1).rolling(7).mean()["Target"]


data["weekly_mean"] = weekly_mean["Close"] / data["Close"]
data["quarterly_mean"] = quarterly_mean["Close"] / data["Close"]
data["annual_mean"] = annual_mean["Close"] / data["Close"]

data["annual_weekly_mean"] = data["annual_mean"] / data["weekly_mean"]
data["annual_quarterly_mean"] = data["annual_mean"] / data["quarterly_mean"]
data["weekly_trend"] = weekly_trend

data["open_close_ratio"] = data["Open"] / data["Close"]
data["high_close_ratio"] = data["High"] / data["Close"]
data["low_close_ratio"] = data["Low"] / data["Close"]

full_predictors = predictors + ["weekly_mean", "quarterly_mean", "annual_mean", "annual_weekly_mean", "annual_quarterly_mean", "open_close_ratio", "high_close_ratio", "low_close_ratio", "weekly_trend"]
predictions = backtest(data.iloc[365:], model, full_predictors)


##print results here 
accuracy = precision_score(predictions["Target"], predictions["Predictions"])

trades_taken = predictions["Predictions"].value_counts()

print(accuracy) #0.597972972972973
print(trades_taken) # Predictions 0.0 - 8075 ; 1.0 - 296
#296 trades taken with ~60% win rate


#Steps to improve

#Improve the technique 
    #Determine how much profit you could earn by trading with this algorithm.

#Improve the algorithm
    #Use a smaller step size! Although it will take longer, it will improve accuracy.
    #Consider discarding older data and keeping only recent data within a specific window.
    #Experiment with a different machine learning algorithm.
    #Adjust the parameters of the random forest model or modify the prediction threshold.

#Add in more predictors
    #Include post-close and pre-open activity.
    #Factor in early trading hours.
    #Monitor trading on exchanges that open before the NYSE to gauge global sentiment.
    #Track economic indicators.
    #Consider interest rates.
    #Pay attention to significant economic news.
    #Mark key dates.
    #Account for dividends.
    #Assess external factors like elections.
    #Note company milestones.
    #Include earnings calls.
    #Review analyst ratings.
    #Factor in major announcements.
    #Observe prices of related stocks.
    #Consider other companies within the same sector.
    #Evaluate key partners, customers, and stakeholders.