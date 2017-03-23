import json
import numpy as np
from os.path import join
from tqdm import tqdm
import statsmodels.api as stats_api
from datetime import datetime
import pandas as pd
import re
from sklearn.metrics import mean_absolute_error

hashtags = {
    'gohawks' : 188136,
    'nfl' : 259024,
    'superbowl' : 1348767
}

# Dictionary to cache models
models = {}

def load_dataframe(filename, lcount):
    with open(filename, 'r') as f:
        df = pd.DataFrame(index=range(lcount), columns=['dateTime', 'tweetCount', 'retweetCount', 'followerSum', 'maxFollowers', 'impressionCount',
            'favoriteCount', 'rankingScore', 'userID', 'numberLongTweets'])
        for i, line in tqdm(enumerate(f), total=lcount):
            tweet_data = json.loads(line)
            date = datetime.fromtimestamp(tweet_data['firstpost_date'])
            df.set_value(i, 'dateTime', date)
            df.set_value(i, 'tweetCount', 1)
            df.set_value(i, 'retweetCount', tweet_data['metrics']['citations']['total'])
            df.set_value(i, 'followerSum', tweet_data['author']['followers'])
            df.set_value(i, 'maxFollowers', tweet_data['author']['followers'])
            df.set_value(i, 'impressionCount', tweet_data.get('metrics').get('impressions'))
            df.set_value(i, 'favoriteCount', tweet_data.get('tweet').get('favorite_count'))
            df.set_value(i, 'rankingScore', tweet_data.get('metrics').get('ranking_score'))
            df.set_value(i, 'userID', tweet_data.get('tweet').get('user').get('id'))
            df.set_value(i, 'longTweet', tweet_data.get('title') > 100)

    return df

def fetch_matrix(df):
    df = df.set_index('dateTime')
    hourlySeries = df.groupby(pd.TimeGrouper(freq='60Min'))

    X = np.zeros((len(hourlySeries), 10))
    Y = np.zeros((len(hourlySeries), 1))

    # Extract features for each hourly interval
    for i,(interval,group) in enumerate(hourlySeries):
        X[i, 0] = group.tweetCount.sum()        # Number of tweets
        X[i, 1] = group.retweetCount.sum()      # Number of retweets
        X[i, 2] = group.followerSum.sum()       # Sum of follower counts
        X[i, 3] = group.maxFollowers.max()      # Maximum size following
        X[i, 4] = interval.hour                 # Hour of the day
        X[i, 5] = group.impressionCount.sum()   # Sum of impression count
        X[i, 6] = group.favoriteCount.sum()     # Sum of favorites
        X[i, 7] = group.rankingScore.sum()      # Sum of rankings
        X[i, 8] = group.userID.nunique()        # Number of unique users tweeting
        X[i, 9] = group.longTweet.sum()         # Number of long tweets

        Y[i, 0] = group.tweetCount.sum()


    # Shift X and Y forward by one to reflect next hours predictions
    X = np.nan_to_num(X[:-1])
    Y = Y[1:]

    return Y, X

print "Training Models First"
for (htag,lcount) in hashtags.iteritems():
    print "###"
    print "#", htag + ":"
    print "###"

    df = load_dataframe(join('tweet_data', 'tweets_#' + htag + '.txt'), lcount)
    firstLine = datetime(2015,2,1,8,0,0)
    secondLine = datetime(2015,2,1,20,0,0)

    # Data Frame for first Interval
    models[(htag, 1)] = stats_api.OLS(*fetch_matrix(df[df.dateTime < firstLine])).fit()

    # Data Frame for second Interval
    models[(htag, 2)] = stats_api.OLS(*fetch_matrix(df[(df.dateTime > firstLine) & (df.dateTime < secondLine)])).fit()

    # Data Frame for third Interval
    models[(htag, 3)] = stats_api.OLS(*fetch_matrix(df[df.dateTime > secondLine])).fit()


print "Predicting test data"
period_to_htag = {
    "1" : "#superbowl",
    "2" : "#superbowl",
    "3" : "#superbowl",
    "4" : "#nfl",
    "5" : "#nfl",
    "6" : "#superbowl",
    "7" : "#superbowl",
    "8" : "#nfl",
    "9" : "#superbowl",
    "10": "#nfl"
}

test_data = {
    "sample10_period3.txt" : 365,
    "sample1_period1.txt" : 730,
    "sample2_period2.txt" : 212273,
    "sample3_period3.txt" : 3638,
    "sample4_period1.txt" : 1646,
    "sample5_period1.txt" : 2059,
    "sample6_period2.txt" : 205554,
    "sample7_period3.txt" : 528,
    "sample8_period1.txt" : 229,
    "sample9_period2.txt" : 11311
}

for (sample,lcount) in test_data.iteritems():
    rg = re.search('.*?(\\d+).*?(\\d+)', sample)
    s_id = rg.group(1)
    period = rg.group(2)

    print "Predicting number of tweets for file", sample
    Y_test, X_test = fetch_matrix(load_dataframe(join('test_data/', sample), lcount))
    model = models[(period_to_htag[s_id][1:], int(period))]

    Y_pred = model.predict(X_test)

    print "Predictions for the 5 hour window are:", Y_pred
    print "Prediction errors are:", mean_absolute_error(Y_test, Y_pred)
