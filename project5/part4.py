import json
import numpy as np
from os.path import join
from tqdm import tqdm
import statsmodels.api as stats_api
from datetime import datetime
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

hashtags = {
    'gohawks' : 188136,
    'nfl' : 259024,
    'sb49' : 826951,
    'gopatriots' : 26232,
    'patriots' : 489713,
    'superbowl' : 1348767
}

feature_names = ['Number of Tweets', 'Number of Retweets', 'Number of Followers', 'Max Number of Followers',
                  'Impression Count', 'Favourite Count', 'Ranking Score', 'Hour of Day', 'Number of Users tweeting',
                  'Number of Long Tweets']

def cross_validate(X, Y):
    errors = []
    for train_index, test_index in KFold(n_splits=10).split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        lm = stats_api.OLS(Y_train, X_train).fit()
        Y_pred = lm.predict(X_test)

        errors.append(mean_absolute_error(Y_test, Y_pred))

    return errors

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

    return X, Y


print "Extracting features from tweets"
for (htag,lcount) in hashtags.iteritems():
    print "###"
    print "#", htag + ":"
    print "###"

    with open(join('tweet_data', 'tweets_#' + htag + '.txt'), 'r') as f:
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


        firstLine = datetime(2015,2,1,8,0,0)
        secondLine = datetime(2015,2,1,20,0,0)

        # Data Frame for first Interval
        df1 = df[df.dateTime < firstLine]
        print "Running cross validation for values before the Feb. 1, 8:00 a.m."
        print "Mean absolute error was", np.mean(cross_validate(*fetch_matrix(df1)))

        # Data Frame for second Interval
        df2 = df[(df.dateTime > firstLine) & (df.dateTime < secondLine)]
        print "Running cross validation for values between Feb. 1, 8:00 a.m. and 8:00 p.m."
        print "Mean absolute error was", np.mean(cross_validate(*fetch_matrix(df2)))

        # Data Frame for third Interval
        df3 = df[df.dateTime > secondLine]
        print "Running cross validation for values after Feb. 1, 8:00 p.m."
        print "Mean absolute error was", np.mean(cross_validate(*fetch_matrix(df3)))
