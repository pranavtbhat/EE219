import json
import numpy as np
from os.path import join
from tqdm import tqdm
import statsmodels.api as stats_api
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

hashtags = {
    'gohawks' : 188136,
    'nfl' : 259024,
    'sb49' : 826951,
    'gopatriots' : 26232,
    'patriots' : 489713,
    'superbowl' : 1348767
}

feature_names = ['Number of Tweets', 'Number of Retweets', 'Number of Followers', 'Max Number of Followers', 'Hour of Day',
                  'Impression Count', 'Favourite Count', 'Ranking Score', 'Number of Users tweeting',
                  'Number of Long Tweets']

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

        df = df.set_index('dateTime')
        hourlySeries = df.groupby(pd.TimeGrouper(freq='60Min'))

        X = np.zeros((len(hourlySeries), 10))
        Y = np.zeros((len(hourlySeries)))

        # Extract features for each hourly interval
        for i,(interval,group) in enumerate(hourlySeries):
            tweet_count = group.tweetCount.sum()
            X[i, 0] = tweet_count                   # Number of tweets
            X[i, 1] = group.retweetCount.sum()      # Number of retweets
            X[i, 2] = group.followerSum.sum()       # Sum of follower counts
            X[i, 3] = group.maxFollowers.max()      # Maximum size following
            X[i, 4] = interval.hour                 # Hour of the day
            X[i, 5] = group.impressionCount.sum() / tweet_count   # Sum of impression count
            X[i, 6] = group.favoriteCount.sum() / tweet_count    # Sum of favorites
            X[i, 7] = group.rankingScore.sum() / tweet_count     # Sum of rankings
            X[i, 8] = group.userID.nunique()        # Number of unique users tweeting
            X[i, 9] = group.longTweet.sum()         # Number of long tweets

            Y[i] = group.tweetCount.sum()


        # Remove nans
        X = np.nan_to_num(X[:-1])
        Y = Y[1:]

        # Train the regression model
        stats_api.add_constant(X)
        result = stats_api.OLS(Y, X).fit()
        print result.summary()

        best_features = result.pvalues.argsort()[:3]
        print "Best features selected are:"

        for index in best_features:
            print feature_names[index]
            plt.title("Scatter plot for {0} against {1}".format(htag, feature_names[index]))
            plt.xlabel("Number of tweets/hour")
            plt.ylabel(feature_names[index])
            plt.scatter(Y, X[:, index])
            plt.savefig("plots/#{0}vs{1}.png".format(htag, feature_names[index]))
            plt.clf()

        print "--------------------------------------------------------------------------------"
