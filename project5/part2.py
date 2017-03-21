import json
import numpy as np
from os.path import join
from tqdm import tqdm
import statsmodels.api as stats_api
from datetime import datetime

hashtags = {
    'gohawks' : 188136,
    'nfl' : 259024,
    'sb49' : 826951,
    'gopatriots' : 26232,
    'patriots' : 489713,
    'superbowl' : 1348767
}

print "Extracting X from tweets"
for (htag,lcount) in hashtags.iteritems():
    print "#", htag + ":"

    with open(join('tweet_data', 'tweets_#' + htag + '.txt'), 'r') as f:
        first_tweet = json.loads(f.readline())
        f.seek(0, 0)
        start_time = first_tweet.get('firstpost_date')

        # Initialize X
        Y = []
        X = []
        tweets_per_hour = 0
        retweets_per_hour = 0
        number_of_followers_hour = 0
        max_number_of_followers = 0

        # Hourly window traversal
        cw = 1
        end_of_window = start_time + cw * 3600

        for line in tqdm(f, total=lcount):
            tweet_data = json.loads(line)
            tweet_time = tweet_data.get('firstpost_date')

            if tweet_time < end_of_window:
                tweets_per_hour += 1
                tweets_per_hour += tweet_data['metrics']['citations']['total']
                number_of_followers_hour += tweet_data['author']['followers']
                max_number_of_followers = max(max_number_of_followers, tweet_data['author']['followers'])

            else:
                X.append([
                    retweets_per_hour,
                    number_of_followers_hour,
                    max_number_of_followers,
                    int(datetime.fromtimestamp(
                        tweet_data.get('firstpost_date')).strftime("%H")
                    )
                ])
                Y.append(tweets_per_hour)

                # Reinitialize X for new hour
                cw += 1
                end_of_window = start_time + cw * 3600
                tweets_per_hour = 1
                retweets_per_hour = tweet_data['metrics']['citations']['total']
                number_of_followers_hour = tweet_data['author']['followers']
                max_number_of_followers = tweet_data['author']['followers']


        # Shift X and Y forward by one to reflect next hours predictions
        Y = list(reversed(np.array(Y[1:])))
        X = X[1:]

        # Train the regression model
        result = stats_api.OLS(Y, X).fit()

        print "The fitted parameters are", result.params
        print "errors", result.bse

        print "p values", result.pvalues
        print "t values", result.tvalues

        print "Accuracy", result.rsquared * 100
