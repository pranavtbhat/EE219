import json
import numpy as np
from os.path import join
from tqdm import tqdm
import statsmodels.api as stats_api
from datetime import datetime
import pandas as pd

hashtags = {
    'gohawks' : 188136,
    'nfl' : 259024,
    'sb49' : 826951,
    'gopatriots' : 26232,
    'patriots' : 489713,
    'superbowl' : 1348767
}

print "Extracting features from tweets"
for (htag,lcount) in hashtags.iteritems():
    print "###"
    print "#", htag + ":"
    print "###"

    with open(join('tweet_data', 'tweets_#' + htag + '.txt'), 'r') as f:
        df = pd.DataFrame(index=range(lcount), columns=['dateTime', 'tweetCount', 'retweetCount', 'followerSum', 'maxFollowers'])
        for i, line in tqdm(enumerate(f), total=lcount):
            tweet_data = json.loads(line)
            date = datetime.fromtimestamp(tweet_data['firstpost_date'])
            df.set_value(i, 'dateTime', date)
            df.set_value(i, 'tweetCount', 1)
            df.set_value(i, 'retweetCount', tweet_data['metrics']['citations']['total'])
            df.set_value(i, 'followerSum', tweet_data['author']['followers'])
            df.set_value(i, 'maxFollowers', tweet_data['author']['followers'])

        df = df.set_index('dateTime')
        hourlySeries = df.groupby(pd.TimeGrouper(freq='60Min'))

        X = np.zeros((len(hourlySeries), 5))
        Y = np.zeros((len(hourlySeries)))

        for i,(interval,group) in enumerate(hourlySeries):
            X[i, 0] = group.tweetCount.sum()
            X[i, 1] = group.retweetCount.sum()
            X[i, 2] = group.followerSum.sum()
            X[i, 3] = group.maxFollowers.max()
            X[i, 4] = interval.hour

            Y[i] = group.tweetCount.sum()


        # Shift X and Y forward by one to reflect next hours predictions
        X = np.nan_to_num(X[:-1])
        Y = Y[1:]

        # Train the regression model
        result = stats_api.OLS(Y, X).fit()

        print result.summary()

        print "--------------------------------------------------------------------------------"

