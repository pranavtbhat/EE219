import json
import numpy as np
from os.path import join
from tqdm import tqdm
import pandas as pd
from datetime import datetime

import matplotlib.pyplot as plt

hashtags = {
    'gohawks' : 188136,
    'nfl' : 259024,
    'sb49' : 826951,
    'gopatriots' : 26232,
    'patriots' : 489713,
    'superbowl' : 1348767
}

{'gopatriots' : 26232}.iteritems()
print "Computing Statistics for hashtags:"
for (htag,lcount) in hashtags.iteritems():
    print "###"
    print "#", htag + ":"
    print "###"

    with open(join('tweet_data', 'tweets_#' + htag + '.txt'), 'r') as f:
        df = pd.DataFrame(index=range(lcount), columns=['dateTime', 'tweetCount', 'retweetCount'])
        users = {}

        for i, line in tqdm(enumerate(f), total=lcount):
            tweet_data = json.loads(line)
            date = datetime.fromtimestamp(tweet_data['firstpost_date'])
            df.set_value(i, 'dateTime', date)
            df.set_value(i, 'tweetCount', 1)
            df.set_value(i, 'retweetCount', tweet_data['metrics']['citations']['total'])

            uid = tweet_data.get('tweet').get('user').get('id')
            users[uid] = tweet_data['author']['followers']

        df = df.set_index('dateTime')
        hourlySeries = df.groupby(pd.TimeGrouper(freq='60Min'))

        tweet_counts = []
        follower_counts = []
        retweet_counts = []

        for i,(interval,group) in enumerate(hourlySeries):
            tweet_count = group.tweetCount.sum()
            tweet_counts.append(tweet_count)

            if tweet_count > 0:
                retweet_counts.append(group.retweetCount.sum() / tweet_count)
            else:
                retweet_counts.append(0)

        print "###"
        print "#", htag
        print "####"
        print "Average number of tweets/hr is", np.mean(tweet_counts)
        print "Average number of followers per user is", sum(users.values()) / float(len(users.keys()))
        print "Average number of retweets per tweet is", np.mean(retweet_counts)

        if htag in ['superbowl', "nfl"]:
            plt.ylabel('Number of tweets')
            plt.xlabel('Hours')
            plt.title('Number of tweets per hour for {}'.format(htag))
            plt.bar(range(len(tweet_counts)),tweet_counts)
            plt.show()
            # plt.savefig('plots/' + htag + '_statistics.png', format='png')

        print "--------------------------------------------------------------------------------"
