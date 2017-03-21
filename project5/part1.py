import json
import numpy as np
from os.path import join
from tqdm import tqdm

import matplotlib.pyplot as plt

hashtags = {
    'gohawks' : 188136,
    'nfl' : 259024,
    'sb49' : 826951,
    'gopatriots' : 26232,
    'patriots' : 489713,
    'superbowl' : 1348767
}

print "Computing Statistics for hashtags:"

for (htag,lcount) in hashtags.iteritems():
    print "#", htag + ":"

    with open(join('tweet_data', 'tweets_#' + htag + '.txt'), 'r') as f:
        cw = 1
        users = {}

        # Fetch first tweet to figure our the start date (since tweets are sorted by time)
        first_tweet = json.loads(f.readline())
        f.seek(0, 0)

        num_tweets_in_hour = []
        number_of_retweets = 0

        start_time = first_tweet.get('firstpost_date')
        end_of_window = start_time + cw * 3600
        hourly_count = 0
        tweet_count = 0

        for line in tqdm(f, total=lcount):
            tweet_data = json.loads(line)
            tweet_time = tweet_data.get('firstpost_date')
            tweet_count += 1

            if tweet_time < end_of_window:
                hourly_count += 1
            else:
                num_tweets_in_hour.append(hourly_count)
                cw += 1
                hourly_count = 0
                end_of_window = start_time + cw * 3600

            user = tweet_data.get('tweet').get('user').get('id')
            users[user] = tweet_data.get('author').get('followers')
            number_of_retweets += tweet_data['metrics']['citations']['total']

        print "Total number of tweets", tweet_count
        print "Average number of tweets per hour", lcount / ((tweet_time - start_time) / 3600.0)
        print "Average number of followers of authors", np.mean(users.values())
        print "Average number of retweets", number_of_retweets / (lcount + 0.0)

        if htag in ['superbowl', "nfl"]:
            plt.ylabel('number of tweets')
            plt.xlabel('hours')
            plt.title('Number of tweets per hour for {}'.format(htag))
            plt.bar(range(len(num_tweets_in_hour)), num_tweets_in_hour)
            plt.savefig('plots/' + htag + '_statistics.png', format='png')
