import json
import datetime, time
import numpy as np
from os.path import join
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')

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

    tweet_list = []
    timestamps = []
    num_tweets = 0

    with open(join('tweet_data', 'tweets_#' + htag + '.txt'), 'r') as f:
        for line in tqdm(f, total=lcount):
            tweet = json.loads(line)
            tweet_list.append(tweet)
            timestamps.append(tweet['firstpost_date'])

        print "Loaded ", num_tweets, "tweets"

        start_date = datetime.datetime(2015, 01, 30, 0, 0, 0)
        end_date = datetime.datetime(2015, 02, 03, 0, 0, 0)
        mintime = int(time.mktime(start_date.timetuple()))
        maxtime = int(time.mktime(end_date.timetuple()))
        date_reference = datetime.datetime(2015, 01, 01, 0, 0, 0)
        time_reference = int(time.mktime(date_reference.timetuple()))

        print "extracting features"

        num_hours = int((maxtime-mintime)/3600)
        num_tweets_in_hour = [0] * num_hours
        num_retweets_in_hour = [0] * num_hours
        sum_followers = [0] * num_hours
        max_followers = [0] * num_hours

        time_of_day = [0] * num_hours
        mintime_daytime = int((mintime-time_reference)/3600) % 24

        for t in range(0,num_hours):
            time_of_day[t] = mintime_daytime + t

        for i in range(0,num_tweets):
            tweet = tweet_list[i]
            tweet_time = tweet['firstpost_date']
            if tweet_time >= mintime:
                hour = int((tweet_time-mintime)/3600)
                if hour >= num_hours:
                    break;
                num_tweets_in_hour[hour] += 1
                try:
                    num_retweets_in_hour += tweet['metrics']['citations']['data'][0]['citations']
                except:
                    pass
                sum_followers[hour] += tweet['tweet']['user']['followers_count']
                max_followers[hour] = max(max_followers[hour], tweet['tweet']['user']['followers_count'])


        print "Average number of tweets per hour", np.mean(num_tweets_in_hour)
        print "Average number of followers of authors", np.mean(sum_followers)
        print "Average number of retweets", np.mean(num_retweets_in_hour)

        if htag in ['superbowl', "nfl"]:
            plt.ylabel('Number of Tweets')
            plt.xlabel('Hour')
            plt.title('Number of Tweets per hour for {}'.format(htag))
            plt.bar(range(len(num_tweets_in_hour)), num_retweets_in_hour)
            plt.savefig(htag + '_statistics.png', format='png')
