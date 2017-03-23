import json
from os.path import join
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import re
import matplotlib.pyplot as plt

hashtags = {
    'gohawks' : 188136,
    'nfl' : 259024,
    'sb49' : 826951,
    'gopatriots' : 26232,
    'patriots' : 489713,
    'superbowl' : 1348767
}

int_period_start = datetime(2015,2,1,14,0,0)
int_period_end = datetime(2015,2,1,20,0,0)

def make_files():
    df = pd.DataFrame(index=range(sum(hashtags.values())), columns=['dateTime', 'language', 'tweetCount', 'text'])

    print "Loading text for hashtags:"
    for (htag,lcount) in hashtags.iteritems():
        print "###"
        print "#", htag + ":"
        print "###"
        with open(join('tweet_data', 'tweets_#' + htag + '.txt'), 'r') as f:
            for i, line in tqdm(enumerate(f), total=lcount):
                tweet_data = json.loads(line)
                date = datetime.fromtimestamp(tweet_data['firstpost_date'])
                language = tweet_data['tweet']['lang']

                text = tweet_data['tweet']['text']
                text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())
                df.set_value(i, 'dateTime', date)
                df.set_value(i, 'language', language)
                df.set_value(i, 'tweetCount', 1)
                df.set_value(i, 'text', text)

    df = df[df.language == "en"]
    df = df[df.dateTime.apply(lambda x : x > int_period_start)]
    df = df[df.dateTime.apply(lambda x : x < int_period_end)]
    df.to_csv(join('frames/', "all" + '.txt'), sep='\t')

def identify_peaks(htag):
    df = pd.read_csv('frames/' + htag + '.txt', sep = '\t')
    df.dateTime = df.dateTime.apply(lambda x : datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    df = df.set_index('dateTime')

    tenSecondSeries = df.groupby(pd.TimeGrouper(freq='1S'))
    seconds = []
    tweetCounts = []

    for i, group in tenSecondSeries:
        seconds.append(i)
        tweetCounts.append(group.tweetCount.sum())

    return seconds, tweetCounts


def get_ratios():
    secondsAll = None
    tweetCountsAll = [0 for i in range(32200)]

    for htag in hashtags.keys():
        seconds, tweetCounts = identify_peaks(htag)

        if secondsAll == None:
            secondsAll = seconds

        tweetCountsAll += tweetCounts[:32201]

    secs = []
    ratios = []
    for i in range(0, len(secondsAll)):
        secs.append(seconds[i])
        firstFive = sum(tweetCountsAll[i:i+5])
        secondFive = sum(tweetCountsAll[i+5:i+10])

        if secondFive == 0:
            ratios.append(1)
        else:
            ratios.append(firstFive / secondFive)

    return secs, ratios


def all_graphs():
    # 32203
    secondsAll = None
    tweetCountsAll = None
    for htag in hashtags.keys():
        print htag
        seconds, tweetCounts = identify_peaks(htag)

        if secondsAll == None:
            secondsAll = seconds[:32201]

        if tweetCountsAll == None:
            tweetCountsAll = tweetCounts[:32201]
        else:
            tweetCountsAll += tweetCounts[:32201]

    plt.plot(range(len(tweetCountsAll)), tweetCountsAll)
    plt.show()

