import json
from os.path import join
from tqdm import tqdm
import pandas as pd
import datetime
import re
import matplotlib.pyplot as plt
import numpy as np

hashtags = {
    'gohawks' : 188136,
    'nfl' : 259024,
    'sb49' : 826951,
    'gopatriots' : 26232,
    'patriots' : 489713,
    'superbowl' : 1348767
}

int_period_start = datetime.datetime(2015,2,1,14,0,0)
int_period_end = datetime.datetime(2015,2,1,20,0,0)

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
                date = datetime.datetime.fromtimestamp(tweet_data['firstpost_date'])
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

def identify_peaks():
    df = pd.read_csv('frames/all.txt', sep = '\t')
    df.dateTime = df.dateTime.apply(lambda x : datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    df = df.set_index('dateTime')

    tenSecondSeries = df.groupby(pd.TimeGrouper(freq='1S'))
    seconds = []
    tweetCounts = []

    for i, group in tenSecondSeries:
        seconds.append(i)
        tweetCounts.append(group.tweetCount.sum())

    return seconds, tweetCounts


def get_ratios():
    secs = []
    ratios = []

    print "Fetching Second wise data"
    seconds, tweetCounts = identify_peaks()

    for i in range(0, len(seconds)):
        secs.append(seconds[i])
        firstFive = sum(tweetCounts[i:i+5])
        secondFive = sum(tweetCounts[i+5:i+10])

        if secondFive == 0:
            ratios.append(1)
        else:
            ratios.append(firstFive / float(secondFive))

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

def create_dump(ts):
    df = pd.read_csv('frames/all.txt', sep = '\t')
    df.dateTime = df.dateTime.apply(lambda x : datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    for t in ts:
        start = t
        end = t + datetime.timedelta(seconds=60)
        tweets = df[df.dateTime.apply(lambda x : x >= start and x <= end)]
        with open('ts/' + str(t) + '.txt', 'w') as f:
            for text in tweets.text:
                f.write(text + '\n')


seconds, ratios = get_ratios()
seconds = np.array(seconds)
ratios = np.array(ratios)

best50 = ratios.argsort()[:50]

bs = seconds[best50]
rs = ratios[best50]

sbs = sorted(bs)
plt.xlim(sbs[0], sbs[-1])
plt.scatter(sbs, rs[bs.argsort()])

create_dump(bs)

