import json
from os.path import join
from tqdm import tqdm
import pandas as pd
from datetime import datetime

hashtags = {
    'gohawks' : 188136,
    'nfl' : 259024,
    'sb49' : 826951,
    'gopatriots' : 26232,
    'patriots' : 489713,
    'superbowl' : 1348767
}

int_period_start = datetime(2015,2,1,14,0,0)
int_period_end = datetime(2015,2,1,23,0,0)

def make_files():
    print "Loading text for hashtags:"
        for (htag,lcount) in hashtags.iteritems():
        print "###"
        print "#", htag + ":"
        print "###"

        with open(join('tweet_data', 'tweets_#' + htag + '.txt'), 'r') as f:
            df = pd.DataFrame(columns=['dateTime', 'tweetCount', 'text'])

            for i, line in tqdm(enumerate(f), total=lcount):
                tweet_data = json.loads(line)
                date = datetime.fromtimestamp(tweet_data['firstpost_date'])
                language = tweet_date['tweet']['lang']

                if language == 'en' and date > int_period_start and date < int_period_end:
                    text = tweet_data['tweet']['text']
                    text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())
                    df = pd.append(df, {"dateTime" : date, 'tweetCount' : 1, 'text' : text})

            df.to_csv(join('frames/', htag + '.txt'), sep='\t')
