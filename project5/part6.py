import numpy as np
from os.path import join
from tqdm import tqdm
import statsmodels.api as stats_api
from datetime import datetime
import pandas as pd
import json

hashtags = {
    'gohawks' : 188136,
    'nfl' : 259024,
    'sb49' : 826951,
    'gopatriots' : 26232,
    'patriots' : 489713,
    'superbowl' : 1348767
}

print "Loading superbowl tweets"
lcount = 1348767

def in_washington(location):
    white_list = [
        "seattle",
        "washington",
        "wa",
        "kirkland"
    ]

    black_list = [
        "dc",
        "d.c.",
        "d.c."
    ]

    flag = False
    location = location.split()

    for s in white_list:
        if s in location:
            flag = True
            break

    for s in black_list:
        if s in location:
            flag = False
            break

    return flag

def in_mas(location):
    white_list = [
        "ma",
        "massachusetts",
        "boston",
        "worcester",
        "salem",
        "plymouth",
        "springfield",
        "arlington",
        "scituate",
        "northampton"
    ]

    location = location.split()

    black_list = [
        "ohio",
    ]
    flag = False

    for s in white_list:
        if s in location:
            flag = True
            break

    for s in black_list:
        if s in location:
            flag = False
            break

    return flag

with open(join('tweet_data', 'tweets_#superbowl.txt'), 'r') as f:
    X = []
    Y = []
    for i, line in tqdm(enumerate(f), total=lcount):
        tweet_data = json.loads(line)
        location = tweet_data.get("tweet").get("user").get("location").lower()

        if in_washington(location):
            X.append(tweet_data.get("title"))
            Y.append(0)
        elif in_mas(location):
            X.append(tweet_data.get("title"))
            Y.append(1)




