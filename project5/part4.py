import json
import numpy as np
from os.path import join
from tqdm import tqdm
import statsmodels.api as stats_api
from datetime import datetime
import collections
from sklearn.cross_validation import KFold

hashtags = {
    'gohawks' : 188136,
    'nfl' : 259024,
    'sb49' : 826951,
    'gopatriots' : 26232,
    'patriots' : 489713,
    'superbowl' : 1348767
}

feature_names = ['Number of Tweets', 'Number of Retweets', 'Number of Followers', 'Max Number of Followers',
                  'Impression Count', 'Favourite Count', 'Ranking Score', 'Hour of Day', 'Number of Users tweeting',
                  'Number of Long Tweets']
                  

def get_feature_dict():
    features = dict.fromkeys(feature_names, 0)
    features['Number of Users tweeting'] = []
    return features
            
def get_feature_stats(features, tweet_data):
    features['Number of Tweets'] += 1
    features['Number of Retweets'] += tweet_data.get('metrics').get('citations').get('total')
    follower_count = tweet_data.get('author').get('followers')
    features['Number of Followers'] += follower_count
    features['Impression Count'] += tweet_data.get('metrics').get('impressions')
    if follower_count > features['Max Number of Followers']:
        features['Max Number of Followers'] = follower_count
    features['Ranking Score'] += tweet_data.get('metrics').get('ranking_score')
    features['Hour of Day'] = int(datetime.fromtimestamp(tweet_data.get('firstpost_date')).strftime("%H"))
    features['Favourite Count'] += tweet_data.get('tweet').get('favorite_count')
    features['Number of Users tweeting'].append(tweet_data.get('tweet').get('user').get('id'))
    features['Number of Long Tweets'] += 1 if len(tweet_data.get('title')) > 100 else 0
    return features

def get_features(features):
    extract = []
    for i in feature_names:
        extract.append(features[i])
    extract[8] = len(set(extract[8]))
    return extract
        
def reset_features_dict():
    features = dict.fromkeys(feature_names, 0)
    features['Number of Users tweeting'] = []
    return features
    
if __name__ == "__main__":
    for (htag,lcount) in hashtags.iteritems():
        print "#", htag + ":"
    
        with open(join('tweet_data', 'tweets_#' + htag + '.txt'), 'r') as f:
            first_tweet = json.loads(f.readline())
            f.seek(0, 0)
            start_time = first_tweet.get('firstpost_date')
    
            # Initialize X
            Y = []
            X = []
            features = get_feature_dict()
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
                    features = get_feature_stats(features, tweet_data)
                else:
                    extracted_features = get_features(features)
                    Y.append(extracted_features[0])
                    X.append(extracted_features[1:])
        
                    features = reset_features_dict()  # reset features for new window calculation
                    features = get_feature_stats(features, tweet_data)  # update stats of tweet
        
                    cw += 1
                    end_of_window = start_time + cw * 3600  # update window
    
            Y = np.roll(np.array(Y), -1)
            Y = collections.deque(Y)
            Y.rotate(-1)
            Y = np.delete(Y,-1)
            del(X[-1])
            
            X = np.array(X)
            Y = np.array(Y)
            average_error = []
            kf = KFold(len(X), n_folds=10, shuffle=False, random_state=None)

            for train_index, test_index in kf:
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = Y[train_index], Y[test_index]
        
                y_train = collections.deque(y_train)
                y_train.rotate(-1)
                y_train = list(y_train)
                X_train = list(X_train)
        
                result = stats_api.OLS(y_train, X_train).fit()
                test_prediction = result.predict(X_test)
                average_error.append(np.mean(abs(test_prediction - y_test)))
                
            print "10 fold error", average_error
            print "Average error", np.mean(average_error)
