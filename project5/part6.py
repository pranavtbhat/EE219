from os.path import join
from tqdm import tqdm
import json

from nltk.stem.snowball import SnowballStemmer
import numpy as np

from sklearn import svm
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve
import sklearn.metrics as smet
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

import re
import string
import random

import matplotlib.pyplot as plt

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

class StemTokenizer(object):
    def __init__(self):
        self.snowball_stemmer = SnowballStemmer("english")

    def __call__(self, doc):
        doc = re.sub('[,.-:/()?{}*$#&]', ' ', doc)
        doc = ''.join(ch for ch in doc if ch not in string.punctuation)
        doc = ''.join(ch for ch in doc if ord(ch) < 128)
        doc = doc.lower()
        words = doc.split()
        words = [word for word in words if word not in text.ENGLISH_STOP_WORDS]

        return [
            self.snowball_stemmer.stem(word) for word in words
        ]

def get_vectorizer():
    return CountVectorizer(
        tokenizer=StemTokenizer(),
        lowercase=True,
        min_df = 2,
        max_df = 0.99
    )

def get_tfid_transformer():
    return TfidfTransformer(
        norm='l2',
        sublinear_tf=True
    )

def get_svd():
    return TruncatedSVD(n_components=100)

def print_statistics(actual, predicted):
    print "Accuracy is ", smet.accuracy_score(actual, predicted) * 100
    print "Precision is ", smet.precision_score(actual, predicted, average='macro') * 100
    print "Recall is ", smet.recall_score(actual, predicted, average='macro') * 100
    print "Confusion Matrix is ", smet.confusion_matrix(actual, predicted)

def plot_roc(actual, predicted, classifier_name):
    x, y, _ = roc_curve(actual, predicted)
    plt.plot(x, y, label="ROC Curve")
    plt.plot([0, 1], [0, 1])

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.2])

    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curves for ' + classifier_name + 'Classifier')
    plt.legend(loc="best")

    plt.savefig('plots/' + classifier_name + '.png', format='png')
    plt.show()


def classify(X, Y, classifier, cname):
    b = 0.85 * X.shape[0]
    X_train = X[:b, :]
    Y_train = Y[:b]

    X_test = X[b:, :]
    Y_test = Y[b:]

    classifier.fit(X_train, Y_train)
    predicted = classifier.predict(X_test)
    predicted_probs = classifier.predict_proba(X_test)

    print_statistics(Y_test, predicted)
    plot_roc(Y_test, predicted_probs[:, 1], cname)

print "Loading superbowl tweets"
lcount = 1348767

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

    pipeline = Pipeline(
        [
            ('vectorize', get_vectorizer()),
            ('tf-idf', get_tfid_transformer()),
            ('svd', get_svd())
        ]
    )

    print "Computing the LSI representation of the dataset"
    X = pipeline.fit_transform(X)
    Y = np.array(Y)

    # Randomly shuffle data
    indexes = range(X.shape[0])
    random.shuffle(indexes)
    indexes = indexes[:5000]
    X_ = X[indexes, :]
    Y_ = Y[indexes]

    print "Statistics of SVM classifier:"
    classify(X_, Y_, svm.SVC(kernel='linear', probability=True), "SVM")

    print "Statistics of AdaBoost Classifier are"
    classify(X_, Y_, AdaBoostClassifier(), "AdaBoost")

    print "Statistics of Random Forest Classifier are"
    classify(X_, Y_, RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1), "RandomForestClassifier")

    print "Statistics of Neural Network Classifier are"
    classify(X_, Y_, MLPClassifier(alpha=1), "Neural Network Classifier")
