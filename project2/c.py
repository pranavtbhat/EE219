from StdSuites.AppleScript_Suite import vector

from sklearn.feature_extraction import text
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from pandas import DataFrame
import nltk
import numpy as np
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer


# RegExpTokenizer reduces term count from 29k to 25k
class StemTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.snowball_stemmer = SnowballStemmer("english", ignore_stopwords=True)
        self.regex_tokenizer = RegexpTokenizer(r'\w+')

    def __call__(self, doc):
        # tmp = [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
        tmp = [self.snowball_stemmer.stem(t) for t in self.regex_tokenizer.tokenize(doc)]
        return tmp

all_categories=['comp.graphics','comp.os.ms-windows.misc']#,'comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey']
all_docs_per_category=[]



for cat in all_categories:
    categories=[cat]
    all_data = fetch_20newsgroups(subset='train',categories=categories).data
    temp = ""
    for doc in all_data:
        temp= temp + " "+doc
    all_docs_per_category.append(temp)


stop_words = text.ENGLISH_STOP_WORDS


# Ignore words appearing in less than 2 documents or more than 99% documents.
# min_df reduces from 100k to 29k
vectorizer = CountVectorizer(analyzer='word',stop_words=stop_words,ngram_range=(1, 1), tokenizer=StemTokenizer(),
                             lowercase=True)#,max_df=0.99, min_df=2)
#
# test_corpus = [
#     'This document is the first document.',
#      'This is the second second document.',
#     'And the third one with extra extra extra text.',
#      'Is this the first document?',
#  ]

vectorized_newsgroups_train = vectorizer.fit_transform(all_docs_per_category)
#print "All terms:", vectorizer.get_feature_names()
print vectorized_newsgroups_train.shape
#print vectorized_newsgroups_train

print type(vectorized_newsgroups_train)

max_term_freq_per_category=[0]*vectorized_newsgroups_train.shape[0]
category_count_per_term=[0]* vectorized_newsgroups_train.shape[1]

for i in range(0,vectorized_newsgroups_train.shape[0],1):
    max_term_freq_per_category[i]=max(vectorized_newsgroups_train[i].data)

category_count_per_term = vectorized_newsgroups_train.sum(axis=0)

print max_term_freq_per_category
print category_count_per_term

# Calculate tc-icf - Notice the matrix is sparse!

