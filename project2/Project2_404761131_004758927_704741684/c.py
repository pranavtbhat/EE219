from StdSuites.AppleScript_Suite import vector
import cPickle
from sklearn.feature_extraction import text
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from pandas import DataFrame
import nltk
import operator
import os
import numpy as np
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
import math

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

def calculate_tcicf(freq, maxFreq, categories, categories_per_term):
    val= ((0.5+(0.5*(freq/float(maxFreq))))*math.log10(categories/float(1+categories_per_term)))
    return val

all_categories=['comp.graphics',
                'comp.os.ms-windows.misc',
                'comp.sys.ibm.pc.hardware',
                'comp.sys.mac.hardware',
                'comp.windows.x',
                'rec.autos',
                'rec.motorcycles',
                'rec.sport.baseball',
                'rec.sport.hockey',
                'alt.atheism',
                'sci.crypt',
                'sci.electronics',
                'sci.med',
                'sci.space',
                'soc.religion.christian',
                'misc.forsale',
                'talk.politics.guns',
                'talk.politics.mideast',
                'talk.politics.misc',
                'talk.religion.misc'
                ]

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
                             lowercase=True,max_df=0.99, min_df=2)
#
# test_corpus = [
#     'This document is the first document.',
#      'This is the second second document.',
#     'And the third one with extra extra extra text.',
#      'Is this the first document?',
#  ]

vectorized_newsgroups_train = vectorizer.fit_transform(all_docs_per_category)
#print "All terms:", vectorizer.get_feature_names()
#print vectorized_newsgroups_train.shape
#print vectorized_newsgroups_train

def calculate():

    max_term_freq_per_category=[0]*vectorized_newsgroups_train.shape[0]
    category_count_per_term=[0]*vectorized_newsgroups_train.shape[1]

    for i in range(0,vectorized_newsgroups_train.shape[0],1):
        max_term_freq_per_category[i]=np.amax(vectorized_newsgroups_train[i,:])

    for i in range(0,vectorized_newsgroups_train.shape[1],1):
        for j in range(0,vectorized_newsgroups_train.shape[0],1):
            category_count_per_term[i]+= (0 if vectorized_newsgroups_train[j,i]==0 else 1)

    # print vectorized_newsgroups_train.shape
    #
    # print len(max_term_freq_per_category)
    # print len(category_count_per_term)

    # Calculate tc-icf - Notice the matrix is sparse!
    # print len(vectorizer.get_feature_names())

    tf_icf = np.zeros((len(vectorizer.get_feature_names()), vectorized_newsgroups_train.shape[1]))

    for i in range(vectorized_newsgroups_train.shape[1]):
        row = vectorized_newsgroups_train[:,i].toarray()
        for j in range(vectorized_newsgroups_train.shape[0]):
            # print row[j,0],max_term_freq_per_category[j],len(all_categories),category_count_per_term[i]
            tf_icf[i][j] = calculate_tcicf(row[j,0],max_term_freq_per_category[j],len(all_categories),category_count_per_term[i])

    # cPickle.dump(tf_icf,open("data/tc_icf.pkl", "wb"))
    return tf_icf

# if not (os.path.isfile("data/tc_icf.pkl")):
#     print "Calculating"
#     tf_icf=calculate()
# else:
#     tf_icf=cPickle.load(open("data/tc_icf.pkl", "rb"))

tf_icf=calculate()


# print top 10 significant term for this class
for category in [2,3,14,15]:
    tficf={}
    term_index=0;
    for term in vectorizer.get_feature_names():
        tficf[term]=tf_icf[term_index][category]
        term_index+=1
    significant_terms = dict(sorted(tficf.iteritems(), key=operator.itemgetter(1), reverse=True)[:10]) #get 10 significant terms
    print significant_terms.keys()


