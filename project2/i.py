from sklearn.datasets import fetch_20newsgroups
from sklearn import svm
import sklearn.metrics as smet
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

###
# Load datsets
###
categories = [
    'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware',
    'misc.forsale',
    'soc.religion.christian'
]

class StemTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.snowball_stemmer = SnowballStemmer("english", ignore_stopwords=True)
        self.regex_tokenizer = RegexpTokenizer(r'\w+')

    def __call__(self, doc):
        # tmp = [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
        tmp = [self.snowball_stemmer.stem(t) for t in self.regex_tokenizer.tokenize(doc)]
        return tmp

stop_words = text.ENGLISH_STOP_WORDS

train = fetch_20newsgroups(subset='train',categories=categories)
test = fetch_20newsgroups(subset='test',categories=categories)

# Ignore words appearing in less than 2 documents or more than 99% documents.
# min_df reduces from 100k to 29k
vectorizer = CountVectorizer(
    analyzer='word',
    stop_words=stop_words,
    ngram_range=(1, 1),
    tokenizer=StemTokenizer(),
    lowercase=True,
    max_df=0.99,
    min_df=2
)

tfidf_transformer = TfidfTransformer(
    norm='l2',
    sublinear_tf=True
)

svd = TruncatedSVD(n_components=50)

pipeline = Pipeline([('vectorize', vectorizer),
                     ('tf-idf', tfidf_transformer),
                     ('svd', svd)])

train_lsi = pipeline.fit_transform(train.data)
test_lsi = pipeline.fit_transform(test.data)

def perform_classification(clf):
    global train_lsi, test_lsi, train, test

    clf.fit(train_lsi, train.target)
    predicted = clf.predict(test_lsi)
    print "Accuracy is ", smet.accuracy_score(test.target, predicted) * 100
    print "Precision is ", smet.precision_score(test.target, predicted, average='macro') * 100

    print "Recall is ", smet.recall_score(test.target, predicted, average='macro') * 100

    print "Confusion Matrix is ", smet.confusion_matrix(test.target, predicted)


print "One Vs One Classification using Naive Bayes"
perform_classification(OneVsOneClassifier(GaussianNB()))

print "One Vs Rest Classifciation using Naive Bayes"
perform_classification(OneVsRestClassifier(GaussianNB()))

print "One Vs One Classification using SVM"
perform_classification(OneVsOneClassifier(svm.SVC(kernel='linear')))

print "One Vs Rest Classificaiton using SVM"
perform_classification(OneVsRestClassifier(svm.SVC(kernel='linear')))
