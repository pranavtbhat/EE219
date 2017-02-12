from sklearn.feature_extraction import text
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from pandas import DataFrame
import nltk
import cPickle
from sklearn.decomposition import TruncatedSVD
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

# Todo: Refactor to call code in b.py

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

stop_words = text.ENGLISH_STOP_WORDS
categories=['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey']
newsgroups_train = fetch_20newsgroups(subset='train',categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test',categories=categories)

# Ignore words appearing in less than 2 documents or more than 99% documents.
# min_df reduces from 100k to 29k
vectorizer = CountVectorizer(analyzer='word',stop_words=stop_words,ngram_range=(1, 1), tokenizer=StemTokenizer(),
                             lowercase=True,max_df=0.99, min_df=2)

tfidf_transformer = TfidfTransformer(norm='l2')

svd = TruncatedSVD(n_components=50, random_state=42)


pipeline = Pipeline([('vectorize', vectorizer),
                     ('tf-idf', tfidf_transformer),
                     ('svd', svd)])

svd_matrix_train = pipeline.fit_transform(newsgroups_train.data)
svd_matrix_test = pipeline.fit_transform(newsgroups_test.data)

cPickle.dump(svd_matrix_train,open("data/Train_LSI.pkl", "wb"))
cPickle.dump(svd_matrix_test,open("data/Test_LSI.pkl", "wb"))