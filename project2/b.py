from sklearn.feature_extraction import text
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from pandas import DataFrame
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

# Uncomment if the machine is missing punkt, wordnet or stopwords modules.
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')



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

print("%d documents" % len(newsgroups_train.filenames))
print("%d categories" % len(newsgroups_train.target_names))

# Ignore words appearing in less than 2 documents or more than 99% documents.
# min_df reduces from 100k to 29k
vectorizer = CountVectorizer(analyzer='word',stop_words=stop_words,ngram_range=(1, 1), tokenizer=StemTokenizer(),
                             lowercase=True,max_df=0.99, min_df=2)

# test_corpus = [
#     'This is the first document.',
#      'This is the second second document.',
#     'And the third one.',
#      'Is this the first document?',
#  ]

vectorized_newsgroups_train = vectorizer.fit_transform(newsgroups_train.data)
#print "All terms:", vectorizer.get_feature_names()
tfidf_transformer = TfidfTransformer(norm='l2')
train_idf = tfidf_transformer.fit_transform(vectorized_newsgroups_train)
print "Number of terms in TF-IDF representation:",train_idf.shape[1]

# pipeline = Pipeline([
#     ('vect', CountVectorizer()),
#     ('tfidf', TfidfTransformer()),
# ])

# parameters = {
#     'vect__max_df': (0.5, 0.75, 1.0),
#     #'vect__max_features': (None, 5000, 10000, 50000),
#     'vect__ngram_range': ((1, 1)),  # unigrams only
#     'analyzer': 'word',
#     'stop_words': stop_words,
#     #'tfidf__use_idf': (True, False),
#     #'tfidf__norm': ('l1', 'l2'),
# }

# multiprocessing requires the fork to happen in a __main__ protected
# block

# find the best parameters for feature extraction -> use later to identify if bigrams help majorly
#grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

# print("Performing grid search...")
# print("pipeline:", [name for name, _ in pipeline.steps])
# print("parameters:")
# pprint(parameters)
# t0 = time()
# grid_search.fit(data.data, data.target)
# print("done in %0.3fs" % (time() - t0))
# print()

# print("Best score: %0.3f" % grid_search.best_score_)
# print("Best parameters set:")
# best_parameters = grid_search.best_estimator_.get_params()
# for param_name in sorted(parameters.keys()):
#     print("\t%s: %r" % (param_name, best_parameters[param_name]))
