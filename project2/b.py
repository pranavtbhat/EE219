from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
import a

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

def get_vectorizer():
    return CountVectorizer(
        analyzer='word',
        stop_words= text.ENGLISH_STOP_WORDS,
        ngram_range=(1, 1),
        tokenizer=StemTokenizer(),
        lowercase=True,
        max_df=0.99,
        min_df=2
    )

def get_tfid_transformer():
    return TfidfTransformer(
        norm='l2',
        sublinear_tf=True
    )

if __name__ == "__main__":
    categories=[
        'comp.graphics',
        'comp.os.ms-windows.misc',
        'comp.sys.ibm.pc.hardware',
        'comp.sys.mac.hardware',
        'rec.autos',
        'rec.motorcycles',
        'rec.sport.baseball',
        'rec.sport.hockey'
    ]

    pipeline = Pipeline(
        [
            ('vectorize', get_vectorizer()),
            ('tf-idf', get_tfid_transformer())
        ]
    )

    train = a.fetch_train(categories)

    print("%d documents" % len(train.filenames))
    print("%d categories" % len(train.target_names))

    train_idf = pipeline.fit_transform(train.data)
    print "Number of terms in TF-IDF representation:",train_idf.shape[1]
