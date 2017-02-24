from sklearn.feature_extraction import text
from sklearn.pipeline import Pipeline
import cPickle
from sklearn.decomposition import TruncatedSVD
import os

import a
import b

stop_words = text.ENGLISH_STOP_WORDS


def get_svd():
    return TruncatedSVD(n_components=50)

def fetch_lsi_representation(train, test):
    pipeline = Pipeline(
        [
            ('vectorize', b.get_vectorizer()),
            ('tf-idf', b.get_tfid_transformer()),
            ('svd', get_svd())
        ]
    )

    svd_matrix_train = pipeline.fit_transform(train.data)
    svd_matrix_test = pipeline.transform(test.data)

    return svd_matrix_train, svd_matrix_test


def fetch_lsi_representation_catched(train, test):
    if not (os.path.isfile("Data/Train_LSI.pkl") and os.path.isfile("Data/Test_LSI.pkl")):

        print "Performing LSI on the TFxIDF matrices for Train and Test"

        svd_matrix_train, svd_matrix_test = fetch_lsi_representation(
            train,
            test
        )
        cPickle.dump(svd_matrix_train, open("data/Train_LSI.pkl", "wb"))
        cPickle.dump(svd_matrix_test, open("data/Test_LSI.pkl", "wb"))

        return svd_matrix_train, svd_matrix_test
    else:
        svd_matrix_train = cPickle.load(open("Data/Train_LSI.pkl", "r"))
        svd_matrix_test = cPickle.load(open("Data/Test_LSI.pkl", "r"))

        return svd_matrix_train, svd_matrix_test


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

    train = a.fetch_train(categories)
    test = a.fetch_test(categories)

    svd_matrix_train, svd_matrix_test = fetch_lsi_representation(
        train,
        test
    )

    print "Size of Training LSI representation is ", svd_matrix_train.shape
    print "Size of Testing LSI representation is ", svd_matrix_test.shape

    cPickle.dump(svd_matrix_train, open("data/Train_LSI.pkl", "wb"))
    cPickle.dump(svd_matrix_test, open("data/Test_LSI.pkl", "wb"))

