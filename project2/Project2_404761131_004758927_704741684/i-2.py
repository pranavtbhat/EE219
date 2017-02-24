from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

import a
import b
import d
import e

categories = [
    'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware',
    'misc.forsale',
    'soc.religion.christian'
]

train = a.fetch_train(categories)
test  = a.fetch_test(categories)

pipeline = Pipeline(
    [
        ('vectorize', b.get_vectorizer()),
        ('tf-idf', b.get_tfid_transformer()),
        ('svd', d.get_svd())
    ]
)

train_lsi = pipeline.fit_transform(train.data)
test_lsi = pipeline.transform(test.data)

def perform_classification(clf):
    global train_lsi, test_lsi, train, test

    clf.fit(train_lsi, train.target)
    predicted = clf.predict(test_lsi)

    e.print_statistics(test.target, predicted)

print "One Vs One Classification using Naive Bayes"
perform_classification(OneVsOneClassifier(GaussianNB()))

print "One Vs Rest Classifciation using Naive Bayes"
perform_classification(OneVsRestClassifier(GaussianNB()))

print "One Vs One Classification using SVM"
perform_classification(OneVsOneClassifier(svm.SVC(kernel='linear')))

print "One Vs Rest Classificaiton using SVM"
perform_classification(OneVsRestClassifier(svm.SVC(kernel='linear')))
