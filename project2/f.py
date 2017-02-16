from sklearn import svm
from sklearn import cross_validation
import numpy as np

import a
import d
import e

def fetch_best_parameter(train, train_lsi):
    print "Learning best parameter value for k!"
    params = list(range(-3, 4))
    scores = []

    for param in params:
        print "Parameter ", param
        classifier = svm.SVC(kernel='linear', C = 10 ** -(param))
        scores.append(
            np.mean(
                cross_validation.cross_val_score(
                    classifier,
                    train_lsi,
                    train.target,
                    cv = 5
                )
            )
        )

    return params[scores.index(max(scores))]

if __name__ == "__main__":
    categories = a.fetch_all_categories()
    train = a.fetch_train(categories)
    test = a.fetch_test(categories)

    train_lsi, test_lsi = d.fetch_lsi_representation_catched(
        train,
        test
    )

    e.two_classify_data(train)
    e.two_classify_data(test)

    best_param = fetch_best_parameter(train, train_lsi)
    print "Best Score was obtained for k = ", best_param

    print "Performing SVM classification using k = ", best_param
    classifier = svm.SVC(kernel='linear', C = 10 ** -(best_param))

    classifier = svm.SVC(kernel='linear')
    print "Training SVM classifier"
    classifier.fit(train_lsi, train.target)

    print "Predicting classifications of testing dataset"
    predicted = classifier.predict(test_lsi)

    e.print_statistics(test.target, predicted)
    e.plot_roc(test.target, predicted, 'SVM_CV')


