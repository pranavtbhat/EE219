import sklearn.metrics as smet
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

import a
import e
import d


if __name__ == "__main__":
    categories = a.fetch_all_categories()

    train = a.fetch_train(categories)
    test = a.fetch_test(categories)

    e.two_classify_data(train)
    e.two_classify_data(test)

    train_lsi, test_lsi = d.fetch_lsi_representation_catched(train, test)

    params = list(range(-3, 4))
    l1_accuracies = []
    l2_accuracies = []

    l1_coef = []
    l2_coef = []

    for param in params:
        l1_classifier = LogisticRegression(
            penalty = 'l1',
            C = 10 ** param,
            solver = 'liblinear'
        )
        l1_classifier.fit(train_lsi, train.target)
        l1_predicted = l1_classifier.predict(test_lsi)
        l1_accuracies.append(
            100 - smet.accuracy_score(test.target, l1_predicted) * 100
        )
        l1_coef.append(np.mean(l1_classifier.coef_))


        l2_classifier = LogisticRegression(
            penalty = 'l2',
            C = 10 ** param,
            solver = 'liblinear'
        )
        l2_classifier.fit(train_lsi, train.target)
        l2_predicted = l2_classifier.predict(test_lsi)
        l2_accuracies.append(
            100 - smet.accuracy_score(test.target, l2_predicted) * 100
        )
        l2_coef.append(np.mean(l2_classifier.coef_))


    for i, param in enumerate(params):
        print "Regularization parameter set to ", param
        print "Accuracy with L1 Regularization is ", l1_accuracies[i]
        print "Mean of coefficients is ", l1_coef[i]

        print "Accuracy with L2 Regularization is ", l2_accuracies[i]
        print "Mean of coefficients is ", l2_coef[i]
        print ""


    plt.plot(l1_accuracies)
    plt.xlabel('Regularization Parameter')
    plt.ylabel('Testing error')
    plt.xticks(range(6), [10 ** param for param in params])
    plt.title("Accuracy of L1 Regularized LogisticRegression against the regularization parameter")
    plt.savefig("plots/i-l1.png", format="png")
    plt.show()
    plt.clf()

    plt.plot(l2_accuracies)
    plt.xlabel('Regularization Parameter')
    plt.ylabel('Testing error')
    plt.xticks(range(6), [10 ** param for param in params])
    plt.title("Accuracy of L2 Regularized LogisticRegression against the regularization parameter")
    plt.savefig("plots/i-l2.png", format="png")
    plt.show()
