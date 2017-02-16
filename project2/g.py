from sklearn.naive_bayes import GaussianNB
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
    classifier = GaussianNB()

    print "Training Naive Bayes classifier"
    classifier.fit(train_lsi, train.target)

    print "Predicting classifications of testing dataset"
    predicted = classifier.predict(test_lsi)

    print "Statistics of Naive Bayes classifiers:"
    e.print_statistics(test.target, predicted)

    e.plot_roc(test.target, predicted, 'Naive_Bayes')
