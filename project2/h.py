from sklearn.linear_model import LogisticRegression


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

    classifier = LogisticRegression()

    print "Training Logistic Regression classifier"
    classifier.fit(train_lsi, train.target)

    print "Predicting classifications of testing dataset"
    predicted = classifier.predict(test_lsi)
    predicted_probs = classifier.predict_proba(test_lsi)

    print "Statistics of LogisticRegression classifiers:"
    e.print_statistics(test.target, predicted)

    e.plot_roc(test.target, predicted_probs[:,1], 'LogisticRegression')
