from sklearn import svm
import sklearn.metrics as smet
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import a
import d


def two_classify_data(data):
    data.target = map(lambda x : int(x < 4), data.target)

def print_statistics(actual, predicted):
    print "Accuracy is ", smet.accuracy_score(actual, predicted) * 100
    print "Precision is ", smet.precision_score(actual, predicted, average='macro') * 100

    print "Recall is ", smet.recall_score(actual, predicted, average='macro') * 100

    print "Confusion Matrix is ", smet.confusion_matrix(actual, predicted)

def plot_roc(actual, predicted, classifier_name):
    x, y, _ = roc_curve(actual, predicted)
    plt.plot(x, y, label="ROC Curve")
    plt.plot([0, 1], [0, 1])

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.2])

    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curves for ' + classifier_name + 'Classifier')
    plt.legend(loc="best")

    plt.savefig('plots/' + classifier_name + '.png', format='png')
    plt.show()


if __name__ == "__main__":

    categories = a.fetch_all_categories()

    train = a.fetch_train(categories)
    test = a.fetch_test(categories)

    two_classify_data(train)
    two_classify_data(test)

    train_lsi, test_lsi = d.fetch_lsi_representation_catched(train, test)
    print "Dataset prepared for SVM"

    classifier = svm.SVC(kernel='linear', probability=True)

    print "Training SVM classifier"
    classifier.fit(train_lsi, train.target)

    print "Predicting classifications of testing dataset"
    predicted = classifier.predict(test_lsi)
    predicted_probs = classifier.predict_proba(test_lsi)

    print "Statistics of SVM classifiers:"
    print_statistics(test.target, predicted)

    plot_roc(test.target, predicted_probs[:,1], 'SVM')
