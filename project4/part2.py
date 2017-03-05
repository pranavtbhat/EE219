from sklearn import metrics
from sklearn.cluster import KMeans
import part1

def print_statistics(actual, predicted):
    print "Accuracy is ", metrics.accuracy_score(actual, predicted) * 100
    print "Precision is ", metrics.precision_score(actual, predicted, average='macro') * 100
    print "Recall is ", metrics.recall_score(actual, predicted, average='macro') * 100
    print "Confusion Matrix is ", metrics.confusion_matrix(actual, predicted)
    
if __name__ == "__main__":
    
    categories = part1.fetch_all_categories()
    train = part1.fetch_train(categories)
    train_idf = part1.get_train_idf()
    
    labels = train.target
    kmeans = KMeans(n_clusters=2).fit(train_idf)
    print_statistics(labels, kmeans.labels_)
    
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, kmeans.labels_))
    print("Completeness: %0.3f" % metrics.completeness_score(labels, kmeans.labels_))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels, kmeans.labels_))
    print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, kmeans.labels_))