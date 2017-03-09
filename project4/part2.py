from sklearn import metrics
from sklearn.cluster import KMeans
import part1

def print_confusion_matrix(actual, predicted):
    print "Confusion Matrix is ", metrics.confusion_matrix(actual, predicted)

def print_scores(actual_labels, predicted_labels):
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(actual_labels, predicted_labels))
    print("Completeness: %0.3f" % metrics.completeness_score(actual_labels, predicted_labels))
    print("Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(actual_labels, predicted_labels))
    print("Adjusted Mutual info score: %.3f" % metrics.adjusted_mutual_info_score(actual_labels, predicted_labels))

if __name__ == "__main__":

    categories = part1.fetch_categories()
    data = part1.fetch_all(categories)
    data_idf = part1.get_data_idf()

    labels = data.target//4 #Since we want to cluster to 2 classes, and the input has 8 classes (0-7)
    kmeans = KMeans(n_clusters=2).fit(data_idf)

    print_confusion_matrix(labels, kmeans.labels_)
    print_scores(labels, kmeans.labels_)



