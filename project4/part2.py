from sklearn import metrics
from sklearn.cluster import KMeans
import numpy as np
import part1

from sklearn.decomposition import TruncatedSVD
import scipy.sparse as sp
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

def print_statistics(actual, predicted):
    print "Confusion Matrix is ", metrics.confusion_matrix(actual, predicted)

if __name__ == "__main__":

    categories = part1.fetch_categories()
    data = part1.fetch_all(categories)
    data_idf = part1.get_data_idf()

    labels = data.target//4 #Since we want to cluster to 2 classes, and the input has 8 classes (0-7)
    kmeans = KMeans(n_clusters=2).fit(data_idf)

    print_statistics(labels, kmeans.labels_)

    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, kmeans.labels_))
    print("Completeness: %0.3f" % metrics.completeness_score(labels, kmeans.labels_))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels, kmeans.labels_))
    print("Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(labels, kmeans.labels_))

