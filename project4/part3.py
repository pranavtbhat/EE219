from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
import part1
import part2
from sklearn.cluster import KMeans
from sklearn import metrics

from sklearn.pipeline import Pipeline
from scipy.sparse import linalg as la
#from numpy import linalg as la
import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

def save_scores(actual_labels, predicted_labels, h, c , v ,r ):
    h.append(metrics.homogeneity_score(actual_labels, predicted_labels))
    c.append(metrics.completeness_score(actual_labels, predicted_labels))
    v.append(metrics.v_measure_score(actual_labels, predicted_labels))
    r.append(metrics.adjusted_rand_score(actual_labels, predicted_labels))

def calculate_scores_for_svd(data_idf, start, end):
    h = []
    c = []
    v = []
    r = []
    ks = []

    for n in range(start, end+1, 1):
        print "Calculating for dimensions = ", n
        svd = TruncatedSVD(n_components=n)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)

        X_lsa = lsa.fit_transform(data_idf)

        labels = all.target // 4  # Since we want to cluster to 2 classes, and the input has 8 classes (0-7)
        kmeans = KMeans(n_clusters=2).fit(X_lsa)

        print metrics.confusion_matrix(labels, kmeans.labels_)
        save_scores(labels, kmeans.labels_, h, c , v ,r )
        ks.append(n)

    plt.plot(ks, h, 'r', lw=6, label='homogenity')
    plt.plot(ks, c, 'y', lw=4, label='completeness')
    plt.plot(ks, v, 'k', lw=2, label='normalized mutual score')
    plt.plot(ks, r, label='rand score')
    plt.grid()
    plt.legend(fontsize='xx-small')
    plt.savefig("plots/scores_svd_part3.png", format='png')
    plt.clf()

def calculate_scores_for_nmf(data_idf, start, end):

    h = []
    c = []
    v = []
    r = []
    ks = []

    for n in range(start, end+1, 1):
        print "Calculating NMF for dimensions = ", n
        nmf = NMF(n_components=n)
        #normalizer = Normalizer(copy=False)
        lsa = make_pipeline(nmf)

        X_lsa = lsa.fit_transform(data_idf)

        labels = all.target // 4  # Since we want to cluster to 2 classes, and the input has 8 classes (0-7)
        kmeans = KMeans(n_clusters=2).fit(X_lsa)

        print metrics.confusion_matrix(labels, kmeans.labels_)
        save_scores(labels, kmeans.labels_, h, c , v ,r )
        ks.append(n)

    plt.plot(ks, h, 'r', lw=6, label='homogenity')
    plt.plot(ks, c, 'y', lw=4, label='completeness')
    plt.plot(ks, v, 'k', lw=2, label='normalized mutual score')
    plt.plot(ks, r, label='rand score')
    plt.grid()
    plt.legend(fontsize='xx-small')
    plt.savefig("plots/scores_nmf_part3.png", format='png')
    plt.clf()


def non_linear_transformations(data_idf, reduced_dim):

    print "Calculating for degree 2 polynomial.."
    svd = TruncatedSVD(n_components=reduced_dim)
    poly = PolynomialFeatures(degree=2)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, poly, normalizer)

    X_lsa = lsa.fit_transform(data_idf)

    labels = all.target // 4  # Since we want to cluster to 2 classes, and the input has 8 classes (0-7)
    kmeans = KMeans(n_clusters=2).fit(X_lsa)

    print metrics.confusion_matrix(labels, kmeans.labels_)
    print("Homogeneity: %0.10f" % metrics.homogeneity_score(labels, kmeans.labels_))
    print("Completeness: %0.10f" % metrics.completeness_score(labels, kmeans.labels_))
    print("V-measure: %0.10f" % metrics.v_measure_score(labels, kmeans.labels_))
    print("Adjusted Rand-Index: %.10f" % metrics.adjusted_rand_score(labels, kmeans.labels_))

    print "Calculating for log features.."
    svd = TruncatedSVD(n_components=reduced_dim)
    poly = FunctionTransformer(np.log1p)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, poly, normalizer)

    X_lsa = lsa.fit_transform(data_idf)

    labels = all.target // 4  # Since we want to cluster to 2 classes, and the input has 8 classes (0-7)
    kmeans = KMeans(n_clusters=2).fit(X_lsa)

    print metrics.confusion_matrix(labels, kmeans.labels_)
    print("Homogeneity: %0.10f" % metrics.homogeneity_score(labels, kmeans.labels_))
    print("Completeness: %0.10f" % metrics.completeness_score(labels, kmeans.labels_))
    print("V-measure: %0.10f" % metrics.v_measure_score(labels, kmeans.labels_))
    print("Adjusted Rand-Index: %.10f" % metrics.adjusted_rand_score(labels, kmeans.labels_))


if __name__ == "__main__":

    categories = part1.fetch_categories()
    all = part1.fetch_all(categories)
    data_idf = part1.get_data_idf()

    U, s, Vt = la.svds(data_idf, k=100)
    print type(s)
    plt.plot(s[::-1])
    plt.grid()
    plt.xlabel('index')
    plt.ylabel('singular value')
    plt.savefig("plots/singular_values_part3.png", format='png')
    plt.clf()

    #elbow at 4, hence checking around that range

    calculate_scores_for_svd(data_idf, 2,80)
    calculate_scores_for_nmf(data_idf, 2,80)
    non_linear_transformations(data_idf, 40)

    print "Plotting TF-IDF"

    labels = all.target // 4
    svd = TruncatedSVD(n_components=2)
    reduced_tfidf = svd.fit_transform(data_idf)
    reduced_tfidf_log = np.log1p(reduced_tfidf)
    x1 = reduced_tfidf_log[labels == 0][:, 0]
    y1 = reduced_tfidf_log[labels == 0][:, 1]
    x2 = reduced_tfidf_log[labels == 1][:, 0]
    y2 = reduced_tfidf_log[labels == 1][:, 1]
    plt.plot(x1,y1, 'r+')
    plt.plot(x2,y2,'g+')
    plt.savefig("plots/tf_idf_log.png", format='png')
    plt.clf()


