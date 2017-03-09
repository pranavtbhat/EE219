import part1
import pickle
import os
from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups
from scipy.sparse import linalg as la
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.cluster import KMeans

def get_data_idf(data):
    if os.path.exists('pkl/all_data_idf.pkl'):
        pkl_file = open('pkl/all_data_idf.pkl', 'rb')
        data_idf = pickle.load(pkl_file)
        pkl_file.close()
        return data_idf

    pipeline = Pipeline(
        [
            ('vectorize', part1.get_vectorizer()),
            ('tf-idf', part1.get_tfid_transformer())
        ]
    )

    print("%d documents" % len(all.filenames))
    print("%d categories" % len(all.target_names))

    data_idf = pipeline.fit_transform(data.data)
    print "Number of terms in TF-IDF representation:", data_idf.shape[1]

    output = open('pkl/all_data_idf.pkl', 'wb')
    pickle.dump(data_idf, output)
    output.close()

    return data_idf


def find_singular_values(X):
    print "Finding singular values using svds.."
    if os.path.exists('pkl/all_singular_values_2000.pkl'):
        pkl_file = open('pkl/all_singular_values_2000.pkl', 'rb')
        s = pickle.load(pkl_file)
        pkl_file.close()
        return s

    U, s, Vt = la.svds(X, k=2000)

    output = open('pkl/all_singular_values_2000.pkl', 'wb')
    pickle.dump(s, output)
    output.close()

    return s

def get_scores(actual_labels, predicted_labels):
    t0 = metrics.homogeneity_score(actual_labels, predicted_labels)
    t1 = metrics.completeness_score(actual_labels, predicted_labels)
    t2 = metrics.adjusted_rand_score(actual_labels, predicted_labels)
    t3 = metrics.adjusted_mutual_info_score(actual_labels, predicted_labels)
    return (t0, t1, t2, t3)

if __name__ == "__main__":
    data = fetch_20newsgroups(
        subset='all',
        shuffle=True,
        random_state=42
    )
    labels = data.target
    metric_names = [
        'homogeneity_score',
        'completeness_score',
        'adjusted_rand_score',
        'adjusted_mutual_info_score'
    ]

    data_idf = get_data_idf(data)

    ###
    # First fix k
    ###
    k = 20
    svd_metrics = []
    nmf_metrics = []

    print "Varying Dimensions"
    ds = range(2, 75)
    for d in ds:
        print "Set d = ", d
        # Prepare SVD pipeline
        svd = TruncatedSVD(n_components = d)
        normalizer = Normalizer(copy=False)
        svd_pipeline = make_pipeline(svd, normalizer)

        # Fit data and get metrics for SVD
        X_SVD = svd_pipeline.fit_transform(data_idf)
        kmeans_svd = KMeans(n_clusters=k).fit(X_SVD)
        svd_metrics.append(get_scores(labels, kmeans_svd.labels_))

        # Prepare NMF pipeline
        nmf = NMF(n_components = d)
        normalizer = Normalizer(copy=False)
        nmf_pipeline = make_pipeline(nmf, normalizer)

        # Fit data and get metrics for NMF
        X_NMF = nmf_pipeline.fit_transform(data_idf)
        kmeans_nmf = KMeans(n_clusters=k).fit(X_NMF)
        nmf_metrics.append(get_scores(labels, kmeans_nmf.labels_))

    # Plot SVD metrics
    for i, metric_name in enumerate(metric_names):
        plt.plot(
            ds,
            map(lambda x : x[i], svd_metrics),
            label = metric_name
        )
    plt.xlabel('Number of Dimensions')
    plt.ylabel('Metric value')
    plt.legend(loc='best')
    plt.savefig('plots/part5_fixk_svd_metrics.png', format='png')
    plt.clf()

    # Plot NMF metrics
    for i, metric_name in enumerate(metric_names):
        plt.plot(
            ds,
            map(lambda x : x[i], nmf_metrics),
            label = metric_name
        )
    plt.xlabel('Number of Dimensions')
    plt.ylabel('Metric value')
    plt.legend(loc='best')
    plt.savefig('plots/part5_fixk_nmf_metrics.png', format='png')
    plt.clf()

    ###
    # Next fix d
    ###
    # Find the best dimension size to represent the TF-IDF matrix
    s = find_singular_values(data_idf)
    plt.plot(sorted(s, reverse=True))
    plt.savefig('plots/part5_singular_values.png', format='png')
    plt.clf()

    # The optimumum dimension is obtained by observing the "knee"
    # in the plot of singular values
    reduced_dim = 50
    svd_metrics = []
    nmf_metrics = []

    print "Varying k"
    ks = range(4, 31, 1)
    for k in ks:
        print "Set k = ", k
        # Prepare SVD pipeline
        svd = TruncatedSVD(n_components = reduced_dim )
        normalizer = Normalizer(copy=False)
        svd_pipeline = make_pipeline(svd, normalizer)

        # Fit data and get metrics for SVD
        X_SVD = svd_pipeline.fit_transform(data_idf)
        kmeans_svd = KMeans(n_clusters=k).fit(X_SVD)
        svd_metrics.append(get_scores(labels, kmeans_svd.labels_))

        # Prepare NMF pipeline
        nmf = NMF(n_components = reduced_dim)
        normalizer = Normalizer(copy=False)
        nmf_pipeline = make_pipeline(nmf, normalizer)

        # Fit data and get metrics for NMF
        X_NMF = nmf_pipeline.fit_transform(data_idf)
        kmeans_nmf = KMeans(n_clusters=k).fit(X_NMF)
        nmf_metrics.append(get_scores(labels, kmeans_nmf.labels_))

    # Plot SVD metrics
    for i, metric_name in enumerate(metric_names):
        plt.plot(
            ks,
            map(lambda x : x[i], svd_metrics),
            label = metric_name
        )
    plt.legend(loc='best')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Metric value')
    plt.savefig('plots/part5_fixd_svd_metrics.png', format='png')
    plt.clf()

    # Plot NMF metrics
    for i, metric_name in enumerate(metric_names):
        plt.plot(
            ks,
            map(lambda x : x[i], nmf_metrics),
            label = metric_name
        )
    plt.legend(loc='best')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Metric value')
    plt.savefig('plots/part5_nmf_metrics.png', format='png')
    plt.clf()
