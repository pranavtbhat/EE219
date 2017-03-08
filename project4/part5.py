import part1
import part2
import pickle
import os
from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups
from scipy.sparse import linalg as la
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans



def get_data_idf():

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

    all = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)

    print("%d documents" % len(all.filenames))
    print("%d categories" % len(all.target_names))

    data_idf = pipeline.fit_transform(all.data)
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

def find_truncated_svd_variance(X):

    print "Finding Truncated SVD variance.."

    if os.path.exists('pkl/all_truncated_variance.pkl'):
        pkl_file = open('pkl/all_truncated_variance.pkl', 'rb')
        var = pickle.load(pkl_file)
        pkl_file.close()
        return var

    svd = TruncatedSVD(n_components=2000)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X_lsa = lsa.fit_transform(X)
    var = svd.explained_variance_ratio_.cumsum()

    output = open('pkl/all_truncated_variance.pkl', 'wb')
    pickle.dump(var, output)
    output.close()

    plt.plot(var)
    plt.savefig("plots/all_truncated_variance.png", format='png')
    plt.clf()

    return var


def save_scores(actual_labels, predicted_labels, h, c , v ,r ):
    h.append(metrics.homogeneity_score(actual_labels, predicted_labels))
    c.append( metrics.completeness_score(actual_labels, predicted_labels))
    v.append( metrics.v_measure_score(actual_labels, predicted_labels))
    r.append( metrics.adjusted_rand_score(actual_labels, predicted_labels))



if __name__ == "__main__":

    data_idf = get_data_idf()

    # Uncomment below to find reduced dim
    # s = find_singular_values(get_data_idf())
    # var = find_truncated_svd_variance(get_data_idf())

    #Selected appropriate reduced dimension using singular values
    reduced_dim = 50

    svd = TruncatedSVD(n_components=reduced_dim)
    normalizer = Normalizer(copy=False)
    pipeline = make_pipeline(svd, normalizer)
    data_lsa = pipeline.fit_transform(data_idf)

    print "Performing Kmeans clustering.."
    #Select appropriate k - one which gives best scores. And convert labels accordingly

    h=[]
    c=[]
    v=[]
    r=[]
    ks=[]

    for k in range(6,100,1):

        print "#clusters = ", k
        kmeans = KMeans(n_clusters=k).fit(data_lsa)

        labels = fetch_20newsgroups(subset='all', shuffle=True, random_state=42).target

        save_scores(labels, kmeans.labels_, h,c,v,r);
        ks.append(k)

    plt.plot(ks, h, label='homogenity')
    plt.plot(ks, c, label='completeness')
    plt.plot(ks, v, label='normalized mutual info')
    plt.plot(ks, r, label='rand score')
    plt.grid()
    plt.savefig("plots/part5.png", format='png')
    plt.clf()

