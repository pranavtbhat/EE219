import part5
import part2
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

if __name__ == "__main__":

    data_idf = part5.get_data_idf()

    # s = part5.find_singular_values(data_idf)
    # var = part5.find_truncated_svd_variance(data_idf)

    # Selected appropriate reduced dimension using singular values
    reduced_dim = 20

    svd = TruncatedSVD(n_components=reduced_dim)
    normalizer = Normalizer(copy=False)
    pipeline = make_pipeline(svd, normalizer)
    data_lsa = pipeline.fit_transform(data_idf)

    print "Performing Kmeans clustering.."
    # Select appropriate k - one which gives best scores. And convert labels accordingly!!!

    k = 6
    kmeans = KMeans(n_clusters=k).fit(data_lsa)

    labels = fetch_20newsgroups(subset='all', shuffle=True, random_state=42).target

    part2.print_confusion_matrix(labels, kmeans.labels_)
    part2.print_scores(labels, kmeans.labels_)
