from sklearn.decomposition import TruncatedSVD
import part1
import part2
from sklearn.cluster import KMeans

from sklearn.pipeline import Pipeline
#from scipy.sparse import linalg as la
from numpy import linalg as la
import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer



if __name__ == "__main__":
    categories=[
        'comp.graphics',
        'comp.os.ms-windows.misc',
        'comp.sys.ibm.pc.hardware',
        'comp.sys.mac.hardware',
        'rec.autos',
        'rec.motorcycles',
        'rec.sport.baseball',
        'rec.sport.hockey'
    ]

    categories = part1.fetch_categories()
    all = part1.fetch_all(categories)
    data_idf = part1.get_data_idf()


    #Uncomment to plot variance vs #components
    # svd = TruncatedSVD(n_components=4000)
    # normalizer = Normalizer(copy=False)
    # lsa = make_pipeline(svd, normalizer)
    #
    # X_lsa = lsa.fit_transform(data_idf)
    # variance = svd.explained_variance_ratio_.cumsum()
    # print variance
    # plt.plot(variance)
    # plt.savefig("plots/variance.png", format='png')
    # plt.clf()
    #
    reduced_dim = 20

    for i in range(2,reduced_dim):

        print "Calculating for dimensions = ", i
        svd = TruncatedSVD(n_components=reduced_dim)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)

        X_lsa = lsa.fit_transform(data_idf)

        labels = all.target // 4  # Since we want to cluster to 2 classes, and the input has 8 classes (0-7)
        kmeans = KMeans(n_clusters=2).fit(data_idf)

        part2.print_confusion_matrix(labels, kmeans.labels_)
        part2.print_scores(labels, kmeans.labels_)


    # Non-linear transformation / logarithm transformation

