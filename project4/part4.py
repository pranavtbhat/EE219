from sklearn.decomposition import TruncatedSVD
import part1
import part2
from sklearn.pipeline import Pipeline
from numpy import linalg as la
import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD




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

    train = part1.fetch_all(categories)

    pipeline = Pipeline(
        [
            ('vectorize', part1.get_vectorizer()),
            ('tf-idf', part1.get_tfid_transformer()),
        ]
    )

    X_train = pipeline.fit_transform(train.data)

    #Obtained in part3
    reduced_dim = 40

    svd = TruncatedSVD(n_components=reduced_dim)
    normalizer = Normalizer(copy=False)
    pipeline = make_pipeline(svd, normalizer)

    lsa_train = pipeline.fit_transform(X_train)
    kmeans = KMeans(n_clusters=2).fit(lsa_train)


    svd = TruncatedSVD(n_components=2)
    two_dimensional_lsa = svd.fit_transform(lsa_train)

    x1 = two_dimensional_lsa[kmeans.labels_ == 0][:, 0]
    y1 = two_dimensional_lsa[kmeans.labels_ == 0][:, 1]

    plt.plot(x1,y1,'r+')
    x2 = two_dimensional_lsa[kmeans.labels_ == 1][:, 0]
    y2 = two_dimensional_lsa[kmeans.labels_ == 1][:, 1]

    plt.plot(x2, y2, 'g+')
    plt.savefig("plots/clusters_2d.png", format='png')
    plt.clf()

    labels = all.target // 4
    svd = TruncatedSVD(n_components=2)
    reduced_X = svd.fit_transform(two_dimensional_lsa)

    x1 = []
    y1 = []
    x2 = []
    y2 = []
    x3 = []
    y3 = []
    x4 = []
    y4 = []
    for j in range(len(labels)):
        if (kmeans.labels_[j] == 0 and labels[j] == 0):
            x1.append(reduced_X[j, 0])
            y1.append(reduced_X[j, 1])
        elif (kmeans.labels_[j] == 0 and labels[j] == 1):
            x2.append(reduced_X[j, 0])
            y2.append(reduced_X[j, 1])
        elif (kmeans.labels_[j] == 1 and labels[j] == 0):
            x3.append(reduced_X[j, 0])
            y3.append(reduced_X[j, 1])
        elif (kmeans.labels_[j] == 1 and labels[j] == 1):
            x4.append(reduced_X[j, 0])
            y4.append(reduced_X[j, 1])

    plt.plot(x1, y1, 'r+')
    plt.plot(x2, y2, 'yo')
    plt.plot(x3, y3, 'bo')
    plt.plot(x4, y4, 'g+')

    part2.print_confusion_matrix(labels, kmeans.labels_)
    plt.savefig("plots/debugging.png", format='png')
    plt.clf()







