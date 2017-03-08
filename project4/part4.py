from sklearn.decomposition import TruncatedSVD
import part1
from sklearn.pipeline import Pipeline
from numpy import linalg as la
import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans




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

    reduced_dim = 12

    svd = TruncatedSVD(n_components=reduced_dim)
    normalizer = Normalizer(copy=False)
    pipeline = make_pipeline(svd, normalizer)

    lsa_train = pipeline.fit_transform(X_train)
    kmeans = KMeans(n_clusters=2).fit(lsa_train)

    svd = TruncatedSVD(n_components=2)
    two_dimensional_lsa = svd.fit_transform(lsa_train)

    print two_dimensional_lsa
    print type(two_dimensional_lsa)

    print two_dimensional_lsa[kmeans.labels_ == 0]
    print two_dimensional_lsa[kmeans.labels_ == 1]
    print kmeans.labels_

    print len(two_dimensional_lsa[kmeans.labels_ == 0])
    print len(two_dimensional_lsa[kmeans.labels_ == 1])

    x1 = two_dimensional_lsa[kmeans.labels_ == 0][:, 0]
    y1 = two_dimensional_lsa[kmeans.labels_ == 0][:, 1]
    print x1
    print y1
    plt.plot(x1,y1,'r+')
    x2 = two_dimensional_lsa[kmeans.labels_ == 1][:, 0]
    y2 = two_dimensional_lsa[kmeans.labels_ == 1][:, 1]
    print x2
    print y2
    plt.plot(x2, y2, 'g+')
    plt.savefig("plots/clusters_2d.png", format='png')
    plt.clf()







