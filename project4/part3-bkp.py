from sklearn.decomposition import TruncatedSVD
import part1
import part2
from sklearn.cluster import KMeans
from sklearn import metrics

from sklearn.pipeline import Pipeline
#from scipy.sparse import linalg as la
from numpy import linalg as la
import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

def save_scores(actual_labels, predicted_labels, h, c , v ,r ):
    h.append(metrics.homogeneity_score(actual_labels, predicted_labels))
    c.append( metrics.completeness_score(actual_labels, predicted_labels))
    v.append( metrics.v_measure_score(actual_labels, predicted_labels))
    r.append( metrics.adjusted_rand_score(actual_labels, predicted_labels))

def print_scores(actual_labels, predicted_labels):
    print("Homogeneity: %0.10f" % metrics.homogeneity_score(actual_labels, predicted_labels))
    print("Completeness: %0.10f" % metrics.completeness_score(actual_labels, predicted_labels))
    print("V-measure: %0.10f" % metrics.v_measure_score(actual_labels, predicted_labels))
    print("Adjusted Rand-Index: %.10f" % metrics.adjusted_rand_score(actual_labels, predicted_labels))

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
    reduced_dim = 100

    h = []
    c = []
    v = []
    r = []
    ks = []

    for i in range(20,21):

        print "Calculating for dimensions = ", i
        svd = TruncatedSVD(n_components=i)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)

        X_lsa = lsa.fit_transform(data_idf)

        labels = all.target // 4  # Since we want to cluster to 2 classes, and the input has 8 classes (0-7)
        kmeans = KMeans(n_clusters=2).fit(X_lsa)

        svd = TruncatedSVD(n_components=2)
        reduced_X=svd.fit_transform(X_lsa)

        x1=[]
        y1=[]
        x2 = []
        y2 = []
        x3 = []
        y3 = []
        x4 = []
        y4 = []
        for j in range(len(labels)):
            if(kmeans.labels_[j]==0 and labels[j]==0):
                x1.append(reduced_X[j, 0])
                y1.append(reduced_X[j, 1])
            elif (kmeans.labels_[j] == 0 and labels[j] == 1):
                x2.append(reduced_X[j, 0])
                y2.append(reduced_X[j, 1])
            # elif (kmeans.labels_[j] == 1 and labels[j] == 0):
            #     x3.append(reduced_X[j, 0])
            #     y3.append(reduced_X[j, 1])
            # elif (kmeans.labels_[j] == 1 and labels[j] == 1):
            #     x4.append(reduced_X[j, 0])
            #     y4.append(reduced_X[j, 1])


        plt.plot(x1,y1,'k+')
        plt.plot(x2, y2, 'g+')
        plt.plot(x3, y3, 'b+')
        plt.plot(x4, y4, 'y+')

        part2.print_confusion_matrix(labels, kmeans.labels_)
        # print_scores(labels,kmeans.labels_)
        # save_scores(labels, kmeans.labels_, h, c, v, r);
        # ks.append(i)

        plt.savefig("plots/debugging.png", format='png')
        plt.clf()

    # plt.plot(ks, h, label='homogenity')
    # plt.plot(ks, c, label='completeness')
    # plt.plot(ks, v, label='normalized mutual info')
    # plt.plot(ks, r, label='rand score')
    # plt.grid()
    # plt.legend(fontsize = 'xx-small')
    # plt.savefig("plots/part3.png", format='png')
    # plt.clf()

    # Non-linear transformation / logarithm transformation

