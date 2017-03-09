import part5
import part2
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import FunctionTransformer
import matplotlib.pyplot as plt
import numpy as np

computer_technologies = [
    'comp.graphics',
    'comp.os.ms-windows.misc',
    'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware',
    'comp.windows.x'
]

recreational_activity = [
    'rec.autos',
    'rec.motorcycles',
    'rec.sport.baseball',
    'rec.sport.hockey'
]

science = [
    'sci.crypt',
    'sci.electronics',
    'sci.med',
    'sci.space'
]

miscellaneus = [ 'misc.forsale' ]

politics = [
    'talk.politics.misc',
    'talk.politics.guns',
    'talk.politics.mideast'
]

religion = [
    'talk.religion.misc',
    'alt.atheism',
    'soc.religion.christian'
]

if __name__ == "__main__":
    # Relabelling and stuff
    classes = [computer_technologies, recreational_activity, science, miscellaneus, politics, religion]
    all_categories = []
    i = 0
    rmap = {}
    for cnum,c in enumerate(classes):
        for category in c:
            all_categories.append(category)
            rmap[i] = cnum
            i += 1

    data = fetch_20newsgroups(
        subset = 'all',
        shuffle = True,
        random_state = 42
    )
    data.target = map(lambda x : rmap[x], data.target)
    data_idf = part5.get_data_idf(data)

    # Find Effective dimensions to retrieve data
    k = 6
    ds = range(2, 75, 1)
    svd_metrics = []
    print "Varying Dimensions"
    for d in ds:
        print "Set d = ", d
        svd = TruncatedSVD(n_components = d)
        poly = FunctionTransformer(np.log1p)
        normalizer = Normalizer(copy=False)
        svd_pipeline = make_pipeline(svd, poly, normalizer)
        X_SVD = svd_pipeline.fit_transform(data_idf)
        kmeans = KMeans(n_clusters = k).fit(X_SVD)
        svd_metrics.append(part5.get_scores(data.target, kmeans.labels_))

        part2.print_confusion_matrix(data.target, kmeans.labels_)
        part2.print_scores(data.target, kmeans.labels_)

    metric_names = [
        'homogeneity_score',
        'completeness_score',
        'adjusted_rand_score',
        'adjusted_mutual_info_score'
    ]
    for i, metric_name in enumerate(metric_names):
        plt.plot(
            ds,
            map(lambda x : x[i], svd_metrics),
            label = metric_name
        )
    plt.legend(loc='best')
    plt.xlabel('Number of Dimensions')
    plt.ylabel('Metric value')
    plt.savefig('plots/part6.png', format='png')
    plt.clf()


