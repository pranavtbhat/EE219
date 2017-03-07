from sklearn.decomposition import TruncatedSVD
import part1
from sklearn.pipeline import Pipeline
from scipy.sparse import linalg as la
import numpy as np

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

    train = part1.fetch_train(categories)
    test = part1.fetch_test(categories)

    pipeline = Pipeline(
        [
            ('vectorize', part1.get_vectorizer()),
            ('tf-idf', part1.get_tfid_transformer()),
        ]
    )

    X_train = pipeline.fit_transform(train.data)
    X_test  = pipeline.transform(test.data)


    # Observe how many singular values are "userful" and discard the rest

    for k in range(10):
        U, s, Vt = la.svds(X_train, k=10)
        print np.diag(s)

    # pick one!
    k = 5
    svd = TruncatedSVD(n_components=k)
    LSI_train = svd.fit_transform(X_train)
    LSI_test = svd.transform(X_test)

    # Normalize?

    # Non-linear transformation / logarithm transformation

