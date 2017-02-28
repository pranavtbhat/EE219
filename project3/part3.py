import numpy as np
import part1
import part2
from matplotlib import pyplot as plt

def get_pr(R, W, U, V, th):
    actual = W * R
    predicted = W * np.dot(U, V)

    actual = np.greater_equal(actual, th)
    predicted = np.greater_equal(predicted, th)

    true_positives = np.count_nonzero(
        np.logical_and(actual, predicted)
    ) + 0.0
    false_positives = np.count_nonzero(
        np.logical_and(predicted, np.logical_not(actual))
    ) + 0.0
    false_negatives = np.count_nonzero(
        np.logical_and(np.logical_not(predicted),actual)
    ) + 0.0

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    return precision, recall

if __name__ == "__main__":
    R, W = part1.load_dataset()

    # Get 10 sets of indices from a random permutation of
    # the non zero indices of W
    rows, cols = np.nonzero(W)

    index_chunks = part2.chunks(
        np.random.permutation(np.array(range(len(rows)))),
        10
    )

    for k in [10, 50, 100]:
        print "Setting k = ", k
        thresholds = range(1, 6)
        precisions = np.zeros((5))
        recalls = np.zeros((5))

        for i in range(10):
            print "Fold: ", i + 1
            train_index = part2.flatten(index_chunks[0:i] + index_chunks[i+1:])
            test_index = index_chunks[i]

            W_train = np.zeros(W.shape)
            W_train[(rows[train_index], cols[train_index])] = 1

            W_test = np.zeros(W.shape)
            W_test[(rows[test_index], cols[test_index])] = 1

            U, V = part1.matrix_factorize(R, W_train, k)

            for index,th in enumerate(thresholds):
                precision, recall = get_pr(R, W_test, U, V, th)
                precisions[index] += precision
                recalls[index] += recall

        print "Precisions: ", precisions / 10
        print "Recalls: ", recalls / 10

        plt.ylabel('Recall')
        plt.xlabel('Precision')
        plt.title('ROC Curve')
        plt.scatter(precisions, recalls, s=30, marker='o')
        plt.plot(precisions,recalls)
        plt.savefig("plots/ROCk" + str(k) + ".png", format='png')
        plt.clf()

