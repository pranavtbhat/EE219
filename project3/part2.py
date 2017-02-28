import numpy as np
import part1

def chunks(arr, n):
    chunk_size = len(arr) / n
    return [arr[i * chunk_size : min((i + 1) * chunk_size, len(arr))] for i in xrange(n)]

def flatten(arr):
    return np.array([x for sub_arr in arr for x in sub_arr])

def abs_error(R, W, U, V):
    return np.sum(W * np.abs(R - np.dot(U, V))) / np.count_nonzero(W)

if __name__ == "__main__":
    R, W = part1.load_dataset()

    # Get 10 sets of indices from a random permutation of
    # the non zero indices of W
    rows, cols = np.nonzero(W)

    index_chunks = chunks(
        np.random.permutation(np.array(range(len(rows)))),
        10
    )

    for k in [10, 50, 100]:
        print "Setting k = ", k
        errors = []
        for i in range(10):
            train_index = flatten(index_chunks[0:i] + index_chunks[i+1:])
            test_index = index_chunks[i]

            W_train = np.zeros(W.shape)
            W_train[(rows[train_index], cols[train_index])] = 1

            W_test = np.zeros(W.shape)
            W_test[(rows[test_index], cols[test_index])] = 1

            U, V = part1.matrix_factorize(R, W_train, k)
            error = abs_error(R, W_test, U, V)
            errors.append(error)

        print "Maximum error accross folds was ", max(errors)
        print "Minimum error accross folds was ", min(errors)
        print "Average error across folds was ", np.mean(errors)



