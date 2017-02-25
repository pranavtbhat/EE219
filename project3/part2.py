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

    for i in range(10):
        train_index = flatten(index_chunks[0:i] + index_chunks[i+1:])
        test_index = index_chunks[i]

        W_train = np.zeros(W.shape)
        W_train[(rows[train_index], cols[train_index])] = 1

        W_test = np.zeros(W.shape)
        W_test[(rows[train_index], cols[train_index])] = 1

        U, V = part1.matrix_factorize(R, W_train, 100)
        in_fold_error = abs_error(R, W_test, U, V)
        out_fold_error = abs_error(R, W, U, V)

        print "Fold ", i + 1
        print "Testing error in fold was : ", in_fold_error
        print "Testing error overall was ", out_fold_error
        print ""



